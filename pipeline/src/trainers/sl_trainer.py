from __future__ import annotations
import time
from typing import Dict, Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.core.trainer_base import BaseTrainer
from src.core.registry import register
from tqdm import tqdm


@register("trainer", "sl")
class SupervisedTrainer(BaseTrainer):
    required_components = ["dataset", "model"]

    def __init__(self, cfg, **components):
        super().__init__(cfg, **components)
        self.model: nn.Module = self.components["model"].to(self.device)

        # channels-last for throughput
        if self.device.type == "cuda" and bool(getattr(getattr(cfg, "speed", {}), "channels_last", True)):
            self.model.to(memory_format=torch.channels_last)

        # Use centrally merged trainer config from builder semantics
        try:
            from omegaconf import OmegaConf
            # Collect only leaf nodes that have a 'name' key
            def _collect(c):
                nodes = []
                base = getattr(c, "trainer", None)
                if base is not None and getattr(base, "name", None) is not None:
                    nodes.append(base)
                inner = getattr(getattr(c, "trainer", None), "trainer", None)
                if inner is not None and getattr(inner, "name", None) is not None:
                    nodes.append(inner)
                exp = getattr(c, "exp", None)
                if exp is not None:
                    node = getattr(exp, "trainer", None)
                    if node is not None and getattr(node, "name", None) is not None:
                        nodes.append(node)
                    node2 = getattr(node, "trainer", None) if node is not None else None
                    if node2 is not None and getattr(node2, "name", None) is not None:
                        nodes.append(node2)
                    exp2 = getattr(exp, "exp", None)
                    if exp2 is not None:
                        node3 = getattr(exp2, "trainer", None)
                        if node3 is not None and getattr(node3, "name", None) is not None:
                            nodes.append(node3)
                        node4 = getattr(node3, "trainer", None) if node3 is not None else None
                        if node4 is not None and getattr(node4, "name", None) is not None:
                            nodes.append(node4)
                return nodes

            nodes = _collect(cfg)
            tr_cfg = nodes[0] if nodes else getattr(cfg, "trainer", {})
            for n in nodes[1:]:
                tr_cfg = OmegaConf.merge(tr_cfg, n)
            self.tr_cfg = tr_cfg
        except Exception:
            self.tr_cfg = getattr(cfg, "trainer", {})
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(getattr(self.tr_cfg, "label_smoothing", 0.0)))

        # Build optimizer from trainer.optimizer via registry
        from src.core.registry import get
        opt_cfg = getattr(self.tr_cfg, "optimizer", None)

        if opt_cfg is None or getattr(opt_cfg, "name", None) is None:
            raise KeyError("Missing trainer.optimizer.name in config")

        opt_factory = get("optimizer", opt_cfg.name)

        if hasattr(opt_factory, "build"):
            self.optimizer = opt_factory().build(opt_cfg, {"model": self.model})
        else:
            self.optimizer = opt_factory(opt_cfg, {"model": self.model})

    def _make_loaders(self) -> Dict[str, DataLoader]:
        ds = self.components["dataset"]
        loader_cfg = getattr(self.tr_cfg, "loader", None)
        batch_size = int(getattr(self.tr_cfg, "batch_size", 256))
        num_workers = int(getattr(loader_cfg, "num_workers", 8)) if loader_cfg else 8
        pin_memory = (self.device.type == "cuda") and (bool(getattr(loader_cfg, "pin_memory", True)) if loader_cfg else True)

        # persistent_workers must be False when num_workers == 0
        persistent_workers = (num_workers > 0) and (bool(getattr(loader_cfg, "persistent_workers", True)) if loader_cfg else True)
        prefetch_factor = int(getattr(loader_cfg, "prefetch_factor", 4)) if loader_cfg else 4

        def mk(dl_ds, shuffle: bool):
            kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
            }
            if num_workers > 0:
                kwargs["prefetch_factor"] = prefetch_factor
            return DataLoader(dl_ds, **kwargs)
            
        train_loader = mk(ds["train"], True)
        valid_loader = mk(ds["valid"], False)
        return {"train": train_loader, "valid": valid_loader}

    def fit(self):
        loaders = self._make_loaders()
        epochs = int(getattr(self.tr_cfg, "epochs", 100))
        use_amp, amp_dtype, scaler = self.get_amp_settings()
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = bool(getattr(getattr(self.cfg, "speed", {}), "cudnn_benchmark", True))

        steps_per_epoch = max(1, len(loaders["train"]))
        total_steps = steps_per_epoch * epochs
        scheduler = self.build_lr_scheduler(self.optimizer, total_steps)

        log_interval = int(getattr(getattr(self.cfg, "logger", {}), "log_interval", 100))

        try:
            pbar_outer = tqdm(total=epochs, desc="Epochs", leave=True)
        except Exception:
            pbar_outer = None

        self.model.train()
        global_step = 0
        try:
            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()
                running_loss = 0.0
                running_correct = 0
                running_total = 0
                running_imgs = 0
                try:
                    pbar_inner = tqdm(total=len(loaders["train"]), desc=f"Train {epoch}/{epochs}", leave=False)
                except Exception:
                    pbar_inner = None

                for batch_idx, batch in enumerate(loaders["train"], start=1):
                    batch = self.to_device(batch)
                    images = batch["images"]
                    targets = batch["targets"]
                    if use_amp:
                        with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                            outputs = self.model(images)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)

                    self.optimizer.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        self.clip_grad(self.model.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 0.0)))
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.clip_grad(self.model.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 0.0)))
                        self.optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    running_correct += (preds == targets).sum().item()
                    running_total += images.size(0)
                    running_imgs += images.size(0)

                    if scheduler is not None:
                        scheduler.step()

                    global_step += 1
                    if pbar_inner is not None:
                        try:
                            pbar_inner.update(1)
                            pbar_inner.set_postfix({
                                "loss": f"{loss.item():.3f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            })
                        except Exception:
                            pass
                    if global_step % log_interval == 0:
                        self.logger.log({
                            "step": global_step,
                            "epoch": epoch,
                            "train/step_loss": float(loss.item()),
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                        })

                if pbar_inner is not None:
                    try:
                        pbar_inner.close()
                    except Exception:
                        pass

                epoch_time = time.perf_counter() - t0
                train_loss = running_loss / max(1, running_total)
                train_acc = running_correct / max(1, running_total)
                imgs_per_sec = running_imgs / max(1e-9, epoch_time)

                # validation
                self.model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch in loaders["valid"]:
                        batch = self.to_device(batch)
                        images = batch["images"]
                        targets = batch["targets"]
                        outputs = self.model(images)
                        preds = outputs.argmax(dim=1)
                        val_correct += (preds == targets).sum().item()
                        val_total += images.size(0)
                val_acc = val_correct / max(1, val_total)
                self.model.train()

                self.logger.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "valid/acc": val_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "speed/imgs_per_sec": imgs_per_sec,
                    "time/epoch_sec": epoch_time,
                })

                # save best by val acc
                self.save_checkpoint(step=epoch, objective=val_acc, extra={"val_acc": val_acc})

                if pbar_outer is not None:
                    try:
                        pbar_outer.update(1)
                        pbar_outer.set_postfix({"train_acc": f"{train_acc:.3f}", "val_acc": f"{val_acc:.3f}"})
                    except Exception:
                        pass
        finally:
            if pbar_outer is not None:
                try:
                    pbar_outer.close()
                except Exception:
                    pass
            try:
                self.logger.finish()
            except Exception:
                pass


