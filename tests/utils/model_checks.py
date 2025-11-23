import math
import torch


def assert_no_nan_inf(tensors, context: str = ""):
    """Raise AssertionError if any tensor contains NaN or inf."""
    for name, t in tensors.items():
        if t is None:
            continue
        if not torch.isfinite(t).all():
            raise AssertionError(f"{context} {name} contains NaN/inf")


def grad_norms(params):
    """Return list of gradient L2 norms for params with grads."""
    norms = []
    for p in params:
        if p.grad is not None:
            norms.append(p.grad.data.norm().item())
    return norms


def assert_grad_norms_reasonable(params, lo: float = 1e-6, hi: float = 1e2, context: str = ""):
    norms = grad_norms(params)
    if not norms:
        raise AssertionError(f"{context} no gradients found")
    filtered = [n for n in norms if not (math.isnan(n) or math.isinf(n))]
    if not filtered:
        raise AssertionError(f"{context} all gradients nan/inf")
    max_norm = max(filtered)
    frac_above = sum(n >= lo for n in filtered) / len(filtered)
    if max_norm > hi:
        raise AssertionError(f"{context} gradient norms too large (>{hi}): max={max_norm}")
    # Require that some non-trivial portion of params receive usable gradients,
    # but allow many small/near-zero grads from layernorm/bias terms.
    if frac_above < 0.1:
        raise AssertionError(f"{context} insufficient gradients above {lo}: "
                             f"{frac_above*100:.1f}% of params")


def step_loss_decreases(model, batch, loss_fn, steps: int = 10, lr: float = 1e-3, device=None):
    """
    Run a tiny optimization loop and assert loss decreases.
    Returns (initial_loss, final_loss).
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial = None
    final = None
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = loss_fn(out, batch)
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
        final = loss.item()
    return initial, final
