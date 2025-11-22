import argparse
import logging
import random
import sys
import time
import math

import carla
from agents.navigation.behavior_agent import BehaviorAgent


def distance_sq(a: carla.Location, b: carla.Location) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def pick_far_spawn_pair(spawn_points):
    if len(spawn_points) < 2:
        raise RuntimeError("Not enough spawn points to create a route.")
    start = random.choice(spawn_points)
    end = max(spawn_points, key=lambda sp: distance_sq(sp.location, start.location))
    if start == end:
        # fallback: pick another random different point
        candidates = [sp for sp in spawn_points if sp != start]
        end = random.choice(candidates)
    return start, end


def set_sync(world: carla.World, tm: carla.TrafficManager, enable: bool, fixed_delta: float = 0.05):
    settings = world.get_settings()
    settings.synchronous_mode = enable
    settings.fixed_delta_seconds = fixed_delta if enable else None
    world.apply_settings(settings)
    tm.set_synchronous_mode(enable)


def spawn_ego_model3(world: carla.World, transform: carla.Transform) -> carla.Vehicle:
    bp_lib = world.get_blueprint_library()
    model3_bp = bp_lib.find("vehicle.tesla.model3")
    model3_bp.set_attribute("role_name", "hero")
    vehicle = world.try_spawn_actor(model3_bp, transform)
    if vehicle is None:
        raise RuntimeError("Failed to spawn Tesla Model 3 at the selected spawn point.")
    return vehicle


def spawn_traffic(client: carla.Client, world: carla.World, tm_port: int, count: int) -> list:
    bp_lib = world.get_blueprint_library()
    vehicle_blueprints = bp_lib.filter("vehicle.*")
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)
    vehicles = []
    for idx in range(min(count, len(spawn_points))):
        bp = random.choice(vehicle_blueprints)
        # prefer drivers
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        vehicle = world.try_spawn_actor(bp, spawn_points[idx])
        if vehicle is not None:
            vehicle.set_autopilot(True, tm_port)
            vehicles.append(vehicle)
    return vehicles


def spawn_walkers(world: carla.World, count: int):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")

    walkers = []
    controllers = []

    for _ in range(count):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        if walker_bp.has_attribute("speed"):
            # choose a walking speed
            speeds = walker_bp.get_attribute("speed").recommended_values
            walker_bp.set_attribute("speed", random.choice(speeds))

        walker = world.try_spawn_actor(walker_bp, carla.Transform(loc))
        if walker is None:
            continue
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
        if controller is None:
            walker.destroy()
            continue
        controllers.append(controller)
        walkers.append(walker)

    # start controllers
    for controller in controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.4 + random.random())  # ~1.4-2.4 m/s

    return walkers, controllers


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_global_distance_to_leading_vehicle(1.0)
    tm.set_hybrid_physics_mode(True)
    tm.global_percentage_speed_difference(10.0)  # NPCs slightly slower

    original_settings = world.get_settings()
    original_sync = original_settings.synchronous_mode

    vehicles = []
    walkers = []
    walker_controllers = []
    ego_vehicle = None
    agent = None
    spectator = None

    try:
        set_sync(world, tm, True, fixed_delta=0.05)
        spectator = world.get_spectator()

        spawn_points = world.get_map().get_spawn_points()
        start_tf, end_tf = pick_far_spawn_pair(spawn_points)
        logging.info(f"Map: {world.get_map().name}, start -> end set")

        ego_vehicle = spawn_ego_model3(world, start_tf)
        logging.info("Spawned Tesla Model 3 (hero)")

        # Spawn background actors
        vehicles = spawn_traffic(client, world, args.tm_port, args.vehicles)
        logging.info(f"Spawned {len(vehicles)} traffic vehicles")

        w, wc = spawn_walkers(world, args.walkers)
        walkers, walker_controllers = w, wc
        logging.info(f"Spawned {len(walkers)} walkers")

        # Make sure all actors are in the world before starting the loop
        world.tick()

        agent = BehaviorAgent(ego_vehicle, behavior=args.behavior)
        # Set the destination for the agent (from current ego location)
        agent.set_destination(end_tf.location)
        logging.info("BehaviorAgent destination set; starting navigation loop")

        start_time = time.time()
        timeout_s = args.timeout
        reached = False

        while True:
            world.tick()

            # Update spectator to follow ego (FPV or chase)
            if spectator is not None and ego_vehicle is not None:
                ego_tf = ego_vehicle.get_transform()
                yaw_deg = ego_tf.rotation.yaw
                yaw = math.radians(yaw_deg)
                if args.view == "fpv":
                    # near-windshield first-person view
                    forward = 0.7
                    height = 1.4
                    dx = forward * math.cos(yaw)
                    dy = forward * math.sin(yaw)
                    loc = ego_tf.location + carla.Location(x=dx, y=dy, z=height)
                    rot = carla.Rotation(pitch=0.0, yaw=yaw_deg)
                elif args.view == "chase":
                    # chase cam behind the car
                    back = 8.0
                    height = 4.0
                    dx = -back * math.cos(yaw)
                    dy = -back * math.sin(yaw)
                    loc = ego_tf.location + carla.Location(x=dx, y=dy, z=height)
                    rot = carla.Rotation(pitch=-10.0, yaw=yaw_deg)
                else:
                    # top-downish debug
                    height = 30.0
                    loc = ego_tf.location + carla.Location(x=0.0, y=0.0, z=height)
                    rot = carla.Rotation(pitch=-90.0, yaw=yaw_deg)
                spectator.set_transform(carla.Transform(loc, rot))

            if agent.done():
                reached = True
                break

            control = agent.run_step()
            ego_vehicle.apply_control(control)

            if (time.time() - start_time) > timeout_s:
                break

        if reached:
            logging.info("Destination reached ✅")
        else:
            logging.info("Timeout expired before reaching destination ⚠️")

        # Let things settle briefly for visual confirmation
        for _ in range(20):
            world.tick()
            time.sleep(0.02)

    finally:
        # Stop walker controllers first
        for c in walker_controllers:
            try:
                c.stop()
            except RuntimeError:
                pass

        # Destroy actors
        actor_ids = []
        for a in walker_controllers + walkers + vehicles + ([ego_vehicle] if ego_vehicle is not None else []):
            if a is not None:
                actor_ids.append(a.id)
        if actor_ids:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_ids])

        # Restore world settings
        try:
            world.apply_settings(original_settings)
            tm.set_synchronous_mode(original_sync)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Spawn Tesla Model 3 with BehaviorAgent in CARLA and run a route with traffic and pedestrians.")
    parser.add_argument("--host", default="10.0.0.121", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--tm-port", dest="tm_port", type=int, default=8000, help="Traffic Manager port")
    parser.add_argument("--vehicles", type=int, default=50, help="Number of traffic vehicles to spawn")
    parser.add_argument("--walkers", type=int, default=50, help="Number of pedestrians to spawn")
    parser.add_argument("--behavior", choices=["cautious", "normal", "aggressive"], default="normal", help="BehaviorAgent driving style")
    parser.add_argument("--timeout", type=float, default=300.0, help="Max seconds to try reaching destination")
    parser.add_argument("--view", choices=["fpv", "chase", "top"], default="fpv", help="Spectator camera view mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)


