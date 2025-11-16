import pyarrow as pa


# Constants (default values; override via config at runtime if needed)
K_ROUTE_POINTS = 16
N_FUTURE_STEPS = 6
FIXED_DELTA_SECONDS = 0.05
FUTURE_DELTA_SECONDS = 0.5
ACTOR_RADIUS_METERS = 60.0
WINDOW_METERS = 80.0


# Encodings / enums
ROAD_OPTION_TO_INT = {
    "LaneFollow": 0,
    "Left": 1,
    "Right": 2,
    "Straight": 3,
    "ChangeLaneLeft": 4,
    "ChangeLaneRight": 5,
}

ACTOR_TYPE_TO_INT = {
    "vehicle": 0,
    "walker": 1,
}


def frames_schema() -> pa.schema:
    return pa.schema([
        ("run_id", pa.string()),
        ("shard_id", pa.int32()),
        ("frame_id", pa.int64()),
        ("sim_time", pa.float64()),
        ("fixed_dt", pa.float32()),
        ("future_dt", pa.float32()),
        ("map", pa.string()),
        ("weather", pa.int16()),
        ("seed", pa.int64()),
        ("ego_x_w", pa.float32()),
        ("ego_y_w", pa.float32()),
        ("ego_yaw_w", pa.float32()),
        ("speed_mps", pa.float32()),
        ("yaw_rate", pa.float32()),
        ("road_option", pa.int8()),
        ("goal_dist_m", pa.float32()),
        ("at_junction", pa.int8()),
        ("dist_to_next_junction", pa.float32()),
        # Extended ego/meta fields for full obs schema
        ("accel_long", pa.float32()),
        ("accel_lat", pa.float32()),
        ("steer_angle_rad", pa.float32()),
        ("steer_norm", pa.float32()),
        ("throttle", pa.float32()),
        ("brake", pa.float32()),
        ("curvature", pa.float32()),
        ("gear", pa.int16()),
        ("speed_limit_mps", pa.float32()),
        ("command", pa.int8()),
        ("time_of_day_sin", pa.float32()),
        ("time_of_day_cos", pa.float32()),
        ("route_progress_m", pa.float32()),
        ("route_id", pa.int32()),
        ("scenario_id", pa.int32()),
        ("frame_ref", pa.string()),
    ])


def route_points_schema() -> pa.schema:
    return pa.schema([
        ("run_id", pa.string()),
        ("shard_id", pa.int32()),
        ("frame_id", pa.int64()),
        ("idx", pa.int16()),
        ("x_ego", pa.float32()),
        ("y_ego", pa.float32()),
        # Optional route polyline enrichments
        ("dx", pa.float32()),
        ("dy", pa.float32()),
        ("curvature", pa.float32()),
        ("s_frac", pa.float32()),
    ])


def actors_schema() -> pa.schema:
    return pa.schema([
        ("run_id", pa.string()),
        ("shard_id", pa.int32()),
        ("frame_id", pa.int64()),
        ("actor_id", pa.int64()),
        ("type_id", pa.int8()),  # 0 vehicle, 1 walker
        ("x_ego", pa.float32()),
        ("y_ego", pa.float32()),
        ("yaw", pa.float32()),
        ("vx", pa.float32()),
        ("vy", pa.float32()),
        ("length", pa.float32()),
        ("width", pa.float32()),
    ])


def traffic_lights_schema() -> pa.schema:
    return pa.schema([
        ("run_id", pa.string()),
        ("shard_id", pa.int32()),
        ("frame_id", pa.int64()),
        ("tl_id", pa.int64()),
        ("state", pa.int8()),
        ("stop_x_ego", pa.float32()),
        ("stop_y_ego", pa.float32()),
    ])


def map_segments_schema() -> pa.schema:
    return pa.schema([
        ("run_id", pa.string()),
        ("shard_id", pa.int32()),
        ("frame_id", pa.int64()),
        ("seg_id", pa.int64()),
        ("kind", pa.int8()),  # 0 lane_centerline, 1 tl_stopline, etc.
        ("coords_x", pa.list_(pa.float32())),
        ("coords_y", pa.list_(pa.float32())),
        ("lane_id", pa.int64()),
        ("lane_type", pa.int8()),
        ("speed_limit", pa.float32()),
    ])


def futures_schema() -> pa.schema:
    return pa.schema([
        ("run_id", pa.string()),
        ("shard_id", pa.int32()),
        ("frame_id", pa.int64()),
        ("i", pa.int8()),
        ("x_ego", pa.float32()),
        ("y_ego", pa.float32()),
        ("v_mps", pa.float32()),
    ])


def object_tokens_schema() -> pa.schema:
    return pa.schema([
        ("run_id", pa.string()),
        ("shard_id", pa.int32()),
        ("frame_id", pa.int64()),
        ("idx", pa.int16()),
        ("actor_id", pa.int64()),  # for logging/debug
        ("type_id", pa.int8()),    # {car/truck=0, ped=1, bike=2, light=3...}
        ("x_ego", pa.float32()),
        ("y_ego", pa.float32()),
        ("sin_yaw", pa.float32()),
        ("cos_yaw", pa.float32()),
        ("length", pa.float32()),
        ("width", pa.float32()),
        ("vx", pa.float32()),
        ("vy", pa.float32()),
        ("oncoming_flag", pa.int8()),
        ("priority_flag", pa.int8()),
    ])


def bev_frames_schema() -> pa.schema:
    return pa.schema([
        ("run_id", pa.string()),
        ("shard_id", pa.int32()),
        ("frame_id", pa.int64()),
        ("C", pa.int16()),
        ("H", pa.int16()),
        ("W", pa.int16()),
        ("dtype", pa.string()),          # e.g., "float32"
        ("encoding", pa.string()),       # e.g., "npy.zstd"
        ("channel_spec", pa.string()),   # semantic description/version of channels
        ("meters_per_px", pa.float32()),
        ("x_fwd_m", pa.float32()),
        ("y_left_m", pa.float32()),
        ("ego_x_w", pa.float32()),
        ("ego_y_w", pa.float32()),
        ("ego_yaw_w", pa.float32()),
        ("spec_version", pa.int16()),
        ("norms_version", pa.int16()),
        ("frame_ref", pa.string()),
        ("data", pa.binary()),           # compressed NumPy blob
    ])


class Schemas:
    FRAMES = frames_schema()
    ROUTE_POINTS = route_points_schema()
    ACTORS = actors_schema()
    TRAFFIC_LIGHTS = traffic_lights_schema()
    MAP_SEGMENTS = map_segments_schema()
    FUTURES = futures_schema()
    OBJECT_TOKENS = object_tokens_schema()
    BEV_FRAMES = bev_frames_schema()


