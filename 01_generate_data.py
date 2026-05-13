import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 15_000
FIELD_SIZE = 100.0


def generate_field_map(n):
    pos_x = np.random.uniform(0, FIELD_SIZE, n)
    pos_y = np.random.uniform(0, FIELD_SIZE, n)

    moisture_base = 0.4 + 0.15 * np.sin(pos_x / 20) + 0.1 * np.cos(pos_y / 15)
    soil_moisture = np.clip(moisture_base + np.random.normal(0, 0.05, n), 0.05, 0.95)

    ph_base = 6.5 - 0.3 * soil_moisture + 0.1 * np.sin(pos_x / 30)
    soil_ph = np.clip(ph_base + np.random.normal(0, 0.15, n), 4.5, 8.5)

    return pos_x, pos_y, soil_moisture, soil_ph


def generate_lidar(n):
    directions = ["lidar_N", "lidar_NE", "lidar_E", "lidar_SE",
                  "lidar_S", "lidar_SW", "lidar_W", "lidar_NW"]

    lidar_data = {}
    for d in directions:
        base = np.random.uniform(0.5, 12.0, n)
        noise = np.random.normal(0, 0.05, n)
        obstacle = np.random.choice([1.0, 0.0], size=n, p=[0.08, 0.92])
        close_dist = np.random.uniform(0.3, 1.5, n)
        dist = np.where(obstacle, close_dist, base + noise)
        lidar_data[d] = np.clip(dist, 0.15, 12.0)

    all_dists = np.stack(list(lidar_data.values()), axis=1)
    lidar_data["lidar_min"] = all_dists.min(axis=1)
    lidar_data["lidar_front_clear"] = lidar_data["lidar_N"]

    return lidar_data


def generate_imu(n, pos_x, pos_y):
    pitch = 3 * np.sin(pos_x / 25) + np.random.normal(0, 1.5, n)
    roll  = 2 * np.cos(pos_y / 20) + np.random.normal(0, 1.0, n)
    return np.clip(pitch, -20, 20), np.clip(roll, -15, 15)


def generate_rtk(pos_x, pos_y, n):
    rtk_x = pos_x + np.random.normal(0, 0.02, n)
    rtk_y = pos_y + np.random.normal(0, 0.02, n)
    rtk_fix_quality = np.random.choice([1, 2, 4, 5], size=n, p=[0.02, 0.05, 0.88, 0.05])
    return rtk_x, rtk_y, rtk_fix_quality


def generate_rgbd(n):
    depth_center = np.random.uniform(0.4, 8.0, n)
    depth_min    = depth_center - np.random.uniform(0, 0.5, n)
    obstacle_detected = (depth_min < 1.0).astype(float)
    return np.clip(depth_center, 0.4, 8.0), np.clip(depth_min, 0.3, 8.0), obstacle_detected


def compute_labels(lidar_min, lidar_N, depth_min, imu_pitch, imu_roll,
                   soil_moisture, soil_ph, rtk_fix):
    collision_risk = np.clip(
        0.8 * np.exp(-lidar_min / 1.5) +
        0.15 * np.exp(-depth_min / 1.0) +
        0.05 * (np.abs(imu_roll) > 10).astype(float),
        0.0, 1.0
    )

    speed = np.clip(
        0.6 * (lidar_N / 12.0) *
        (1 - 0.4 * np.abs(imu_pitch) / 20) *
        (1 - 0.3 * collision_risk) *
        (rtk_fix == 4).astype(float) * 0.3 + 0.7,
        0.0, 1.0
    )

    needs_treatment = (
        (soil_moisture < 0.3) |
        (soil_ph < 5.8) |
        (soil_ph > 7.5)
    ).astype(float)

    nav_quality = (rtk_fix == 4).astype(float)

    return collision_risk, speed, needs_treatment, nav_quality


def generate_dataset(n=N):
    print(f"Generuji {n:,} simulačních vzorků...")

    pos_x, pos_y, soil_moisture, soil_ph = generate_field_map(n)
    lidar = generate_lidar(n)
    imu_pitch, imu_roll = generate_imu(n, pos_x, pos_y)
    rtk_x, rtk_y, rtk_fix = generate_rtk(pos_x, pos_y, n)
    depth_center, depth_min, cam_obstacle = generate_rgbd(n)

    collision_risk, speed, needs_treatment, nav_quality = compute_labels(
        lidar["lidar_min"], lidar["lidar_N"], depth_min,
        imu_pitch, imu_roll, soil_moisture, soil_ph, rtk_fix
    )

    df = pd.DataFrame({
        "rtk_x": rtk_x, "rtk_y": rtk_y, "rtk_fix": rtk_fix,
        **lidar,
        "imu_pitch": imu_pitch, "imu_roll": imu_roll,
        "soil_moisture": soil_moisture, "soil_ph": soil_ph,
        "depth_center": depth_center, "depth_min": depth_min,
        "cam_obstacle": cam_obstacle,
        "out_collision_risk":  collision_risk,
        "out_speed":           speed,
        "out_needs_treatment": needs_treatment,
        "out_nav_quality":     nav_quality,
    })

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/robot_simulation.csv", index=False)

    print(f"✓ Dataset uložen: data/robot_simulation.csv")
    print(f"  Vzorků:  {len(df):,}")
    print(f"  Featury: {len([c for c in df.columns if not c.startswith('out_')])}")
    print(f"  Labely:  {len([c for c in df.columns if c.startswith('out_')])}")

    return df


if __name__ == "__main__":
    df = generate_dataset()