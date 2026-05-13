import numpy as np
import joblib
import time
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SensorData:
    # RTK GNSS
    rtk_x: float
    rtk_y: float
    rtk_fix: int          # 4=RTK fix, 5=float, 2=DGPS, 1=GPS only

    # LiDAR (8 směrů, vzdálenosti v metrech)
    lidar_N:  float
    lidar_NE: float
    lidar_E:  float
    lidar_SE: float
    lidar_S:  float
    lidar_SW: float
    lidar_W:  float
    lidar_NW: float

    # IMU
    imu_pitch: float
    imu_roll:  float

    # Půdní senzory
    soil_moisture: float
    soil_ph:       float

    # RGB-D kamera
    depth_center: float
    depth_min:    float
    cam_obstacle: float

    def to_feature_vector(self) -> np.ndarray:
        lidar_vals = [self.lidar_N, self.lidar_NE, self.lidar_E, self.lidar_SE,
                      self.lidar_S, self.lidar_SW, self.lidar_W, self.lidar_NW]
        return np.array([[
            self.rtk_x, self.rtk_y, self.rtk_fix,
            *lidar_vals,
            min(lidar_vals),   # lidar_min
            self.lidar_N,      # lidar_front_clear
            self.imu_pitch, self.imu_roll,
            self.soil_moisture, self.soil_ph,
            self.depth_center, self.depth_min, self.cam_obstacle,
        ]])


@dataclass
class RobotState:
    collision_risk:  float
    optimal_speed:   float
    needs_treatment: bool
    nav_quality_ok:  bool
    inference_ms:    float

    @property
    def alert_level(self) -> str:
        if self.collision_risk > 0.7:   return "CRITICAL"
        elif self.collision_risk > 0.4: return "WARNING"
        else:                           return "OK"

    def __str__(self):
        return (
            f"alert={self.alert_level} | "
            f"kolize={self.collision_risk:.2f} | "
            f"rychlost={self.optimal_speed:.2f} | "
            f"ošetření={'ANO' if self.needs_treatment else 'NE'} | "
            f"GPS={'OK' if self.nav_quality_ok else 'DEGRADOVANÁ'} | "
            f"{self.inference_ms:.3f}ms"
        )


class SurrogateInference:
    def __init__(self, models_dir: str = "models/"):
        self._models = {}
        for name, filename in {
            "collision_risk":  "surrogate_out_collision_risk.pkl",
            "speed":           "surrogate_out_speed.pkl",
            "needs_treatment": "surrogate_out_needs_treatment.pkl",
            "nav_quality":     "surrogate_out_nav_quality.pkl",
        }.items():
            path = Path(models_dir) / filename
            if path.exists():
                self._models[name] = joblib.load(path)
            else:
                print(f"⚠ Model nenalezen: {path}")
        print(f"✓ Surrogate models načteny ({len(self._models)}/4)")

    def predict(self, sensor: SensorData) -> RobotState:
        X = sensor.to_feature_vector()
        t0 = time.perf_counter()

        return RobotState(
            collision_risk  = float(np.clip(self._models["collision_risk"].predict(X)[0], 0, 1)),
            optimal_speed   = float(np.clip(self._models["speed"].predict(X)[0], 0, 1)),
            needs_treatment = bool(self._models["needs_treatment"].predict(X)[0]),
            nav_quality_ok  = bool(self._models["nav_quality"].predict(X)[0]),
            inference_ms    = (time.perf_counter() - t0) * 1000,
        )


if __name__ == "__main__":
    surrogate = SurrogateInference("models/")

    scenarios = [
        ("Normální jízda polem", SensorData(
            rtk_x=45.2, rtk_y=30.1, rtk_fix=4,
            lidar_N=8.0, lidar_NE=7.5, lidar_E=6.0, lidar_SE=5.5,
            lidar_S=9.0, lidar_SW=8.0, lidar_W=7.0, lidar_NW=8.5,
            imu_pitch=2.0, imu_roll=1.0,
            soil_moisture=0.55, soil_ph=6.8,
            depth_center=6.0, depth_min=5.5, cam_obstacle=0.0,
        )),
        ("Blízká překážka vpředu", SensorData(
            rtk_x=50.0, rtk_y=50.0, rtk_fix=4,
            lidar_N=0.6, lidar_NE=0.8, lidar_E=5.0, lidar_SE=6.0,
            lidar_S=8.0, lidar_SW=7.0, lidar_W=6.0, lidar_NW=0.9,
            imu_pitch=1.0, imu_roll=0.5,
            soil_moisture=0.4, soil_ph=6.5,
            depth_center=0.7, depth_min=0.5, cam_obstacle=1.0,
        )),
        ("Suchá kyselá půda", SensorData(
            rtk_x=80.0, rtk_y=20.0, rtk_fix=4,
            lidar_N=10.0, lidar_NE=9.5, lidar_E=11.0, lidar_SE=10.0,
            lidar_S=9.0, lidar_SW=10.5, lidar_W=9.0, lidar_NW=11.0,
            imu_pitch=0.5, imu_roll=0.2,
            soil_moisture=0.18, soil_ph=5.2,
            depth_center=8.0, depth_min=7.5, cam_obstacle=0.0,
        )),
    ]

    print("\nTest scénáře:")
    print("-" * 60)
    for name, data in scenarios:
        state = surrogate.predict(data)
        print(f"\n[{name}]")
        print(f"  {state}")