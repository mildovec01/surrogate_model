import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

FEATURES = [
    "rtk_x", "rtk_y", "rtk_fix",
    "lidar_N", "lidar_NE", "lidar_E", "lidar_SE",
    "lidar_S", "lidar_SW", "lidar_W", "lidar_NW",
    "lidar_min", "lidar_front_clear",
    "imu_pitch", "imu_roll",
    "soil_moisture", "soil_ph",
    "depth_center", "depth_min", "cam_obstacle",
]

TARGETS = {
    "out_collision_risk":  "regressor",
    "out_speed":           "regressor",
    "out_needs_treatment": "classifier",
    "out_nav_quality":     "classifier",
}

RF_PARAMS = dict(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)


def train_all(csv_path="data/robot_simulation.csv"):
    print("=" * 55)
    print("  SURROGATE MODEL — TRÉNINK")
    print("=" * 55)

    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    results = {}

    for target, model_type in TARGETS.items():
        print(f"\n▶ {target} ({model_type})")
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if model_type == "regressor":
            model = RandomForestRegressor(**RF_PARAMS)
        else:
            model = RandomForestClassifier(**RF_PARAMS)

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)

        if model_type == "regressor":
            mae = mean_absolute_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            print(f"  MAE: {mae:.4f}  |  R²: {r2:.4f}")
            metrics = {"mae": mae, "r2": r2}
        else:
            acc = (y_pred == y_test).mean()
            print(f"  Přesnost: {acc*100:.2f}%")
            print(classification_report(y_test, y_pred, digits=3))
            metrics = {"accuracy": acc}

        print(f"  Čas tréninku: {train_time:.2f}s")

        single = X_test.iloc[[0]]
        runs = 1000
        t0 = time.perf_counter()
        for _ in range(runs):
            model.predict(single)
        inference_ms = (time.perf_counter() - t0) / runs * 1000
        print(f"  Inference: {inference_ms:.3f} ms / predikce")

        fi = pd.Series(model.feature_importances_, index=FEATURES)
        top5 = fi.nlargest(5)
        print(f"  Top 5 featury:")
        for feat, imp in top5.items():
            bar = "█" * int(imp * 40)
            print(f"    {feat:<22} {bar} {imp:.3f}")

        model_path = f"models/surrogate_{target}.pkl"
        joblib.dump(model, model_path)

        results[target] = {
            **metrics,
            "train_time_s": round(train_time, 3),
            "inference_ms": round(inference_ms, 4),
            "model_path": model_path,
            "feature_importance": fi.to_dict(),
            "model_type": model_type,
            "y_test": y_test.values,
            "y_pred": y_pred,
        }

    joblib.dump(results, "results/training_results.pkl")
    print("\n✓ Modely uloženy do models/")
    print("✓ Výsledky uloženy do results/training_results.pkl")

    return results


def compute_learning_curves(csv_path="data/robot_simulation.csv"):
    print("\nPočítám learning curves...")
    df = pd.read_csv(csv_path)
    X = df[FEATURES]

    lc_results = {}
    for target in ["out_collision_risk", "out_needs_treatment"]:
        y = df[target]
        model_type = TARGETS[target]

        if model_type == "regressor":
            model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            scoring = "r2"
        else:
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            scoring = "accuracy"

        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.05, 1.0, 10),
            cv=3, scoring=scoring, n_jobs=-1,
        )

        lc_results[target] = {
            "train_sizes": train_sizes,
            "train_scores_mean": train_scores.mean(axis=1),
            "test_scores_mean":  test_scores.mean(axis=1),
            "scoring": scoring,
        }
        print(f"  ✓ {target}")

    joblib.dump(lc_results, "results/learning_curves.pkl")
    print("✓ Learning curves uloženy")
    return lc_results


if __name__ == "__main__":
    results = train_all()
    lc = compute_learning_curves()
    