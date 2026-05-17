import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

Path("results/grafy").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "figure.dpi":        100,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "lines.linewidth":   1.5,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
})

FEATURES = [
    "rtk_x", "rtk_y", "rtk_fix",
    "lidar_N", "lidar_NE", "lidar_E", "lidar_SE",
    "lidar_S", "lidar_SW", "lidar_W", "lidar_NW",
    "lidar_min", "lidar_front_clear",
    "imu_pitch", "imu_roll",
    "soil_moisture", "soil_ph",
    "depth_center", "depth_min", "cam_obstacle",
]


def plot_scatter_collision(results):
    y_test = results["out_collision_risk"]["y_test"]
    y_pred = results["out_collision_risk"]["y_pred"]
    r2     = results["out_collision_risk"]["r2"]

    idx = np.random.choice(len(y_test), 800, replace=False)
    yt, yp = y_test[idx], y_pred[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(yt, yp, s=8, alpha=0.4, color="steelblue", label="Vzorky")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Ideální shoda")
    ax.set_xlabel("Skutečná hodnota (simulace)")
    ax.set_ylabel("Predikce (surrogate model)")
    ax.set_title(f"Riziko kolize — predikce vs. realita  (R² = {r2:.3f})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/grafy/01_scatter_collision.png")
    plt.close()
    print("✓ 01_scatter_collision.png")


def plot_scatter_speed(results):
    y_test = results["out_speed"]["y_test"]
    y_pred = results["out_speed"]["y_pred"]
    r2     = results["out_speed"]["r2"]

    idx = np.random.choice(len(y_test), 800, replace=False)
    yt, yp = y_test[idx], y_pred[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(yt, yp, s=8, alpha=0.4, color="steelblue", label="Vzorky")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Ideální shoda")
    ax.set_xlabel("Skutečná hodnota (simulace)")
    ax.set_ylabel("Predikce (surrogate model)")
    ax.set_title(f"Optimální rychlost — predikce vs. realita  (R² = {r2:.3f})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/grafy/02_scatter_speed.png")
    plt.close()
    print("✓ 02_scatter_speed.png")


def plot_lc_collision(lc_results):
    lc = lc_results["out_collision_risk"]
    sizes = lc["train_sizes"]
    train = lc["train_scores_mean"]
    test  = lc["test_scores_mean"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sizes, train, "o-", color="steelblue", label="Tréninková sada")
    ax.plot(sizes, test,  "s-", color="darkorange", label="Testovací sada")
    ax.set_xlabel("Počet trénovacích vzorků")
    ax.set_ylabel("R² skóre")
    ax.set_title("Learning curve — riziko kolize")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/grafy/03_lc_collision.png")
    plt.close()
    print("✓ 03_lc_collision.png")


def plot_lc_treatment(lc_results):
    lc = lc_results["out_needs_treatment"]
    sizes = lc["train_sizes"]
    train = lc["train_scores_mean"] * 100
    test  = lc["test_scores_mean"]  * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sizes, train, "o-", color="steelblue", label="Tréninková sada")
    ax.plot(sizes, test,  "s-", color="darkorange", label="Testovací sada")
    ax.set_xlabel("Počet trénovacích vzorků")
    ax.set_ylabel("Přesnost (%)")
    ax.set_title("Learning curve — potřeba ošetření půdy")
    ax.set_ylim(50, 105)
    ax.set_yticks(np.arange(50, 110, 10))
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/grafy/04_lc_treatment.png")
    plt.close()
    print("✓ 04_lc_treatment.png")


def plot_fi_collision(results):
    fi = pd.Series(results["out_collision_risk"]["feature_importance"]).nlargest(10)
    rename = {
        "lidar_min":         "LiDAR min. vzdálenost",
        "depth_min":         "RGB-D min. hloubka",
        "depth_center":      "RGB-D střed záběru",
        "cam_obstacle":      "RGB-D překážka",
        "lidar_N":           "LiDAR sever",
        "lidar_front_clear": "LiDAR volná cesta",
        "imu_roll":          "IMU boční náklon",
        "imu_pitch":         "IMU náklon dopředu",
        "soil_moisture":     "Vlhkost půdy",
        "soil_ph":           "pH půdy",
    }
    labels = [rename.get(k, k) for k in fi.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels[::-1], fi.values[::-1], color="steelblue", height=0.6)
    ax.set_xlabel("Relativní důležitost")
    ax.set_title("Důležitost senzorů — riziko kolize")
    ax.set_xlim(0, fi.values.max() * 1.15)
    ax.grid(axis="x", alpha=0.4, linestyle="--")
    plt.tight_layout()
    plt.savefig("results/grafy/05_fi_collision.png")
    plt.close()
    print("✓ 05_fi_collision.png")


def plot_fi_speed(results):
    fi = pd.Series(results["out_speed"]["feature_importance"]).nlargest(10)
    rename = {
        "lidar_front_clear": "LiDAR volná cesta",
        "lidar_N":           "LiDAR sever",
        "rtk_fix":           "RTK kvalita fixu",
        "lidar_min":         "LiDAR min. vzdálenost",
        "imu_pitch":         "IMU náklon dopředu",
        "imu_roll":          "IMU boční náklon",
        "depth_center":      "RGB-D střed záběru",
        "depth_min":         "RGB-D min. hloubka",
        "soil_moisture":     "Vlhkost půdy",
        "soil_ph":           "pH půdy",
    }
    labels = [rename.get(k, k) for k in fi.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels[::-1], fi.values[::-1], color="steelblue", height=0.6)
    ax.set_xlabel("Relativní důležitost")
    ax.set_title("Důležitost senzorů — optimální rychlost")
    ax.set_xlim(0, fi.values.max() * 1.15)
    ax.grid(axis="x", alpha=0.4, linestyle="--")
    plt.tight_layout()
    plt.savefig("results/grafy/06_fi_speed.png")
    plt.close()
    print("✓ 06_fi_speed.png")


def plot_hist_collision(results):
    y_test = results["out_collision_risk"]["y_test"]
    y_pred = results["out_collision_risk"]["y_pred"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(y_test, bins=30, alpha=0.6, color="steelblue", label="Simulace (skutečnost)")
    ax.hist(y_pred, bins=30, alpha=0.6, color="darkorange", label="Surrogate (predikce)")
    ax.set_xlabel("Riziko kolize")
    ax.set_ylabel("Počet vzorků")
    ax.set_title("Distribuce rizika kolize")
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/grafy/07_hist_collision.png")
    plt.close()
    print("✓ 07_hist_collision.png")


def plot_hist_speed(results):
    y_test = results["out_speed"]["y_test"]
    y_pred = results["out_speed"]["y_pred"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(y_test, bins=30, alpha=0.6, color="steelblue", label="Simulace (skutečnost)")
    ax.hist(y_pred, bins=30, alpha=0.6, color="darkorange", label="Surrogate (predikce)")
    ax.set_xlabel("Optimální rychlost (normalizovaná)")
    ax.set_ylabel("Počet vzorků")
    ax.set_title("Distribuce optimální rychlosti")
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/grafy/08_hist_speed.png")
    plt.close()
    print("✓ 08_hist_speed.png")


def plot_residuals_collision(results):
    y_test = results["out_collision_risk"]["y_test"]
    y_pred = results["out_collision_risk"]["y_pred"]
    residuals = y_pred - y_test

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, residuals, s=6, alpha=0.3, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Skutečná hodnota")
    ax.set_ylabel("Chyba predikce (predikce − skutečnost)")
    ax.set_title("Residuály — riziko kolize")
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/grafy/09_residuals_collision.png")
    plt.close()
    print("✓ 09_residuals_collision.png")


def plot_residuals_speed(results):
    y_test = results["out_speed"]["y_test"]
    y_pred = results["out_speed"]["y_pred"]
    residuals = y_pred - y_test

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, residuals, s=6, alpha=0.3, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Skutečná hodnota")
    ax.set_ylabel("Chyba predikce (predikce − skutečnost)")
    ax.set_title("Residuály — optimální rychlost")
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/grafy/10_residuals_speed.png")
    plt.close()
    print("✓ 10_residuals_speed.png")


def plot_latency(results):
    labels = ["Plná simulace\n(Gazebo)", "Kolize", "Rychlost", "Ošetření půdy", "Kvalita GPS"]
    times  = [
        500.0,
        results["out_collision_risk"]["inference_ms"],
        results["out_speed"]["inference_ms"],
        results["out_needs_treatment"]["inference_ms"],
        results["out_nav_quality"]["inference_ms"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, times, color=["darkorange"] + ["steelblue"] * 4, width=0.5)
    for bar, t in zip(bars, times):
        label = f"{t:.0f} ms" if t >= 1 else f"{t:.2f} ms"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                label, ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Latence [ms] (log škála)")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 3000)
    ax.axhline(16.67, color="gray", linestyle="--", linewidth=1,
               label="60 FPS limit (16.67 ms)")
    ax.set_title("Latence inference — surrogate vs. plná simulace")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4, linestyle="--", which="both")
    plt.tight_layout()
    plt.savefig("results/grafy/11_latency.png")
    plt.close()
    print("✓ 11_latency.png")


if __name__ == "__main__":
    print("Načítám výsledky...")
    results = joblib.load("results/training_results.pkl")
    lc      = joblib.load("results/learning_curves.pkl")

    print("\nGeneruji grafy:")
    plot_scatter_collision(results)
    plot_scatter_speed(results)
    plot_lc_collision(lc)
    plot_lc_treatment(lc)
    plot_fi_collision(results)
    plot_fi_speed(results)
    plot_hist_collision(results)
    plot_hist_speed(results)
    plot_residuals_collision(results)
    plot_residuals_speed(results)
    plot_latency(results)

    print(f"\n✓ Hotovo — 11 grafů v results/grafy/")