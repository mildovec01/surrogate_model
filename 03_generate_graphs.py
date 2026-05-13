"""
Generátor grafů pro plakát / prezentaci
Vyžaduje nejdřív spustit 01 a 02.
Výstup: results/grafy/*.png  (300 DPI)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("results/grafy").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

COLORS = {
    "primary":   "#2D6A4F",
    "secondary": "#52B788",
    "accent":    "#F4A261",
    "neutral":   "#495057",
}

FEATURES = [
    "rtk_x", "rtk_y", "rtk_fix",
    "lidar_N", "lidar_NE", "lidar_E", "lidar_SE",
    "lidar_S", "lidar_SW", "lidar_W", "lidar_NW",
    "lidar_min", "lidar_front_clear",
    "imu_pitch", "imu_roll",
    "soil_moisture", "soil_ph",
    "depth_center", "depth_min", "cam_obstacle",
]

RENAME = {
    "lidar_min":         "LiDAR – min. vzdálenost",
    "lidar_N":           "LiDAR – sever (přímý)",
    "lidar_front_clear": "LiDAR – volná cesta",
    "lidar_NE":          "LiDAR – severovýchod",
    "lidar_NW":          "LiDAR – severozápad",
    "depth_min":         "RGB-D – min. hloubka",
    "depth_center":      "RGB-D – střed záběru",
    "cam_obstacle":      "RGB-D – překážka",
    "imu_pitch":         "IMU – náklon dopředu",
    "imu_roll":          "IMU – boční náklon",
    "soil_moisture":     "Vlhkost půdy",
    "soil_ph":           "pH půdy",
    "rtk_x":             "RTK GNSS – X",
    "rtk_y":             "RTK GNSS – Y",
    "rtk_fix":           "RTK – kvalita fixu",
}


def plot_feature_importance(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Důležitost vstupních senzorů (Feature Importance)",
                 fontsize=14, fontweight="bold")

    for ax, target, title in zip(
        axes,
        ["out_collision_risk", "out_speed"],
        ["Predikce rizika kolize", "Predikce optimální rychlosti"]
    ):
        fi = pd.Series(results[target]["feature_importance"]).nlargest(10)
        labels = [RENAME.get(k, k) for k in fi.index]
        vals   = fi.values

        bars = ax.barh(labels[::-1], vals[::-1],
                       color=COLORS["primary"], alpha=0.85, height=0.6)
        for bar, val in zip(bars, vals[::-1]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("Relativní důležitost")
        ax.set_xlim(0, vals.max() * 1.2)
        ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.savefig("results/grafy/01_feature_importance.png")
    plt.close()
    print("✓ 01_feature_importance.png")


def plot_learning_curves(lc_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Learning Curves — kolik dat surrogate potřebuje?",
                 fontsize=14, fontweight="bold")

    rename_target = {
        "out_collision_risk":  ("Riziko kolize (R²)", "R² skóre"),
        "out_needs_treatment": ("Potřeba ošetření půdy (přesnost)", "Přesnost (%)"),
    }

    for ax, (target, lc) in zip(axes, lc_results.items()):
        title, ylabel = rename_target[target]
        sizes = lc["train_sizes"]
        train = lc["train_scores_mean"].copy()
        test  = lc["test_scores_mean"].copy()

        if "%" in ylabel:
            train *= 100
            test  *= 100

        ax.plot(sizes, train, "o-", color=COLORS["primary"],
                label="Tréninková sada", linewidth=2, markersize=5)
        ax.plot(sizes, test, "s--", color=COLORS["accent"],
                label="Testovací sada", linewidth=2, markersize=5)

        diff = np.abs(train - test)
        opt_idx = np.argmin(diff[len(diff)//2:]) + len(diff)//2
        ax.axvline(sizes[opt_idx], color=COLORS["secondary"],
                   linestyle=":", alpha=0.7, label=f"Optimum ~{sizes[opt_idx]:,.0f}")
        ax.fill_between(sizes, train, test, alpha=0.08, color=COLORS["secondary"])

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("Počet trénovacích vzorků")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/grafy/02_learning_curves.png")
    plt.close()
    print("✓ 02_learning_curves.png")


def plot_predictions_vs_real(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Přesnost predikce — surrogate vs. simulace",
                 fontsize=14, fontweight="bold")

    for ax, target, title in zip(
        axes,
        ["out_collision_risk", "out_speed"],
        ["Riziko kolize", "Optimální rychlost"]
    ):
        y_test = results[target]["y_test"]
        y_pred = results[target]["y_pred"]
        r2     = results[target].get("r2", 0)

        idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
        yt, yp = y_test[idx], y_pred[idx]

        ax.scatter(yt, yp, alpha=0.25, s=12, color=COLORS["primary"])
        lims = [min(yt.min(), yp.min()) - 0.05, max(yt.max(), yp.max()) + 0.05]
        ax.plot(lims, lims, "--", color=COLORS["accent"],
                linewidth=1.5, label="Ideální shoda")

        ax.set_title(f"{title}  (R² = {r2:.3f})", fontweight="bold", pad=10)
        ax.set_xlabel("Skutečná hodnota (simulace)")
        ax.set_ylabel("Predikce (surrogate)")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("results/grafy/03_predictions_scatter.png")
    plt.close()
    print("✓ 03_predictions_scatter.png")


def plot_latency_benchmark(results):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Rychlost inference — surrogate vs. simulace",
                 fontsize=14, fontweight="bold")

    labels = [
        "Plná simulace\n(odhad Gazebo)",
        "Surrogate\nriziko kolize",
        "Surrogate\nrychlost",
        "Surrogate\nošetření půdy",
        "Surrogate\nkvalita GPS",
    ]
    times_ms = [
        500.0,
        results["out_collision_risk"]["inference_ms"],
        results["out_speed"]["inference_ms"],
        results["out_needs_treatment"]["inference_ms"],
        results["out_nav_quality"]["inference_ms"],
    ]
    bar_colors = [COLORS["accent"]] + [COLORS["primary"]] * 4

    bars = ax.bar(labels, times_ms, color=bar_colors, width=0.5, alpha=0.9)
    for bar, t in zip(bars, times_ms):
        label = f"{t:.0f} ms" if t >= 1 else f"{t:.3f} ms"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Latence [ms]  (log škála)")
    ax.set_yscale("log")
    ax.set_ylim(0.0001, 2000)
    ax.axhline(16.67, color=COLORS["secondary"], linestyle="--",
               alpha=0.6, label="60 FPS limit (16.67 ms)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("results/grafy/04_latency_benchmark.png")
    plt.close()
    print("✓ 04_latency_benchmark.png")


def plot_summary_table(results):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    fig.suptitle("Souhrn výsledků Surrogate Modelu",
                 fontsize=14, fontweight="bold")

    rename = {
        "out_collision_risk":  "Riziko kolize",
        "out_speed":           "Optimální rychlost",
        "out_needs_treatment": "Potřeba ošetření",
        "out_nav_quality":     "Kvalita navigace",
    }

    rows = []
    for target, r in results.items():
        acc_str = f"R² = {r['r2']:.4f}" if r["model_type"] == "regressor" \
                  else f"{r['accuracy']*100:.2f}%"
        rows.append([
            rename[target],
            r["model_type"].capitalize(),
            acc_str,
            f"{r['train_time_s']:.2f} s",
            f"{r['inference_ms']:.4f} ms",
        ])

    col_labels = ["Výstup", "Typ modelu", "Přesnost", "Čas tréninku", "Inference"]
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor(COLORS["primary"])
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#F8F9FA" if i % 2 == 0 else "white")

    plt.savefig("results/grafy/05_summary_table.png")
    plt.close()
    print("✓ 05_summary_table.png")


if __name__ == "__main__":
    print("Načítám výsledky tréninku...")
    results = joblib.load("results/training_results.pkl")
    lc      = joblib.load("results/learning_curves.pkl")

    print("\nGeneruji grafy:")
    plot_feature_importance(results)
    plot_learning_curves(lc)
    plot_predictions_vs_real(results)
    plot_latency_benchmark(results)
    plot_summary_table(results)

    print("\n✓ Hotovo — results/grafy/")