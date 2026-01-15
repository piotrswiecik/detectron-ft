import csv

import matplotlib.pyplot as plt


def load_data(csv_path):
    thresholds = []
    avg_ious = []
    min_ious = []
    max_ious = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            thresholds.append(float(row["threshold"]))
            avg_ious.append(float(row["average_iou"]))
            min_ious.append(float(row["min_iou"]))
            max_ious.append(float(row["max_iou"]))

    return thresholds, avg_ious, min_ious, max_ious


def plot_iou_curve(csv_path, output_path=None):
    thresholds, avg_ious, min_ious, max_ious = load_data(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(thresholds, min_ious, max_ious, alpha=0.3, label="Min-Max Range")
    ax.plot(thresholds, avg_ious, linewidth=2, marker="o", markersize=3, label="Average IoU")

    ax.set_xlabel("Score Threshold")
    ax.set_ylabel("IoU")
    ax.set_title("IoU vs Score Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    plot_iou_curve("iou_curve_results.csv", output_path="iou_curve_plot.png")
