import os
import csv
import matplotlib.pyplot as plt



import logging

LOGGER = logging.getLogger(__name__)


def log_metrics_to_csv(csv_path, headers, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new_file = not os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(headers)
        writer.writerows(rows)


def read_metrics_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    headers = rows[0]
    data = list(zip(*rows[1:]))
    return headers, data


def plot_training_curves(csv_path, save_path="./plots/training_curve.png", title="Training Metrics"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    headers, data = read_metrics_csv(csv_path)
    epochs = list(map(int, data[0]))

    plt.figure(figsize=(12, 6))
    for i, header in enumerate(headers[1:], 1):  # Skip 'epoch'
        values = list(map(float, data[i]))
        plt.plot(epochs, values, label=header)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    LOGGER.info(f"Saved plot: {save_path}")
