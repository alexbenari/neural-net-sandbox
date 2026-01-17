import csv
import os
import statistics
import webbrowser
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")

VALUE_BIN_SIZE = 100_000_000
VALUE_MAX = 1_000_000_000


def _load_values(csv_path):
    values = []
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            value_str, label_str = row
            value = int(value_str)
            values.append(value)
            labels.append(int(label_str))
    return values, labels


def _value_bins(values):
    bins = [0 for _ in range(VALUE_MAX // VALUE_BIN_SIZE)]
    for value in values:
        index = min(value // VALUE_BIN_SIZE, len(bins) - 1)
        bins[index] += 1
    labels = []
    for i in range(len(bins)):
        start = i * VALUE_BIN_SIZE
        end = (i + 1) * VALUE_BIN_SIZE
        labels.append(f"{start // 1_000_000}M-{end // 1_000_000}M")
    return labels, bins


def _label_distribution(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    items = sorted(counts.items())
    x_labels = [str(item[0]) for item in items]
    y_values = [item[1] for item in items]
    return x_labels, y_values


def _stats(values, labels):
    return {
        "count": len(values),
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "mean": statistics.mean(values) if values else 0,
        "label_min": min(labels) if labels else 0,
        "label_max": max(labels) if labels else 0,
        "label_mean": statistics.mean(labels) if labels else 0,
    }


def _save_bar_chart(title, x_labels, y_values, output_path):
    fig_width = max(8, len(x_labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.bar(range(len(y_values)), y_values, color="#3b82f6")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_data_probes(output_path=None, open_browser=True):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join("tests", "data-probes")
    if output_path is None:
        output_path = os.path.join(base_dir, f"probes-{timestamp}.html")
    assets_dir = os.path.join(base_dir, f"probes-{timestamp}-assets")
    os.makedirs(assets_dir, exist_ok=True)

    formats = [
        ("int", "int"),
        ("digit", "int"),
        ("digit1h", "int"),
        ("binary", "int"),
    ]
    splits = ["train", "test", "eval"]
    sections = []

    for format_name, prefix in formats:
        for split in splits:
            csv_path = os.path.join("data", f"{prefix}-{split}.csv")
            values, labels = _load_values(csv_path)
            stats = _stats(values, labels)
            range_labels, range_counts = _value_bins(values)
            output_labels, output_counts = _label_distribution(labels)
            range_chart_path = os.path.join(
                assets_dir, f"{format_name}-{split}-ranges.png"
            )
            output_chart_path = os.path.join(
                assets_dir, f"{format_name}-{split}-outputs.png"
            )
            _save_bar_chart(
                f"{format_name}-{split}: value ranges",
                range_labels,
                range_counts,
                range_chart_path,
            )
            _save_bar_chart(
                f"{format_name}-{split}: output distribution",
                output_labels,
                output_counts,
                output_chart_path,
            )
            sections.append(
                {
                    "format": format_name,
                    "split": split,
                    "stats": stats,
                    "range_chart": os.path.relpath(range_chart_path, base_dir),
                    "output_chart": os.path.relpath(output_chart_path, base_dir),
                }
            )

    html_parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<title>Data Probes</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; color: #111827; }",
        "h1 { margin-bottom: 8px; }",
        ".section { margin: 32px 0; padding-bottom: 24px; border-bottom: 1px solid #e5e7eb; }",
        ".stats { font-size: 14px; margin: 8px 0 16px; }",
        ".charts { display: grid; gap: 16px; }",
        ".charts img { max-width: 100%; height: auto; border: 1px solid #e5e7eb; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Dataset Probes</h1>",
        "<p>Static probes for dataset distributions and label stats.</p>",
    ]

    for section in sections:
        stats = section["stats"]
        html_parts.append("<div class='section'>")
        html_parts.append(f"<h2>{section['format']} / {section['split']}</h2>")
        html_parts.append(
            "<div class='stats'>"
            f"count: {stats['count']}, "
            f"min: {stats['min']}, max: {stats['max']}, mean: {stats['mean']:.2f}, "
            f"label min: {stats['label_min']}, label max: {stats['label_max']}, "
            f"label mean: {stats['label_mean']:.2f}"
            "</div>"
        )
        html_parts.append("<div class='charts'>")
        html_parts.append(f"<img src='{section['range_chart']}' alt='value ranges' />")
        html_parts.append(
            f"<img src='{section['output_chart']}' alt='output distribution' />"
        )
        html_parts.append("</div>")
        html_parts.append("</div>")

    html_parts.extend(["</body>", "</html>"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(html_parts))

    if open_browser:
        abs_path = os.path.abspath(output_path)
        webbrowser.open(f"file://{abs_path}")

    return output_path


if __name__ == "__main__":
    run_data_probes()
