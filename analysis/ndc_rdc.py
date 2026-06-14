import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MetricRow:
    rate_index: int
    avg_accuracy: float
    avg_accuracy_std: float
    ndc: int
    delta_rdc: float
    delta_rdc_std: float


def load_recall_array(
    recall_path: str | None,
    correct_path: str | None,
    total_path: str | None,
) -> np.ndarray:
    if recall_path is not None:
        return np.load(recall_path).astype(np.float64)

    if correct_path is None or total_path is None:
        raise ValueError("Provide either --recall or both --correct and --total.")

    correct = np.load(correct_path).astype(np.float64)
    total = np.load(total_path).astype(np.float64)
    return correct / total * 100.0


def summarize_sweep(
    recall_sweep: np.ndarray,
    baseline_recall: np.ndarray,
) -> list[MetricRow]:
    if recall_sweep.ndim != 3:
        raise ValueError("Expected recall_sweep with shape [num_rates, num_seeds, num_classes].")
    if baseline_recall.ndim != 1:
        raise ValueError("Expected baseline_recall with shape [num_classes].")
    if recall_sweep.shape[-1] != baseline_recall.shape[0]:
        raise ValueError("Class dimension mismatch between recall_sweep and baseline_recall.")

    class_mean_by_rate = recall_sweep.mean(axis=1)
    rows: list[MetricRow] = []

    for rate_index, class_recall in enumerate(class_mean_by_rate):
        delta = class_recall - baseline_recall
        degraded = delta[delta < 0]
        rows.append(
            MetricRow(
                rate_index=rate_index,
                avg_accuracy=float(recall_sweep[rate_index].mean()),
                avg_accuracy_std=float(recall_sweep[rate_index].mean(axis=1).std()),
                ndc=int((delta < 0).sum()),
                delta_rdc=float(degraded.mean()) if degraded.size else 0.0,
                delta_rdc_std=float(degraded.std()) if degraded.size else 0.0,
            )
        )

    return rows


def print_summary(rows: list[MetricRow]) -> None:
    print("rate_index,avg_accuracy,avg_accuracy_std,ndc,delta_rdc,delta_rdc_std")
    for row in rows:
        print(
            f"{row.rate_index},"
            f"{row.avg_accuracy:.4f},"
            f"{row.avg_accuracy_std:.4f},"
            f"{row.ndc},"
            f"{row.delta_rdc:.4f},"
            f"{row.delta_rdc_std:.4f}"
        )

    best = max(rows, key=lambda row: row.avg_accuracy)
    print()
    print(
        "best_rate_index="
        f"{best.rate_index} "
        f"(avg_accuracy={best.avg_accuracy:.4f}, ndc={best.ndc}, delta_rdc={best.delta_rdc:.4f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize NDC and delta-RDC from DropMix rate sweeps."
    )
    parser.add_argument("--recall", type=Path, default=None, help="Path to recall npy array.")
    parser.add_argument("--correct", type=Path, default=None, help="Path to correct-count npy array.")
    parser.add_argument("--total", type=Path, default=None, help="Path to total-count npy array.")
    parser.add_argument(
        "--baseline-recall",
        type=Path,
        required=True,
        help="Path to a 1D npy array with vanilla per-class recall.",
    )
    parser.add_argument(
        "--reverse-rates",
        action="store_true",
        help="Reverse the first axis before computing metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    recall_sweep = load_recall_array(
        recall_path=str(args.recall) if args.recall else None,
        correct_path=str(args.correct) if args.correct else None,
        total_path=str(args.total) if args.total else None,
    )
    baseline_recall = np.load(args.baseline_recall).astype(np.float64)

    if args.reverse_rates:
        recall_sweep = recall_sweep[::-1]

    rows = summarize_sweep(recall_sweep=recall_sweep, baseline_recall=baseline_recall)
    print_summary(rows)


if __name__ == "__main__":
    main()
