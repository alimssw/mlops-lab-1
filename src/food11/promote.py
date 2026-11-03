import argparse
import math
import os
from pathlib import Path

from mlflow import MlflowClient

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASETS = {
    "processed": DATA_DIR / "food11_processed",
    "mini": DATA_DIR / "food11_processed_mini",
}
MODEL_NAME = "food11"
METRIC = "test_accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    return parser.parse_args()


def test_split_size(run) -> int:
    root = DATASETS[run.data.params["dataset"]] / "evaluation"
    return sum(1 for _ in root.rglob("*.jpg"))


def main() -> None:
    args = parse_args()
    client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

    candidate = client.get_model_version(MODEL_NAME, str(args.version))
    acc_candidate = client.get_run(candidate.run_id).data.metrics[METRIC]

    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_run = client.get_run(champion.run_id)
    acc_champion = champion_run.data.metrics[METRIC]

    n = test_split_size(champion_run)
    se = math.sqrt(acc_champion * (1 - acc_champion) / n)
    threshold = acc_champion + 2 * se

    print(f"candidate: version {candidate.version}, run {candidate.run_id[:8]}, {METRIC} {acc_candidate:.4f}")
    print(f"champion: version {champion.version}, run {champion.run_id[:8]}, {METRIC} {acc_champion:.4f}")
    print(f"n_test {n}, se {se:.4f}, threshold {threshold:.4f} (champion + 2*se)")

    if acc_candidate >= threshold:
        decision = "champion"
        client.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
        print(f"decision: promoted, champion alias moved from version {champion.version} to {candidate.version}")
    else:
        decision = "challenger"
        client.set_registered_model_alias(MODEL_NAME, "challenger", candidate.version)
        print(f"decision: not promoted, challenger alias set on version {candidate.version}, champion stays at {champion.version}")

    tags = {
        "gate_decision": decision,
        "gate_candidate_accuracy": f"{acc_candidate:.4f}",
        "gate_champion_accuracy": f"{acc_champion:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, candidate.version, key, value)


if __name__ == "__main__":
    main()
