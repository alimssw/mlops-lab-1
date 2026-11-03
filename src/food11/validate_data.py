import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

if __package__:
    from .data import CLASSES, IMAGE_SIZE, SPLITS
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.food11.data import CLASSES, IMAGE_SIZE, SPLITS


def list_images(root: Path) -> dict[str, dict[str, list[Path]]]:
    return {
        split: {
            category: sorted(p for p in (root / split / category).glob("*") if p.is_file())
            for category in CLASSES
            if (root / split / category).is_dir()
        }
        for split in SPLITS
    }


def check_class_folders(root: Path) -> list[str]:
    problems = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            problems.append(f"{split}: missing")
            continue
        found = {p.name for p in split_dir.iterdir() if p.is_dir()}
        for name in sorted(set(CLASSES) - found):
            problems.append(f"{split}: missing class folder {name!r}")
        for name in sorted(found - set(CLASSES)):
            problems.append(f"{split}: unexpected folder {name!r}")
    return problems


def check_images(images: dict[str, dict[str, list[Path]]]) -> list[str]:
    problems = []
    for split, categories in images.items():
        for category, paths in categories.items():
            for path in paths:
                rel = f"{split}/{category}/{path.name}"
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB":
                            problems.append(f"{rel}: mode {img.mode}, expected RGB")
                        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                            problems.append(f"{rel}: size {img.size}, expected ({IMAGE_SIZE}, {IMAGE_SIZE})")
                except (OSError, ValueError, SyntaxError) as exc:
                    problems.append(f"{rel}: cannot open ({exc})")
    return problems


def check_min_per_class(images: dict[str, dict[str, list[Path]]], min_per_class: int) -> list[str]:
    return [
        f"{split}/{category}: {len(paths)} images, minimum is {min_per_class}"
        for split, categories in images.items()
        for category, paths in categories.items()
        if len(paths) < min_per_class
    ]


def check_duplicates(images: dict[str, dict[str, list[Path]]]) -> list[str]:
    seen: dict[str, tuple[str, str]] = {}
    problems = []
    for split, categories in images.items():
        for category, paths in categories.items():
            for path in paths:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                rel = f"{split}/{category}/{path.name}"
                if digest in seen and seen[digest][0] != split:
                    problems.append(f"{rel} duplicates {seen[digest][1]}")
                else:
                    seen.setdefault(digest, (split, rel))
    return problems


def check_class_balance(images: dict[str, dict[str, list[Path]]], tau: float) -> list[str]:
    counts = {split: Counter({c: len(p) for c, p in categories.items()}) for split, categories in images.items()}
    train_total = sum(counts["training"].values())
    val_total = sum(counts["validation"].values())
    if not train_total or not val_total:
        return ["training or validation split is empty"]
    problems = []
    for category in CLASSES:
        train_share = counts["training"][category] / train_total
        val_share = counts["validation"][category] / val_total
        if abs(train_share - val_share) > tau:
            problems.append(f"{category}: training share {train_share:.3f}, validation share {val_share:.3f}, tau {tau}")
    return problems


def run_checks(root: Path, min_per_class: int, tau: float) -> list[tuple[str, list[str]]]:
    images = list_images(root)
    return [
        ("class folders", check_class_folders(root)),
        ("images open, RGB, 128x128", check_images(images)),
        (f"at least {min_per_class} images per class", check_min_per_class(images, min_per_class)),
        ("no image in two splits", check_duplicates(images)),
        (f"train/val class share within {tau}", check_class_balance(images, tau)),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args(argv)

    failed = False
    for name, problems in run_checks(args.root, args.min_per_class, args.tau):
        status = "FAIL" if problems else "PASS"
        failed = failed or bool(problems)
        print(f"{status}  {name}")
        for problem in problems[:10]:
            print(f"      {problem}")
        if len(problems) > 10:
            print(f"      ... {len(problems) - 10} more")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
