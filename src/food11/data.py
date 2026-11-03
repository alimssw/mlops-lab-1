from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "food11_raw"
PROCESSED_DIR = DATA_DIR / "food11_processed"
MINI_DIR = DATA_DIR / "food11_processed_mini"

SPLITS = ["training", "evaluation", "validation"]

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

IMAGE_SIZE = 128
MINI_LIMIT = 100


def category_from_filename(filename: str) -> str:
    return CLASSES[int(filename.split("_", 1)[0]) - 1]


def resize_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.LANCZOS)


def prepare_split(split: str) -> None:
    src_dir = RAW_DIR / split
    mini_counts = {category: 0 for category in CLASSES}

    for image_path in sorted(src_dir.glob("*.jpg")):
        category = category_from_filename(image_path.name)

        with Image.open(image_path) as img:
            resized = resize_image(img)

        out_dir = PROCESSED_DIR / split / category
        out_dir.mkdir(parents=True, exist_ok=True)
        resized.save(out_dir / image_path.name)

        if mini_counts[category] < MINI_LIMIT:
            mini_dir = MINI_DIR / split / category
            mini_dir.mkdir(parents=True, exist_ok=True)
            resized.save(mini_dir / image_path.name)
            mini_counts[category] += 1

    print(f"{split}: processed {sum(1 for _ in (PROCESSED_DIR / split).rglob('*.jpg'))} images")


def main() -> None:
    for split in SPLITS:
        prepare_split(split)


if __name__ == "__main__":
    main()
