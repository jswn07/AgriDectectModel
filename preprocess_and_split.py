import os
import random
from PIL import Image

# ================= CONFIG =================
RAW_DIR = "raw"
OUTPUT_DIR = "dataset"
IMAGE_SIZE = (224, 224)

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

SEED = 42
random.seed(SEED)
# =========================================

def create_output_dirs(class_names):
    for split in SPLIT_RATIO:
        for cls in class_names:
            path = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(path, exist_ok=True)

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

def collect_classes():
    classes = []

    for crop in os.listdir(RAW_DIR):
        crop_path = os.path.join(RAW_DIR, crop)

        for disease in os.listdir(crop_path):
            class_name = f"{crop.capitalize()}_{disease.replace(' ', '').replace('_', '')}"
            classes.append((crop, disease, class_name))

    return classes

def preprocess_and_split():
    classes = collect_classes()
    class_names = [c[2] for c in classes]

    create_output_dirs(class_names)

    for crop, disease, class_name in classes:
        src_dir = os.path.join(RAW_DIR, crop, disease)
        images = os.listdir(src_dir)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * SPLIT_RATIO["train"])
        val_end = train_end + int(total * SPLIT_RATIO["val"])

        split_map = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, files in split_map.items():
            for file in files:
                src_path = os.path.join(src_dir, file)
                dst_path = os.path.join(OUTPUT_DIR, split, class_name, file)

                if not is_valid_image(src_path):
                    print(f"Skipping corrupted image: {src_path}")
                    continue

                try:
                    img = Image.open(src_path).convert("RGB")
                    img = img.resize(IMAGE_SIZE)
                    img.save(dst_path)
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")

    print("Preprocessing and dataset splitting completed successfully.")

if __name__ == "__main__":
    preprocess_and_split()
