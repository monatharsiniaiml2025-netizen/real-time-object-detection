from pathlib import Path
import shutil

source = Path("source_dataset")
target = Path("custom_dataset")

# Original class ID -> new class ID
mapping = {
    0: 1,  # bottle
    2: 0,  # phone -> cellphone
    3: 2   # remote
}

for split in ["train", "valid", "test"]:
    image_dir = source / split / "images"
    label_dir = source / split / "labels"

    out_images = target / "images" / split
    out_labels = target / "labels" / split

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for label_file in label_dir.glob("*.txt"):
        new_lines = []

        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue

            old_id = int(parts[0])
            if old_id in mapping:
                parts[0] = str(mapping[old_id])
                new_lines.append(" ".join(parts))

        if new_lines:
            image_file = next(
                (image_dir / (label_file.stem + ext)
                 for ext in [".jpg", ".jpeg", ".png"]
                 if (image_dir / (label_file.stem + ext)).exists()),
                None
            )

            if image_file:
                shutil.copy2(image_file, out_images / image_file.name)
                (out_labels / label_file.name).write_text(
                    "\n".join(new_lines) + "\n"
                )

    print(f"{split}: completed")

print("3-class dataset copy created successfully.")
