"""Preprocess PPE dataset: convert YOLO bounding box annotations to per-image binary concept CSV."""
import os
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle", 9: "no_gloves", 10: "no_boots",
}
PPE_POSITIVE = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles"}
PPE_NEGATIVE = {7: "no_helmet", 8: "no_goggle", 9: "no_gloves", 10: "no_boots"}

CONCEPTS = ["has_helmet", "has_gloves", "has_vest", "has_boots", "has_goggles"]

rows = []
for split in ["train", "val", "test"]:
    lbl_dir = os.path.join(DATA_DIR, "labels", split)
    img_dir = os.path.join(DATA_DIR, "images", split)
    if not os.path.isdir(lbl_dir):
        continue
    for f in sorted(os.listdir(lbl_dir)):
        if not f.endswith(".txt"):
            continue
        stem = f.replace(".txt", "")

        # Find image file (jpg or png)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(img_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = os.path.join("images", split, stem + ext)
                break
        if img_path is None:
            continue

        # Parse bounding boxes
        classes_present = set()
        for line in open(os.path.join(lbl_dir, f)):
            parts = line.strip().split()
            if len(parts) >= 5:
                classes_present.add(int(parts[0]))

        # Binary concepts: PPE item detected in image
        has_helmet = int(0 in classes_present)
        has_gloves = int(1 in classes_present)
        has_vest = int(2 in classes_present)
        has_boots = int(3 in classes_present)
        has_goggles = int(4 in classes_present)

        # Also check negative indicators
        missing_any = any(c in classes_present for c in PPE_NEGATIVE)

        # Label: compliant if all PPE present and no missing indicators
        compliant = int(
            has_helmet and has_gloves and has_vest and has_boots and has_goggles
            and not missing_any
        )

        rows.append({
            "image_path": img_path,
            "split": split,
            "has_helmet": has_helmet,
            "has_gloves": has_gloves,
            "has_vest": has_vest,
            "has_boots": has_boots,
            "has_goggles": has_goggles,
            "compliant": compliant,
        })

df = pd.DataFrame(rows)
out_path = os.path.join(DATA_DIR, "ppe_cbm.csv")
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} images to {out_path}")
print(f"Splits: {df.groupby('split').size().to_dict()}")
print(f"Compliant: {df['compliant'].sum()} ({df['compliant'].mean()*100:.1f}%)")
print(f"Non-compliant: {(1-df['compliant']).sum()} ({(1-df['compliant']).mean()*100:.1f}%)")
print(f"\nConcept prevalence:")
for c in CONCEPTS:
    print(f"  {c:15s}: {df[c].sum():5d} ({df[c].mean()*100:.1f}%)")
