import os
import random
import yaml
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

# Configuration
path = "."
ksplit = 5

def analyze_dataset_pool(dataset_path, ksplit=5):
    """Analyze the merged dataset pool and simulate K-Fold splits"""
    # Collect all label files from train/val/test splits
    labels = []
    for split in ['train', 'val', 'test']:
        labels.extend(sorted((dataset_path / split / 'labels').glob("*.txt")))

    print(f"Total label files: {len(labels)}")

    # Read data.yaml for classes
    yaml_file = dataset_path / 'data.yaml'
    with open(yaml_file, encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = list(range(len(classes)))

    # Create DataFrame for labels
    index = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)

    # Count instances per class per image
    for label in labels:
        lbl_counter = Counter()
        with open(label) as lf:
            lines = lf.readlines()
        for line in lines:
            lbl_counter[int(float(line.split(" ", 1)[0]))] += 1
        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0).astype(int)

    # Overall pool analysis
    total_instances = labels_df.sum()
    total_images = len(labels_df)

    print("\n=== MERGED DATA POOL ANALYSIS ===")
    print(f"Total annotated images: {total_images}")
    print(f"Total instances: {total_instances.sum()}")
    print("Class distribution:")
    for cls_id, name in enumerate(classes):
        count = total_instances[cls_id]
        pct = (count / total_instances.sum()) * 100
        print(f"  {name} (class {cls_id}): {int(count)} instances ({pct:.1f}%)")

    # Per-image class presence
    images_with_classes = {cls: 0 for cls in cls_idx}
    images_with_multiple = 0
    for _, row in labels_df.iterrows():
        present_classes = [cls for cls in cls_idx if row[cls] > 0]
        for cls in present_classes:
            images_with_classes[cls] += 1
        if len(present_classes) > 1:
            images_with_multiple += 1

    print("\nImages containing each class:")
    for cls_id, name in enumerate(classes):
        count = images_with_classes[cls_id]
        pct = (count / total_images) * 100
        print(f"  {name}: {count} images ({pct:.1f}%)")
    print(f"Images with multiple classes: {images_with_multiple} ({images_with_multiple/total_images*100:.1f}%)")

    # Simulate K-Fold split for analysis
    random.seed(0)
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)
    kfolds = list(kf.split(labels_df))

    print("\n=== K-FOLD SPLIT ANALYSIS (SIMULATION) ===")
    fold_summaries = []
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # Detailed fold stats
        fold_summary = {
            'fold': n,
            'train_images': len(train_indices),
            'val_images': len(val_indices),
            'train_instances': train_totals.sum(),
            'val_instances': val_totals.sum()
        }
        for cls_id in cls_idx:
            fold_summary[f'train_{cls_id}'] = train_totals[cls_id]
            fold_summary[f'val_{cls_id}'] = val_totals[cls_id]
        fold_summaries.append(fold_summary)

        print(f"\nFold {n}:")
        print(f"  Train: {len(train_indices)} images, {train_totals.sum():.0f} instances")
        print(f"  Val: {len(val_indices)} images, {val_totals.sum():.0f} instances")
        for cls_id, name in enumerate(classes):
            train_pct = (train_totals[cls_id] / train_totals.sum()) * 100 if train_totals.sum() > 0 else 0
            val_pct = (val_totals[cls_id] / val_totals.sum()) * 100 if val_totals.sum() > 0 else 0
            print(f"    {name} - Train: {train_totals[cls_id]:.0f} ({train_pct:.1f}%), Val: {val_totals[cls_id]:.0f} ({val_pct:.1f}%)")

    # Summary statistics
    val_images = [s['val_images'] for s in fold_summaries]
    val_instances = [s['val_instances'] for s in fold_summaries]

    print("\n=== K-FOLD SUMMARY STATISTICS ===")
    print(f"Average validation images per fold: {sum(val_images)/len(val_images):.1f} ± {pd.Series(val_images).std():.1f}")
    print(f"Validation images range: {min(val_images)} - {max(val_images)}")
    print(f"Average validation instances per fold: {sum(val_instances)/len(val_instances):.1f} ± {pd.Series(val_instances).std():.1f}")

    for cls_id, name in enumerate(classes):
        val_cls = [s[f'val_{cls_id}'] for s in fold_summaries]
        print(f"Average {name} val instances: {sum(val_cls)/len(val_cls):.1f} ± {pd.Series(val_cls).std():.1f}")

    return labels_df, classes, kfolds

def create_kfold_splits(dataset_path, labels_df, classes, kfolds, ksplit=5):
    """Create K-Fold splits and directories using pre-computed analysis"""
    cls_idx = list(range(len(classes)))

    save_path = Path(dataset_path / "kfold")

    # Check if k-fold splits already exist
    if save_path.exists():
        print(f"\n=== USING EXISTING K-FOLD SPLITS ===")
        print(f"Found existing splits at: {save_path}")

        # Verify the existing splits are valid
        ds_yamls = []
        for i in range(1, ksplit + 1):
            yaml_file = save_path / f"split_{i}" / f"split_{i}_dataset.yaml"
            if not yaml_file.exists():
                print(f"Warning: Missing YAML file {yaml_file}, recreating splits...")
                save_path = None  # Force recreation
                break
            ds_yamls.append(str(yaml_file))

        if len(ds_yamls) == ksplit:
            print(f"Using {len(ds_yamls)} existing fold configurations")
            return ds_yamls

    # Create new splits if they don't exist or are invalid
    print(f"\n=== CREATING NEW K-FOLD SPLITS ===")
    print(f"Creating splits at: {save_path}")

    save_path.mkdir(parents=True, exist_ok=True)

    # Create folds DataFrame
    index = labels_df.index
    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df.loc[labels_df.iloc[train].index, f"split_{i}"] = "train"
        folds_df.loc[labels_df.iloc[val].index, f"split_{i}"] = "val"

    # Calculate label distribution
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    # Collect all .jpg images from train/val/test splits
    images = []
    for split in ['train', 'val', 'test']:
        images.extend(sorted((dataset_path / split / 'images').glob("*.jpg")))

    ds_yamls = []

    for split in folds_df.columns:
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(str(dataset_yaml))

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.resolve().as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )

    # Create symlinks for files (efficient alternative to copying)
    for image in tqdm(images, desc="Creating symlinks"):
        label = dataset_path / image.parent.parent.name / 'labels' / f"{image.stem}.txt"
        if not label.exists():
            continue
        for split, k_split in folds_df.loc[image.stem].items():
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"
            # Create relative symlinks instead of absolute ones
            img_rel_path = os.path.relpath(image, img_to_path)
            lbl_rel_path = os.path.relpath(label, lbl_to_path)
            os.symlink(img_rel_path, img_to_path / image.name)
            os.symlink(lbl_rel_path, lbl_to_path / label.name)

    # Save records
    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

    # Save detailed analysis report (only when creating new splits)
    fold_summaries = []
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        fold_summary = {
            'fold': n,
            'train_images': len(train_indices),
            'val_images': len(val_indices),
            'train_instances': train_totals.sum(),
            'val_instances': val_totals.sum()
        }
        for cls_id in cls_idx:
            fold_summary[f'train_{cls_id}'] = train_totals[cls_id]
            fold_summary[f'val_{cls_id}'] = val_totals[cls_id]
        fold_summaries.append(fold_summary)

    total_instances = labels_df.sum()
    total_images = len(labels_df)
    images_with_classes = {cls: 0 for cls in cls_idx}
    images_with_multiple = 0
    for _, row in labels_df.iterrows():
        present_classes = [cls for cls in cls_idx if row[cls] > 0]
        for cls in present_classes:
            images_with_classes[cls] += 1
        if len(present_classes) > 1:
            images_with_multiple += 1

    with open(save_path / "kfold_analysis_report.txt", "w") as f:
        f.write("K-FOLD CROSS-VALIDATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("MERGED DATA POOL ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total annotated images: {total_images}\n")
        f.write(f"Total instances: {total_instances.sum()}\n")
        f.write("Class distribution:\n")
        for cls_id, name in enumerate(classes):
            count = total_instances[cls_id]
            pct = (count / total_instances.sum()) * 100
            f.write(f"  {name} (class {cls_id}): {int(count)} instances ({pct:.1f}%)\n")

        f.write("\nImages containing each class:\n")
        for cls_id, name in enumerate(classes):
            count = images_with_classes[cls_id]
            pct = (count / total_images) * 100
            f.write(f"  {name}: {count} images ({pct:.1f}%)\n")
        f.write(f"Images with multiple classes: {images_with_multiple} ({images_with_multiple/total_images*100:.1f}%)\n")

        f.write("\nK-FOLD SPLIT ANALYSIS\n")
        f.write("-" * 25 + "\n")
        for summary in fold_summaries:
            f.write(f"\nFold {summary['fold']}:\n")
            f.write(f"  Train: {summary['train_images']} images, {summary['train_instances']:.0f} instances\n")
            f.write(f"  Val: {summary['val_images']} images, {summary['val_instances']:.0f} instances\n")
            for cls_id, name in enumerate(classes):
                train_count = summary[f'train_{cls_id}']
                val_count = summary[f'val_{cls_id}']
                train_pct = (train_count / summary['train_instances']) * 100 if summary['train_instances'] > 0 else 0
                val_pct = (val_count / summary['val_instances']) * 100 if summary['val_instances'] > 0 else 0
                f.write(f"    {name} - Train: {train_count:.0f} ({train_pct:.1f}%), Val: {val_count:.0f} ({val_pct:.1f}%)\n")

        f.write("\nSUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        val_images = [s['val_images'] for s in fold_summaries]
        val_instances = [s['val_instances'] for s in fold_summaries]
        f.write(f"Average validation images per fold: {sum(val_images)/len(val_images):.1f} ± {pd.Series(val_images).std():.1f}\n")
        f.write(f"Validation images range: {min(val_images)} - {max(val_images)}\n")
        f.write(f"Average validation instances per fold: {sum(val_instances)/len(val_instances):.1f} ± {pd.Series(val_instances).std():.1f}\n")
        for cls_id, name in enumerate(classes):
            val_cls = [s[f'val_{cls_id}'] for s in fold_summaries]
            f.write(f"Average {name} val instances: {sum(val_cls)/len(val_cls):.1f} ± {pd.Series(val_cls).std():.1f}\n")

    print(f"Created {len(ds_yamls)} fold configurations")
    return ds_yamls

# Main execution
if __name__ == "__main__":
    dataset_path = Path(".")

    # Analyze dataset pool and simulate k-fold splits
    labels_df, classes, kfolds = analyze_dataset_pool(dataset_path, ksplit)

    # Create actual k-fold splits and directories
    ds_yamls = create_kfold_splits(dataset_path, labels_df, classes, kfolds, ksplit)

    print(f"\nK-fold splits created successfully!")
    print(f"Dataset YAML files: {ds_yamls}")
    print(f"Use these YAML files with kfold_train.py to train on each fold.")