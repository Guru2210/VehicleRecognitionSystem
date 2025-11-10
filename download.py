from PIL import Image
import os
import sys
import getopt
import pickle
from pathlib import Path

"""
download.py

Two modes:
1) If -d/--dataset is provided (path to archive or archive/Dataset), traverse the dataset tree
   and resize images into ./resized and create pickles/jsondata.pkl containing per-image metadata.
2) Otherwise, original behaviour (if image_downloader is available) is preserved.

Output:
- resized/  (images resized to 224x224)
- pickles/jsondata.pkl (list of dicts: {'resized_name', 'label', 'orig_path'})
"""

def usage():
    print("Usage: python download.py [-n NUM] [-d DATASET_ROOT]")
    print("  -n NUM            optional: max number of images to process (overall)")
    print("  -d DATASET_ROOT   optional: path to dataset root (archive or archive/Dataset)")
    print("If -d is not provided the script falls back to the original downloader/resizer behaviour if available.")


def collect_images_from_dataset(dataset_root):
    """Traverse dataset_root and collect (path, label) pairs.
    Expected format: dataset_root/Make/Model/*.jpg
    Label will be 'Make_Model'."""
    image_entries = []
    # If user passed archive root, prefer archive/Dataset if exists
    if os.path.isdir(os.path.join(dataset_root, 'Dataset')):
        dataset_root = os.path.join(dataset_root, 'Dataset')

    for make in sorted(os.listdir(dataset_root)):
        make_dir = os.path.join(dataset_root, make)
        if not os.path.isdir(make_dir):
            continue
        for model in sorted(os.listdir(make_dir)):
            model_dir = os.path.join(make_dir, model)
            if not os.path.isdir(model_dir):
                continue
            label = f"{make}_{model}"
            for fname in sorted(os.listdir(model_dir)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                image_entries.append((os.path.join(model_dir, fname), label))
    return image_entries


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def process_and_resize(image_entries, n=None, out_dir='resized'):
    ensure_dir(out_dir)
    ensure_dir('pickles')

    jsondata = []
    count = 0
    for i, (orig_path, label) in enumerate(image_entries):
        if n is not None and count >= n:
            break
        try:
            img = Image.open(orig_path).convert('RGB')
        except Exception as e:
            print(f"Skipping {orig_path}: {e}")
            continue

        new_name = f"{i:06d}_" + os.path.basename(orig_path)
        out_path = os.path.join(out_dir, new_name)
        try:
            img = img.resize((224, 224))
            img.save(out_path)
            img.close()
        except Exception as e:
            print(f"Failed saving {out_path}: {e}")
            continue

        jsondata.append({'resized_name': new_name, 'label': label, 'orig_path': orig_path})
        count += 1

    # Save jsondata.pkl
    json_pkl = os.path.join('pickles', 'jsondata.pkl')
    with open(json_pkl, 'wb') as f:
        pickle.dump(jsondata, f)

    print(f"Processed and resized {len(jsondata)} images -> {out_dir}")
    print(f"Saved jsondata to {json_pkl}")
    return jsondata


def main(argv):
    n = None
    dataset_root = None
    try:
        opts, args = getopt.getopt(argv, "n:d:h", ['help', 'dataset='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-n'):
            try:
                n = int(arg)
            except ValueError:
                print("-n must be an integer")
                sys.exit(2)
        elif opt in ('-d', '--dataset'):
            dataset_root = arg
        elif opt in ('-h', '--help'):
            usage()
            sys.exit()

    if dataset_root:
        if not os.path.exists(dataset_root):
            print(f"Dataset root {dataset_root} not found")
            sys.exit(2)
        entries = collect_images_from_dataset(dataset_root)
        if not entries:
            print(f"No images found under {dataset_root}")
            sys.exit(2)
        process_and_resize(entries, n=n, out_dir='resized')
    else:
        # Fallback to original behavior if image_downloader is available
        try:
            from image_downloader import jsonloader, downloader
            from resize_images import resizer
            if n is None:
                print("All images will be downloaded")
            downloader(n, jsonloader())
            resizer(n)
        except Exception as e:
            print("No dataset provided and image_downloader/resizer fallback failed:", e)
            usage()
            sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])