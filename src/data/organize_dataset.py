import os
import shutil
from pathlib import Path 

from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2] # adjust depth based on folder level

def get_classes(class_list_path, skip_lines=0):
    class_list = []

    with open(class_list_path, 'r') as f:
        lines = f.readlines()[skip_lines:]  # skip first 2 lines

        for line in lines:
            category = line.strip().split()[0]  # get first word
            class_list.append(category)

    return class_list


def copy_files(split_name, classes, base_input_dir, base_output_dir):
    img_paths = f'{base_input_dir}/{split_name}.txt' # list of paths for split
    labels_path = f'{base_input_dir}/{split_name}_cate.txt' # list of each paths corresponding label

    class_counts = defaultdict(int) # to keep track of image numbering

    with open(img_paths, 'r') as f:
        lines_paths = [line.strip() for line in f]

    with open(labels_path, 'r') as f:
        lines_labels = [int(line.strip()) for line in f]

    for img_path, label in zip(lines_paths, lines_labels):
        # prep target directory
        category = classes[label]
        dest_dir = os.path.join(base_output_dir, split_name, category)
        os.makedirs(dest_dir, exist_ok=True)

        # copy file over
        src_path = os.path.join(base_input_dir, img_path)
        target_path = os.path.join(dest_dir, f'{class_counts[category]:08}.jpg')
        
        print(f"Copying {img_path}")
        shutil.copy(src_path, target_path)

        # increment count for category
        class_counts[category] += 1

    print("Done")


if __name__ == "__main__":
    source_data_root = ROOT / 'data/raw'
    target_data_root = ROOT / 'data/split'
    classes = get_classes(ROOT / 'data/raw/list_category_cloth.txt', skip_lines=2)

    copy_files('train', classes, source_data_root, target_data_root)
    copy_files('val', classes, source_data_root, target_data_root)
    copy_files('test', classes, source_data_root, target_data_root)