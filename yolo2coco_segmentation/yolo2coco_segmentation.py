import os
import shutil
from PIL import Image, ImageFile
import json
import argparse
from tqdm import tqdm


def check_image_path(path: str):
    if not isinstance(path, str):
        print("Error: The image path (yolo_image_path) must be specified as a string.")
        return False
    if not os.listdir(path):
        print("Warning: The folder containing images (yolo_image_path) is empty. It is recommended to add images "
              "before running the function.")
        return False
    return True


def check_classes(classes_argument):
    if not isinstance(classes_argument, list) or not all(isinstance(cls, str) for cls in classes_argument):
        print("Error: The 'classes' argument must be provided as a list of string elements.")
        return False
    return True


def remove_ipynb_dir(path: str) -> str:
    """Removes the '.ipynb_checkpoints' directory if it exists in the provided path."""

    ipynb_path = os.path.join(path, ".ipynb_checkpoints")
    remove_exist_dir(ipynb_path)

    return path


def remove_exist_dir(path: str) -> str:
    """Deletes the directory if it exists."""

    shutil.rmtree(path, ignore_errors=True)

    return path


def find_right_image(label: str, image_path: str, exts: set) -> str:
    """Finds the image corresponding to the annotation."""

    basename = os.path.splitext(label)[0]
    return next((basename + ext for ext in exts if os.path.exists(os.path.join(image_path, basename + ext))))


def find_image_exts(image_path: str, all_image_exts: set) -> set:
    """Returns the common image formats."""

    return {os.path.splitext(file)[1] for file in os.listdir(remove_ipynb_dir(image_path))
            if os.path.splitext(file)[1].lower() in all_image_exts}


def yolo2coco_segmentation(yolo_image_path: str, yolo_label_path: str, classes: list,
                           coco_output_dir: str) -> None:
    """Converts YOLO dataset for segmentation into COCO dataset for segmentation."""

    all_image_exts = {".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".raw", ".heic", ".webp"}
    if not check_image_path(yolo_image_path) or not check_classes(classes):
        return

    # Создание coco директории
    # os.mkdir(remove_exist_dir(coco_output_dir))
    # coco_image_path = os.path.join(coco_output_dir, "images")
    # os.mkdir(coco_image_path)
    coco_annotation_path = os.path.join(coco_output_dir)
    # os.mkdir(coco_annotation_path)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cls_id, "name": name} for cls_id, name in enumerate(classes, start=1)],
    }
    image_id_counter = 1

    for label in tqdm(os.listdir(remove_ipynb_dir(yolo_label_path))):
        # coco save
        image_name = find_right_image(label=label, image_path=yolo_image_path,
                                      exts=find_image_exts(image_path=yolo_image_path, all_image_exts=all_image_exts))
        cur_image_path = os.path.join(yolo_image_path, image_name)
        width, height = Image.open(cur_image_path).convert('RGB').size
        coco_data["images"].append(
            {"id": image_id_counter,
             "file_name": f"{image_name}",
             "width": width,
             "height": height})

        with open(os.path.join(yolo_label_path, label), "r") as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split()
            class_id = int(data[0]) + 1  # According to COCO standards, numbering starts from one.
            polygon_points = [float(x) for x in data[1:]]
            num_points = len(polygon_points) // 2
            segmentation_per_line = []
            for i in range(num_points):
                segmentation_per_line.append([polygon_points[i * 2] * width, polygon_points[i * 2 + 1] * height])

            # chuyển về toạ độ pixel
            pts = []
            for i in range(num_points):
                x = polygon_points[i * 2] * width
                y = polygon_points[i * 2 + 1] * height
                pts.append([x, y])

            # bbox [x_min, y_min, w, h]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

            annotation = {
                "image_id": image_id_counter,
                "id": len(coco_data["annotations"]) + 1,
                "category_id": class_id,
                "bbox": bbox,
                "area": 0,
                "iscrowd": 0,
                "segmentation": [sum(segmentation_per_line, [])],
            }
            coco_data["annotations"].append(annotation)

        image_id_counter += 1
        # shutil.copy(cur_image_path, os.path.join(coco_image_path, image_name))

    with open(os.path.join(coco_annotation_path, name_json_arg), 'w') as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)

    print(f"The COCO dataset for segmentation has been successfully created. The directory is: {coco_output_dir}")


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parser = argparse.ArgumentParser(description="Perform coco segmentation dataset from yolo segmentation dataset.")
    parser.add_argument("--yolo_image_path", required=True, help="Path to the folder containing images "
                                                                 "from yolo dataset.")
    parser.add_argument("--yolo_label_path", required=True, help="Path to the folder containing labels "
                                                                 "from yolo dataset.")
    parser.add_argument("--classes", required=True, help="List of classes separated by commas.")
    parser.add_argument("--coco_output_dir", required=False, help="output directory name, "
                                                                  "default: 'coco_segmentation_dataset'")
    parser.add_argument("--name_json", required=False, help="output directory name, "
                                                                  "default: 'coco_segmentation_dataset'")

    args = parser.parse_args()

    yolo_image_path_arg = args.yolo_image_path
    yolo_label_path_arg = args.yolo_label_path
    classes_arg = args.classes.split(",")
    coco_output_dir_arg = args.coco_output_dir
    name_json_arg = args.name_json
    if not coco_output_dir_arg:
        coco_output_dir_arg = "coco_segmentation_dataset"

    yolo2coco_segmentation(yolo_image_path=yolo_image_path_arg, yolo_label_path=yolo_label_path_arg,
                           classes=classes_arg, coco_output_dir=coco_output_dir_arg)
