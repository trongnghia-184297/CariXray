# CariXray Experiments

## 1. Introduction
This repository provides experimental components accompanying the paper:  
**"CariXray: A Periapical X-ray Dataset for Machine Vision-based Dental Caries Recognition."**  
It includes dataset preparation tools, benchmark implementations, and reproducibility resources.

---

## 2. Data

- **Full dataset access**: The complete CariXray dataset can be obtained at [link].  
- **Sample dataset for experiments**: A small subset of data for quick testing and reproducing experiments can be found at [link].  
- **Detection label conversion**: Scripts are provided to convert detection labels from **YOLO Darknet format** to **COCO format**.  
- **Segmentation label conversion**: Scripts are provided to convert segmentation labels from **YOLO Darknet format** to **COCO format**.  
- **Cross-validation split**: Tools to generate **k-fold cross-validation** splits (default: 5 folds) are included to ensure reproducible experiments.  

---

## 3. Benchmarking

| Id | Task        | Model       | Repository/Link |
|----|-------------|-------------|-----------------|
| 1  | Detection   | YOLOv8      | [Ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics) |
| 2  | Detection   | YOLOv9      | [YOLOv9 repo](https://github.com/WongKinYiu/yolov9) |
| 3  | Detection   | YOLOR       | [WongKinYiu/yolor](https://github.com/WongKinYiu/yolor) |
| 4  | Detection   | Faster R-CNN| [Detectron2](https://github.com/facebookresearch/detectron2) |
| 5  | Detection   | RetinaNet   | [Detectron2](https://github.com/facebookresearch/detectron2) |
| 6  | Segmentation| Mask R-CNN  | [Detectron2](https://github.com/facebookresearch/detectron2) |
| 7  | Segmentation| Mask2Former | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) |
| 8  | Segmentation| YOLACT      | [dbolya/yolact](https://github.com/dbolya/yolact) |
| 9  | Segmentation| FastInst    | [myownskyW7/FastInst](https://github.com/zhanghang1989/FastInst) |
| 10 | Segmentation| YOLOv8-seg  | [Ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics) |
| 11 | Segmentation| YOLOv9-seg  | [YOLOv9 repo](https://github.com/WongKinYiu/yolov9) |
| 12 | Segmentation| YOLOR-based | [WongKinYiu/yolor](https://github.com/WongKinYiu/yolor) |

---

## License
This project is released under an open license for research and educational purposes.
