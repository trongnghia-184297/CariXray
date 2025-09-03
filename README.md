# CariXray Experiments

## 1. Introduction
This repository provides experimental components accompanying the paper:  
**"CariXray: A Periapical X-ray Dataset for Machine Vision-based Dental Caries Recognition."**  
It includes dataset preparation tools, benchmark implementations, and reproducibility resources.

---

## 2. Data

### 2.1. Dataset Access
- **Full dataset access**: The complete CariXray dataset can be obtained at [[link](https://drive.google.com/drive/folders/14Km_y3JvuJesQtbDPFqSZhu2Zgv7xzS9?usp=sharing)].  
- **Sample dataset for experiments**: A small subset of data for quick testing and reproducing experiments can be found at [[link](https://drive.google.com/drive/folders/1Nfj7m_PDUWAgBbEPD3lBJHHklNhpZaTv?usp=sharing)].  

## 2.2 Label format conversion
- **Detection label conversion**: Scripts are provided to convert detection labels from **YOLO Darknet format** to **COCO format**.
    ```
    cd yolo2coco_detection
    bash yolo2coco.bash
    ```


- **Segmentation label conversion**: Scripts are provided to convert segmentation labels from **YOLO Darknet format** to **COCO format**.
    ```
    cd yolo2coco_segmentation
    bash run.bash
    ```


## 2.3 K-fold cross-validation split
- **Cross-validation split**: Tools to generate **k-fold cross-validation** splits (default: 5 folds) are included to ensure reproducible experiments.  
    ```
    cd kfold
    python split_kfold.py
    ```

---

## 3. Benchmarking

| Id | Task        | Model       | Repository/Link |
|----|-------------|-------------|-----------------|
| 1  | Detection   | Faster R-CNN| [link](https://github.com/facebookresearch/detectron2)                 |
| 2  | Detection   | RetinaNet   | [link](https://github.com/facebookresearch/detectron2)                 |
| 3  | Detection   | YOLOv7      | [link](https://github.com/WongKinYiu/yolov7)                |
| 4  | Detection   | YOLOv8      | [link](https://github.com/ultralytics/ultralytics)               |
| 5  | Detection   | YOLOv9      | [link](https://github.com/WongKinYiu/yolov9)                |
| 6  | Detection   | YOLOv10     | [link](https://github.com/ultralytics/ultralytics)                |
| 7  | Detection   | DAMO-YOLO   | [link](https://github.com/tinyvision/DAMO-YOLO)                |
| 8  | Detection   | YOLOR-Based | [link](https://github.com/WongKinYiu/yolov9)                |
| 9  | Segmentation| Mask R-CNN  | [link](https://github.com/facebookresearch/detectron2)                 |
| 10 | Segmentation| FastInst    | [link](https://github.com/junjiehe96/FastInst)                |
| 11 | Segmentation| Mask2Former | [link](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former)                |
| 12 | Segmentation| YOLACT      | [link](https://github.com/dbolya/yolact)                |
| 13 | Segmentation| YOLOv7-seg  | [link](https://github.com/RizwanMunawar/yolov7-segmentation)                |
| 14 | Segmentation| YOLOv8-seg  | [link](https://github.com/ultralytics/ultralytics)                |
| 15 | Segmentation| YOLOv9-seg  | [link](https://github.com/ultralytics/ultralytics)                |
| 16 | Segmentation| YOLOR-Based | [link](https://github.com/WongKinYiu/yolov9)                |


---

## License
This project is released under an open license for research and educational purposes.
