fold=fold_5
check=train
type=SEGMENTATION
# python main.py --path /home/adminn/Documents/CARIES/DATA/SEGMENTATION
echo $fold

python yolo2coco_segmentation.py \
    --yolo_image_path=/home/adminn/Documents/CARIES/DATA/$type/K_folds/$fold/$check/images \
    --yolo_label_path=/home/adminn/Documents/CARIES/DATA/$type/K_folds/$fold/$check/labels \
    --classes=caries --coco_output_dir=output --name_json=$check.json