# python main.py --path /home/nghia/Documents/Caries/20240603_data_crawling_review/20240823_data_test_detect/yolo_darknet_tococo/test_json_clinic_update --output test_clinic.json
# python main.py --path /home/nghia/Documents/Caries/20240603_data_crawling_review/20240823_data_test_detect/yolo_darknet_tococo/test_json_crawling_update --output test_crawling.json

fold=fold_5
check=train

echo $fold

python main.py --path /home/adminn/Documents/CARIES/DATA/DETECTION/K_folds/$fold/$check/total \
    --output $fold.json