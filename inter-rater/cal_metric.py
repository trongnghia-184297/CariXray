from src import *
import pandas as pd
import argparse
import unicodedata
import os
import shutil
import json
from tqdm import tqdm
import numpy as np
import re

class CAL_METRIC():
    def __init__(self, path_label, path_pred, path_image, list_images_target, IoU_thresh):
        self.path_label = path_label
        self.path_pred = path_pred
        self.list_images_target = list_images_target
        self.path_image = path_image

        # list_IoU_thresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        self.list_IoU_thresh = [IoU_thresh]
        # self.list_IoU_thresh = [0.75]
        # self.list_IoU_thresh = [0.25]

    # def extract_image(self, path):
    #     list_elements = list_item(path)
    #     list_images = [item for item in list_elements if os.path.splitext(item)[1] != ".txt"]
    #     return list_images


    def extract_bbox(self, path, h, w):
        coor_yolo = parse_yolo_labels(path)
        coor_xyxy = [yolo_to_xyxy(box, h, w) for box in coor_yolo]
        return coor_xyxy



    def cal_metric(self, TP, FP, FN):
        # Calculate Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        # Calculate Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # Calculate F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision,5), round(recall,5), round(f1_score,5)


    def process(self):
        # data_label = json.load(open(self.path_label, 'r'))
        # data_predict = json.load(open(self.path_pred, 'r'))
        # data_predict_name_img = [item["image_name"] for item in data_predict]
        # # print(data_predict_name_img)

        # print(len(data_predict))
        # # print(data_predict[0])

        # list_images = self.extract_image(self.path_image)

        all_P = []
        all_R = []
        all_F1 = []

        for IoU_thresh in self.list_IoU_thresh:
            TP_all = 0
            FP_all = 0
            FN_all = 0

            list_TP = []
            list_FP = []
            list_FN = []

            list_bbox_FP = []
            list_bbox_FN = []

            for idx_img, image_name in enumerate(tqdm(self.list_images_target)):
                try:
                    path_img = os.path.join(self.path_image, image_name)
                    img = cv2.imread(path_img)
                    h, w = img.shape[0], img.shape[1]
                    path_label_single = os.path.join(self.path_label, os.path.splitext(image_name)[0] + ".txt")
                    path_pred_single = os.path.join(self.path_pred, os.path.splitext(image_name)[0] + ".txt")

                    list_bbox_label = self.extract_bbox(path_label_single, h, w)
                    list_bbox_pred = self.extract_bbox(path_pred_single, h, w)
                except ValueError:
                    print("No pair matching for image: {}".format(image_name))
                    continue

                # INITIAL VARIABLE
                num_total_labels = len(list_bbox_label)
                num_total_preds = len(list_bbox_pred)
                
                TP = 0
                FP = 0
                
                bboxes_FP = []
                bboxes_FN = []
                
                TP_box_list_pred = create_clue(num_total_labels)
                # FN = 0

                
                list_clues_label = create_clue(num_total_labels)
                list_clues_preds = create_clue(num_total_preds)
                
                for idx_bb_pred, bb_pred in enumerate(list_bbox_pred):
                    IoU = 0
                    
                    for idx_bb_label, bb_label in enumerate(list_bbox_label):
                        IOU_cal = calculate_IOU(bb_pred, bb_label)
                        if IOU_cal > IoU:
                            IoU = IOU_cal
                            idx_label_target = idx_bb_label
                            
                    # print(IoU)
                            
                    if IoU >= IoU_thresh:
                        if list_clues_label[idx_label_target] == 1:
                            FP += 1
                            TP_box_list_pred[idx_label_target] = bb_pred
                            
                        else:
                            TP += 1
                            list_clues_label[idx_label_target] += 1
                            TP_box_list_pred[idx_label_target] = bb_pred
                        
                    elif IoU < IoU_thresh:
                        FP += 1    
                
                FN = list_clues_label.count(0)
                    
                # print("TP: {}".format(TP))
                # print("FP: {}".format(FP))
                # print("FN: {}".format(FN))

                # Update total    
                TP_all += TP
                FP_all += FP
                FN_all += FN
                
                list_TP.append(TP)
                list_FP.append(FP)
                list_FN.append(FN)
                
                # Handle bounding box
                for idx_FP, FP_box in enumerate(list_bbox_pred):
                    # print(FP_box)
                    if FP_box not in TP_box_list_pred:
                        bboxes_FP.append(FP_box)
                        
                
                for idx_FN, FN_clue in enumerate(list_clues_label):
                    if FN_clue == 0:
                        bboxes_FN.append(list_bbox_label[idx_FN])
                        
                        
                list_bbox_FP.append(bboxes_FP)
                list_bbox_FN.append(bboxes_FN)
                
                
                # print(list_bbox_label[-1])
                # print(len(list_bbox_label))

            Precision, Recall, F1_score = self.cal_metric(TP_all, FP_all, FN_all)
            # print("Precision: {}".format(Precision))
            # print("Recall: {}".format(Recall))
            # print("F1: {}".format(F1_score))
            all_P.append(Precision)
            all_R.append(Recall)
            all_F1.append(F1_score)

        PRECISION = round((sum(all_P))/len(all_P), 4)
        RECALL = round((sum(all_R))/len(all_R), 4)
        F1 = round((sum(all_F1))/len(all_F1), 4)
            
        # print("EVALUATING for bounding box----------------------------")
        # print("Precision: {}".format(round((sum(all_P))/len(all_P), 4)))
        # print("Recall: {}".format(round((sum(all_R))/len(all_R), 4)))
        # print("F1: {}".format(round((sum(all_F1))/len(all_F1), 4)))

        return PRECISION, RECALL, F1
