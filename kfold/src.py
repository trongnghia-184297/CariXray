import re
import os
import cv2
import json
from tqdm import tqdm
import shutil
import random
import pandas as pd
import numpy as np

def atoi(text):
    return int(text) if text.isdigit() else text
 
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
   
def list_item(path):
    item_list = os.listdir(path)
    item_list.sort(key = natural_keys)
    return item_list

#######----Create folder----########
def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def yolo_to_xyxy(yolo_coors, image_width, image_height):
    # YOLO format (x, y, w, h)

    x1 = int((yolo_coors[0] - yolo_coors[2]/2) * image_width)
    y1 = int((yolo_coors[1] - yolo_coors[3]/2) * image_height)
    x2 = int((yolo_coors[0] + yolo_coors[2]/2) * image_width)
    y2 = int((yolo_coors[1] + yolo_coors[3]/2) * image_height)
    return [x1, y1, x2, y2]

def xyxy_to_yolo(box, width, height):
    dw = 1./width
    dh = 1./height
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]


def parse_yolo_labels(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # print("check-line")

    for line in lines:
        values = line.strip().split()
        class_label = values[0]  # The first value is the class label (optional)
        label, x_center, y_center, width, height = map(float, values)
        # x, y = x_center * image_width, y_center * image_height
        # w, h = width * image_width, height * image_height
        bounding_boxes.append([label, x_center, y_center, width, height])

    return bounding_boxes


def draw_list_rect(image, list_coors, list_LL, color, LL=True):
    thickness = -1
    radius = 5
    image_draw = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX 

    for idx, coor in enumerate(list_coors):
        image_draw = cv2.circle(image_draw, (coor[0], coor[1]), radius, color, thickness)
        image_draw = cv2.rectangle(image_draw, (coor[0], coor[1]), (coor[2], coor[3]), color, 2)
        coor_point_x = int((coor[0]+coor[2])/2)
        coor_point_y = int((coor[1]+coor[3])/2)
        # image_draw = cv2.putText(image_draw, str(idx), (coor_point_x, coor_point_y+8), font,  0.8, (0,0,255), 2, cv2.LINE_AA)
        image_draw = cv2.putText(image_draw, str(idx), (coor_point_x, coor_point_y+8), font,  0.8, (0,0,255), 2, cv2.LINE_AA)
        if LL:
            # text = "{}_{}_{}_{}".format(list_LL[idx][0],list_LL[idx][1],list_LL[idx][2],list_LL[idx][3])
            text = "{}_{}_{}_{}".format(list_LL[idx][0],list_LL[idx][2],list_LL[idx][4],list_LL[idx][6])
            image_draw = cv2.putText(image_draw, text, (coor[0], coor[1]), font,  0.8, (0,0,255), 2, cv2.LINE_AA)

    return image_draw



def draw_list_text(image, list_coors, list_LL, color):
    thickness = -1
    radius = 5
    image_draw = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX 

    for idx, coor in enumerate(list_coors):
        coor_point_x = int((coor[0]+coor[2])/2)
        coor_point_y = int((coor[1]+coor[3])/2)
        text = "{}_{}_{}_{}".format(list_LL[idx][0],list_LL[idx][1],list_LL[idx][2],list_LL[idx][3])
        image_draw = cv2.putText(image_draw, text, (coor_point_x-20, coor_point_y-20), font,  0.3, (0,0,255), 1, cv2.LINE_AA)

        image_draw = cv2.putText(image_draw, text, (coor[0], coor[1]), font,  0.4, (0,0,255), 1, cv2.LINE_AA)

    return image_draw


def calculate_area(pred, label):
    # print(pred)
    # update_pred = extend_bbox(pred, 10000, 10000, 0.04)
    # print(update_pred)
    # Extract coordinates from the bounding boxes
    x1, y1, x2, y2 = pred
    x3, y3, x4, y4 = label

    # Calculate the intersection area
    x_intersection = max(0, min(x2, x4) - max(x1, x3))
    y_intersection = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = x_intersection * y_intersection

    # Calculate the areas of each bounding box
    area_pred = (x2 - x1) * (y2 - y1)
    area_label = (x4 - x3) * (y4 - y3)

    # Calculate the Union area
    union_area = area_pred + area_label - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    # return round(iou,4), 
    return intersection_area, area_pred, area_label

def calculate_iou(pred, label):
    # print(pred)
    # update_pred = extend_bbox(pred, 10000, 10000, 0.04)
    # print(update_pred)
    # Extract coordinates from the bounding boxes
    x1, y1, x2, y2 = pred
    x3, y3, x4, y4 = label

    # Calculate the intersection area
    x_intersection = max(0, min(x2, x4) - max(x1, x3))
    y_intersection = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = x_intersection * y_intersection

    # Calculate the areas of each bounding box
    area_pred = (x2 - x1) * (y2 - y1)
    area_label = (x4 - x3) * (y4 - y3)

    # Calculate the Union area
    union_area = area_pred + area_label - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    # return round(iou,4), 
    return iou



def is_point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    
    # Check if the point's coordinates are within the bounding box range
    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    else:
        return False
    
def is_box_inside_bbox(small_box, big_box):
    tolerance = 0
    x1, y1, x2, y2 = small_box
    x3, y3, x4, y4 = big_box
    
    # Check if the small box's coordinates are within the big box range
    if x3-tolerance < x1 < x4+tolerance and y3-tolerance < y1 < y4+tolerance and x3-tolerance < x2+tolerance < x4 and y3-tolerance < y2 < y4+tolerance:
        return True
    else:
        return False
    
def compare_area_box(old_box, new_box):
    # Extract coordinates from the bounding boxes
    x1, y1, x2, y2 = old_box
    x3, y3, x4, y4 = new_box

    # Calculate the areas of each bounding box
    area_old_box = (x2 - x1) * (y2 - y1)
    area_new_box = (x4 - x3) * (y4 - y3)

    # Calculate the Union area
    if area_new_box > area_old_box:
        return True
    else:
        return False


            
def get_yolo_label(file_path):
    # Initialize an empty list to store appeared classes
    appeared_classes = []

    # Read the YOLO format file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract classes from each line and add to the list if not already present
    for line in lines:
        # Split the line to extract class (assuming class is the first item)
        class_name = line.strip().split(' ')[0]

        # Add the class to the list if not already present
        if int(class_name)+1 not in appeared_classes:
            appeared_classes.append(int(class_name)+1)

    sorted_appeared_classes = sorted(appeared_classes)

    return sorted_appeared_classes


def resize_2000(img_A):

    h_A, w_A = img_A.shape[:2]

    if h_A > 2000 or w_A > 2000:

        if h_A >= w_A:
            max_dim = h_A  
            new_h = 2000
            new_w = int((w_A / h_A) * 2000)
        else:
            max_dim = w_A  
            new_w = 2000
            new_h = int((h_A / w_A) * 2000)

        # Resize A to new dimension
        img_A_resized = cv2.resize(img_A, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:  
        img_A_resized = img_A

    return img_A_resized


def split_into_sublists(my_list, n):
    """
    Split a list into sublists of size n.

    Parameters:
    - my_list: The input list to be split.
    - n: The size of each sublist.

    Returns:
    A list of sublists.
    """
    return [my_list[i:i + n] for i in range(0, len(my_list), n)]


def check_condition_proportion(img, bbox):
    h,w,c = img.shape
    x_min, y_min, x_max, y_max = bbox
    bbox_h = y_max - y_min
    bbox_w = x_max - x_min

    if bbox_h > h*0.2 or bbox_w > w*0.2:
        return True
    else:
        return False
    

def filter_outer_boxes(boxes):
    """
    Loại bỏ các box nằm hoàn toàn trong box khác.
    
    Args:
        boxes (list of tuples): Danh sách bounding box dạng (x_min, y_min, x_max, y_max).
    
    Returns:
        list of tuples: Danh sách chỉ chứa các box ngoài cùng.
    """
    outer_boxes = []
    for i, box_a in enumerate(boxes):
        x_min_a, y_min_a, x_max_a, y_max_a = box_a
        is_inner = False
        for j, box_b in enumerate(boxes):
            if i == j:
                continue
            x_min_b, y_min_b, x_max_b, y_max_b = box_b
            # Kiểm tra box_a có nằm hoàn toàn trong box_b không
            if x_min_a >= x_min_b and y_min_a >= y_min_b and x_max_a <= x_max_b and y_max_a <= y_max_b:
                is_inner = True
                break
        if not is_inner:
            outer_boxes.append(box_a)
    return outer_boxes


def filter_duplicate_box(boxes, threshold=0.6):
    extracted_boxes = []
    for i, box_a in enumerate(boxes):
        x_min_a, y_min_a, x_max_a, y_max_a = box_a
        is_keep = True
        for j, box_b in enumerate(boxes):
            if i == j:
                continue
            x_min_b, y_min_b, x_max_b, y_max_b = box_b
            if calculate_iou(box_a, box_b) > threshold:
                # Calculate the areas of each bounding box
                area_box_a = (x_max_a - x_min_a) * (y_max_a - y_min_a)
                area_box_b = (x_max_b - x_min_b) * (y_max_b - y_min_b)
                if area_box_a < area_box_b:
                    is_keep = False
                    break
        if is_keep:
            extracted_boxes.append(box_a)
    return extracted_boxes
                