import re
import os
import cv2
import json
from tqdm import tqdm
import shutil
import random
import numpy as np
import pandas as pd

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


def get_images_folder(path):
    list_elements = list_item(path)
    list_imgs = [item for item in list_elements if os.path.splitext(item)[1] == ".jpg"]
    
    return list_imgs


def yolo_to_xyxy(yolo_coors, image_width, image_height):
    # YOLO format (x, y, w, h)

    x1 = int((yolo_coors[0] - yolo_coors[2]/2) * image_width)
    y1 = int((yolo_coors[1] - yolo_coors[3]/2) * image_height)
    x2 = int((yolo_coors[0] + yolo_coors[2]/2) * image_width)
    y2 = int((yolo_coors[1] + yolo_coors[3]/2) * image_height)
    return [x1, y1, x2, y2]


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
        bounding_boxes.append([x_center, y_center, width, height])

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
        # image_draw = cv2.putText(image_draw, str(idx), (coor_point_x, coor_point_y+8), font,  0.6, (0,0,255), 2, cv2.LINE_AA)
        # image_draw = cv2.putText(image_draw, str(idx), (coor_point_x, coor_point_y+8), font,  1, (0,0,255), 2, cv2.LINE_AA)
        if LL:
            text = "{}_{}_{}_{}".format(list_LL[idx][0],list_LL[idx][1],list_LL[idx][2],list_LL[idx][3])
            # image_draw = cv2.putText(image_draw, text, (coor[0], coor[1]), font,  0.4, (0,0,255), 1, cv2.LINE_AA)
            image_draw = cv2.putText(image_draw, text, (coor[0], coor[1]), font, 1, (0,0,255), 2, cv2.LINE_AA)

    return image_draw


def draw_list_IoU(image, list_coors, list_IoU, color):
    thickness = -1
    radius = 5
    image_draw = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX 

    for idx, coor in enumerate(list_coors):
        # image_draw = cv2.circle(image_draw, (coor[0], coor[1]), radius, color, thickness)
        # image_draw = cv2.rectangle(image_draw, (coor[0], coor[1]), (coor[2], coor[3]), color, 2)
        coor_point_x = int((coor[0]+coor[2])/2)
        coor_point_y = int((coor[1]+coor[3])/2)
        # image_draw = cv2.putText(image_draw, str(idx), (coor_point_x, coor_point_y+8), font,  0.6, (0,0,255), 2, cv2.LINE_AA)
        # image_draw = cv2.putText(image_draw, str(idx), (coor_point_x, coor_point_y+8), font,  1, (0,0,255), 2, cv2.LINE_AA)
        image_draw = cv2.putText(image_draw, str(list_IoU[idx]), (coor_point_x, coor_point_y+8), font,  1, color, 2, cv2.LINE_AA)

    return image_draw


def extract_IoU(list_bbox_labels, list_bbox_preds):
    list_IoU_labels = []
    
    for idx_box_label, bb_label in enumerate(list_bbox_labels):
        IoU = 0
        for idx_bb_pred, bb_pred in enumerate(list_bbox_preds):
            IoU_cal = calculate_IOU(bb_label, bb_pred)
            if IoU_cal > IoU:
                IoU = IoU_cal

        list_IoU_labels.append(IoU)            
    
    return list_IoU_labels
    
    


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


def calculate_IOU(pred, label):
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
    return round(iou, 2)


def polygon_iou(points_pred, points_gt):
    """
    Compute IoU between two polygons given as:
      - [[x,y], ...]  (một đa giác)
      - hoặc [[[x,y], ...], [[x,y], ...], ...] (nhiều phần)
    Trả về IoU float trong [0,1].
    """
    # ---- Try exact geometry with shapely ----
    try:
        from shapely.geometry import Polygon, MultiPolygon
        def to_shape(pts):
            if not pts:
                return None
            if isinstance(pts[0][0], (int, float)):  # single polygon
                poly = Polygon(pts)
            else:  # multi-polygon
                poly = MultiPolygon([Polygon(p) for p in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)  # fix self-intersections
            return poly

        A = to_shape(points_pred)
        B = to_shape(points_gt)
        if A is None or B is None:
            return 0.0
        inter = A.intersection(B).area
        union = A.union(B).area
        return float(inter / union) if union > 0 else 0.0

    except Exception:
        # ---- Fallback: rasterize with OpenCV ----
        import numpy as np, cv2, math

        def to_arrays(pts):
            if not pts:
                return []
            if isinstance(pts[0][0], (int, float)):
                pts = [pts]
            return [np.asarray(p, dtype=np.float32) for p in pts]

        A = to_arrays(points_pred)
        B = to_arrays(points_gt)
        if not A or not B:
            return 0.0

        max_x = max(float(a[:, 0].max()) for a in A + B)
        max_y = max(float(a[:, 1].max()) for a in A + B)
        w = int(math.ceil(max_x)) + 2
        h = int(math.ceil(max_y)) + 2

        maskA = np.zeros((h, w), np.uint8)
        maskB = np.zeros((h, w), np.uint8)
        for a in A:
            cv2.fillPoly(maskA, [a.astype(np.int32)], 1)
        for b in B:
            cv2.fillPoly(maskB, [b.astype(np.int32)], 1)

        inter = np.logical_and(maskA, maskB).sum().astype(float)
        union = np.logical_or(maskA, maskB).sum().astype(float)
        return float(inter / union) if union > 0 else 0.0



def is_point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    
    # Check if the point's coordinates are within the bounding box range
    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    else:
        return False
    


def concatenate_img(list_imgs):
    list_img_padding = []
    for img in list_imgs[:-1]:
        list_img_padding.append(img)
        h = img.shape[0]
        blank_image = np.zeros((h,10,3), np.uint8)
        # list_img_padding.append(np.ones_like(img)*255)
        list_img_padding.append(blank_image)
    list_img_padding.append(list_imgs[-1])
    imgs = np.hstack(list_img_padding)
    
    return imgs


def draw_text(img, text):
    h, w, c = img.shape
    
    thickness = -1
    radius = 5
    image_draw = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    image_draw = cv2.putText(image_draw, text, (w-350, h-20), font,  1.6, (0,0,255), 2, cv2.LINE_AA)
    
    return image_draw



def extract_type(path, postfix):
    list_elements = list_item(path)
    list_extract = [item for item in list_elements if os.path.splitext(item)[1] == postfix]
    return list_extract


def get_bbox_csv(csv_path):
    df = pd.read_csv(csv_path)
    list_bbox = df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
    # print(list_bbox)
    
    return list_bbox

def get_bbox_json_pred(json_pred_path):
    json_data = json.load(open(json_pred_path))
    physical_box = [item["phys"] for item in json_data]
    list_bbox = [[item[0][0], item[0][1], item[2][0], item[2][1]] for item in physical_box]
    # pass
    return list_bbox



def create_clue(length):
    list_zero = [0 for _ in range(length)]
    return list_zero
    
    

def cal_metric(TP, FP, FN):
    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision,5), round(recall,5), round(f1_score,5)


def imwrite2(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    

def visual_for_check_usingOCR(img_not_vertical, img_use_vertical, list_bbox_ocr, path_save):
    h_not_vertical, w_not_vertical = img_not_vertical.shape[:2]
    part_width = w_not_vertical // 3

    img_label = img_not_vertical[0:h_not_vertical, 0:1 * part_width]
    img_label = draw_list_rect(img_label, list_bbox_ocr, [], [0,0,255], False)

    img_not_vertical_extract = img_not_vertical[0:h_not_vertical, 2 * part_width:w_not_vertical]
    img_use_vertical_extract = img_use_vertical[0:h_not_vertical, 2 * part_width:w_not_vertical]
    combine_img = np.hstack([img_label, img_not_vertical_extract, img_use_vertical_extract])
    cv2.imwrite(path_save, combine_img)


def convert_segCOCO_to_point(list_segCOCO):
    list_point = []
    for i in range(0, len(list_segCOCO), 2):
        x = list_segCOCO[i]
        y = list_segCOCO[i + 1]
        list_point.append([x, y])
    return list_point