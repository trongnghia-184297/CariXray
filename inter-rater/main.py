from src import *
from cal_metric import CAL_METRIC
from typing import List, Sequence, Optional, Union
import argparse
import logging

def bootstrap_sample(
    items: Sequence, 
    n: int = 500, 
    seed: Optional[Union[int, random.Random]] = None
) -> List:
    """
    Trả về 1 list dài n, được resample *có hoàn lại* từ `items`.
    - items: list/sequence đầu vào (không rỗng)
    - n: kích thước mẫu (mặc định 500)
    - seed: số nguyên hoặc đối tượng random.Random để tái lập kết quả (tùy chọn)
    """
    if not items:
        raise ValueError("`items` không được rỗng.")
    
    rng = seed if isinstance(seed, random.Random) else random.Random(seed) if seed is not None else random
    return rng.choices(items, k=n)


parser = argparse.ArgumentParser()
parser.add_argument('--path_image', dest='path_image',
                    type=str, default="", help='')
parser.add_argument('--path_label', dest='path_label',
                    type=str, default="", help='')
parser.add_argument('--path_pred', dest='path_pred',
                    type=str, default="", help='')
parser.add_argument('--path_log', dest='path_log',
                    type=str, default="", help='')


args = parser.parse_args()
path_label = args.path_label
path_pred = args.path_pred
path_image = args.path_image
path_log = args.path_log

list_elements = list_item(path_image)
make_folder(os.path.dirname(path_log))
list_images = [item for item in list_elements if os.path.splitext(item)[1] != ".txt"]
# Inter-rater
list_thresh = [0.25, 0.5, 0.75]
num_times = 2000


# open log file
if path_log != "":
    import logging
    logging.basicConfig(filename=path_log, level=logging.INFO)
    logging.info("Path label: {}".format(path_label))
    logging.info("Path pred: {}".format(path_pred))
    logging.info("Path image: {}".format(path_image))
    logging.info("Number of images: {}".format(len(list_images)))

# Main
# print("simple eval")
# cal = CAL_METRIC(path_label, path_pred, path_image, list_images, 0.75)
# PRECISION, RECALL, F1 = cal.process()
# print("Precision: {}".format(PRECISION))
# print("Recall: {}".format(RECALL))
# print("F1: {}".format(F1))


for thresh in tqdm(list_thresh):
    print("-"*50)
    logging.info("-"*50)

    print(f"IoU threshold: {thresh}")
    logging.info(f"IoU threshold: {thresh}")

    print("----RUNNING AVERAGE METRIC----")
    logging.info("----RUNNING AVERAGE METRIC----")

    cal = CAL_METRIC(path_label, path_pred, path_image, list_images, thresh)
    PRECISION_AVG, RECALL_AVG, F1_AVG = cal.process()
    print("----RUNNING BOOTSTRAP METRIC----")
    logging.info("----RUNNING BOOTSTRAP METRIC----")
    all_P = []
    all_R = []
    all_F1 = []
    for i in range(num_times):
        # cal = CAL_METRIC(path_label, path_pred, path_image, list_images[:100], 0.5)
        cal = CAL_METRIC(path_label, path_pred, path_image, bootstrap_sample(list_images, n=len(list_images)), thresh)
        PRECISION, RECALL, F1 = cal.process()
        all_P.append(PRECISION)
        all_R.append(RECALL)
        all_F1.append(F1)

    # sorting from low to high
    all_P = sorted(all_P)
    all_R = sorted(all_R)
    all_F1 = sorted(all_F1)

    # get 2.5th percentile and 97.5th percentile
    p_lower = all_P[int(0.025 * len(all_P))]
    p_upper = all_P[int(0.975 * len(all_P))]
    r_lower = all_R[int(0.025 * len(all_R))]
    r_upper = all_R[int(0.975 * len(all_R))]
    f1_lower = all_F1[int(0.025 * len(all_F1))]
    f1_upper = all_F1[int(0.975 * len(all_F1))]

    # print with format "F1@IoU=0.5 = 0.76 (95% CI: 0.73--0.79)"
    print(f"Precision@IoU={thresh} = {PRECISION_AVG} (95% CI: {p_lower}--{p_upper})")
    logging.info(f"Precision@IoU={thresh} = {PRECISION_AVG} (95% CI: {p_lower}--{p_upper})")

    print(f"Recall@IoU={thresh} = {RECALL_AVG} (95% CI: {r_lower}--{r_upper})")
    logging.info(f"Recall@IoU={thresh} = {RECALL_AVG} (95% CI: {r_lower}--{r_upper})")

    print(f"F1@IoU={thresh} = {F1_AVG} (95% CI: {f1_lower}--{f1_upper})")
    logging.info(f"F1@IoU={thresh} = {F1_AVG} (95% CI: {f1_lower}--{f1_upper})")

    print("-"*50)
    logging.info("-"*50)
