from src import *

path_in = "SEGMENTATION/TOTAL"
path_out = "SEGMENTATION/K_folds"

# path_in = "DETECTION/TOTAL"
# path_out = "DETECTION/K_folds"

path_in_images = os.path.join(path_in, "images")
path_in_labels = os.path.join(path_in, "labels")

# split into 5 folds, if the total number of images is not divisible by 5, the last fold will have the remaining images
# each fold will have a subfolder named "fold_0", "fold_1", ..., "fold_4", then the subfolder will be "train", "test"
# In each folder "train" and "test", there are "images" and "labels" subfolders,
# the train subfolder will contain 80% of the images and labels, the test subfolder will contain 20% of the images and labels
# The images and labels will be copied from the original path to the new path, maintaining the same structure
# For example, if the original path is "SEGMENTATION/TOTAL/images",
# the new path will be "SEGMENTATION/K_folds/fold_0/images",
# "SEGMENTATION/K_folds/fold_0/labels", "SEGMENTATION
# the "images" subfolder will contain the images for that fold, the "labels" subfolder will contain the labels for that fold

make_folder(path_out)
for i in range(5):
    fold_path = os.path.join(path_out, f"fold_{i}")
    make_folder(fold_path)
    make_folder(os.path.join(fold_path,"train", "images"))
    make_folder(os.path.join(fold_path,"train", "labels"))
    make_folder(os.path.join(fold_path,"test", "images"))
    make_folder(os.path.join(fold_path,"test", "labels"))

item_list = list_item(path_in_images)
random.shuffle(item_list)
num_images = len(item_list)
fold_size = num_images // 5
remainder = num_images % 5

# Distribute images into folds
start_index = 0
for i in range(5):
    end_index = start_index + fold_size + (1 if i < remainder else 0)
    fold_images = item_list[start_index:end_index]
    
    # Copy images to the current fold's images folder
    for image in fold_images:
        src_image_path = os.path.join(path_in_images, image)
        dst_image_path = os.path.join(path_out, f"fold_{i}", "test/images", image)
        shutil.copy(src_image_path, dst_image_path)

    # Copy corresponding labels to the current fold's labels folder
    for label in fold_images:
        extension = os.path.splitext(label)[1]
        label_name = label.replace(extension, '.txt')  # Assuming labels are .txt files
        src_label_path = os.path.join(path_in_labels, label_name)
        dst_label_path = os.path.join(path_out, f"fold_{i}", "test/labels", label_name)
        shutil.copy(src_label_path, dst_label_path)

    start_index = end_index
    print(f"Fold {i} created with {len(fold_images)} images.")

    # Now copy the rest images (image that are not in this fold) to the train folder
    for image in item_list:
        if image not in fold_images:
            src_image_path = os.path.join(path_in_images, image)
            dst_image_path = os.path.join(path_out, f"fold_{i}", "train/images", image)
            shutil.copy(src_image_path, dst_image_path)

            label_name = os.path.splitext(image)[0] + '.txt'  # Assuming labels are .txt files
            src_label_path = os.path.join(path_in_labels, label_name)
            dst_label_path = os.path.join(path_out, f"fold_{i}", "train/labels", label_name)
            shutil.copy(src_label_path, dst_label_path)

