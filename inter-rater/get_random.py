from src import *

path_in = "/home/adminn/Documents/CARIES/DATA/DETECTION/K_folds/fold_3/test"
path_out = "random-10-percent"

path_in_images = os.path.join(path_in, "images")
path_in_labels = os.path.join(path_in, "labels")

# get random 500 images from path_in_images
list_images = list_item(path_in_images)
random.shuffle(list_images)
list_images = list_images[:500]


count = 0
for idx_img, image_name in enumerate(tqdm(list_images)):
    base_name = os.path.splitext(image_name)[0]
    label_name = base_name + ".txt"

    path_image_in = os.path.join(path_in_images, image_name)
    path_label_in = os.path.join(path_in_labels, label_name)

    if not os.path.exists(path_label_in):
        print("No label for image: {}".format(image_name))
        continue

    shutil.copy(path_image_in, path_out)
    shutil.copy(path_label_in, path_out)

    count += 1

print("Total images copied: {}".format(count))
print("Length of output folder: {}".format(len(os.listdir(path_out))))


