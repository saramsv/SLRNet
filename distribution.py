import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re


def get_paths(filename):
    data = pd.read_csv(filename, names=['img', 'segm', 'w', 'h'])
    segms = data.segm.values
    pattern = "new(.*?)png"
    return ["new"+re.search(pattern, i).group(1)+"png" for i in segms]


# get_pixel_counts for a dir of anno
def get_pixel_counts(annotations_list):
    pixel_dis = {}
    for i in tqdm(annotations_list):
        # print(i)
        # read img in grayscale
        img = cv2.imread(i, 0)
        tmp_count = np.unique(img, return_counts=True)
        # update dict
        for i in range(len(tmp_count[0])):
            # print(tmp_count[0][i], tmp_count[1][i])
            try:
                pixel_dis[tmp_count[0][i]]+=tmp_count[1][i]
            except:
                pixel_dis[tmp_count[0][i]]=tmp_count[1][i]

    sorted_items = sorted(pixel_dis.items())
    return dict(sorted_items)

# pixel 0: bg
# bg=False: to excluded bg when calculating distribution 
def print_distribution(pixel_dis, bg=True):
    if bg:
        total_p = sum(pixel_dis.values())
    else:
        total_p = sum(pixel_dis.values()) - pixel_dis[0]
    dists = []
    #print("class:  count:  distribution")
    for k, v in pixel_dis.items():
        if (not bg) and (k==0):
            continue
        dis = v / total_p
        dists.append(dis)
        #print("{}: {}: {:.3f}".format(k, v, dis))
    print(dists)
    return dists

def main(train_annotations_colored_list, bg=True):
    total_img_list = train_annotations_colored_list[:]
    train_pixel_dis = get_pixel_counts(train_annotations_colored_list)
    return print_distribution(train_pixel_dis, bg=bg)

if __name__=="__main__":
    train_anno_dir = "/data/sara/semantic-segmentation-pytorch/new_train_noBack_annotations/"
    val_anno_dir = "/data/sara/semantic-segmentation-pytorch/new_val_noBack_annotations/"
    train_annotations_colored_list = get_paths('data/mit_seq_2')#["{}{}".format(train_anno_dir, val) for val in os.listdir(train_anno_dir)]
    train_annotations_colored_list.extend(get_paths('data/decom_seq_data_new_label_noBack.odgt'))
    main(train_annotations_colored_list, False)
