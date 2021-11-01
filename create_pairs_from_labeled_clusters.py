import pandas as pd
import numpy as np
import glob
import os
import cv2
import json
import sys


cluster_file = sys.argv[1] #"/data/sara/semantic-segmentation-pytorch/data/5random_donors_cleaned_clusters.uniq"
dest_filename = sys.argv[2] #"/data/sara/semantic-segmentation-pytorch/data/5random_donors_seqs_with_ann_pairs"
all_img_dir = "/usb/sara_img/"
labels_dir = "/data/sara/semantic-segmentation-pytorch/all_body_part_annotations/"
extension = ".png"
labeled_imgs = "/home/mousavi/exportsFromMongo/clusters/all_labeled_20210816"

def label_pairs(filename, all_img_dir, labels_dir, dest_filename, extension):
    df_seqs = pd.read_csv(filename , sep=":", names=['img','cluster_id'])
    df_labels = pd.read_csv(labeled_imgs , sep=",", names=['img', 'label'])
    h, w = 400, 600
    with open(dest_filename + ".odgt", "w") as fw:

        for img in glob.glob(labels_dir + '/*.png'):
            img_name = img.split("/")[-1].replace(extension, '')
            img_path = os.path.join(all_img_dir, img_name + ".JPG")

            matches = df_seqs[df_seqs['img'].str.contains(img_name)]
            if matches.shape[0] > 0:
                cluster_name = matches.reset_index(drop=True)['cluster_id'][0]
                seq_match = df_seqs[df_seqs['cluster_id'] == cluster_name].reset_index(drop=True)

                #img_path = "/usb/sara_img/" + img_path
                if seq_match.shape[0] >= 2:
                    img_names = [img_path]
                    label1 = "unlabeled"
                    labels = [label1]

                    label = df_labels[df_labels['img'].str.contains(img_name)]
                    if label.shape[0] > 0:
                        label = label.reset_index(drop=True)['label'][0]
                        labels = [label]
                    for row in seq_match.iterrows():
                        if row[1].img not in img_names:
                            img_names.append("/usb/sara_img/" + row[1].img.split('/')[-1].replace('.icon', ''))
                            label2 = df_labels[df_labels['img'].str.contains(row[1].img.split('/')[-1])]
                            if label2.shape[0]>0:
                                labels.append(label2.reset_index(drop=True)['label'][0])
                            else:
                                labels.append("unlabeled")
                        if len(img_names) == 2:
                            new_line = {}
                            new_line["fpath_img"], new_line["fpath_segm"], new_line["width"], new_line["height"], new_line["same_class"] = \
                                    img_names, labels, w, h, labels[0] == labels[1]
                            json.dump(new_line, fw)
                            fw.write('\n')
                            img_names = [img_path]
                            labels = [label1]


label_pairs(cluster_file, all_img_dir ,labels_dir, dest_filename + "_labeled", extension)

df = pd.read_json(dest_filename + "_labeled.odgt", lines=True)
print(df['same_class'].value_counts())
good_pairs_percentage =  df[df['same_class'] == True].shape[0] /df.shape[0]
print(good_pairs_percentage)

