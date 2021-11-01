import pandas as pd
import numpy as np
import glob
import os
import cv2
import json
import sys


cluster_file = sys.argv[1] #"/data/sara/semantic-segmentation-pytorch/data/5random_donors_cleaned_clusters.uniq"
dest_filename = sys.argv[2] #"/data/sara/semantic-segmentation-pytorch/data/5random_donors_seqs_with_ann_pairs"
labeled_imgs = "/home/mousavi/exportsFromMongo/clusters/all_labeled_20210816"

def eval_purity(data):
    clusters= data['cluster_id'].unique()
    purities = []
    for cluster in clusters:
        labels = data[data['cluster_id'] == cluster]['label']
        labels_count = labels.value_counts()
        count_of_dominant_class = labels_count.max()
        purity = count_of_dominant_class/labels.size
        purities.append(purity *100)
    return purities



def label_pairs(filename, dest_filename):
    df_seqs = pd.read_csv(filename , sep=":", names=['img','cluster_id'])
    df_labels = pd.read_csv(labeled_imgs , sep=",", names=['img', 'label'])
    with open(dest_filename, "w") as fw:
        for row in df_seqs.iterrows():
                img_name = "/usb/sara_img/" + row[1].img.split('/')[-1].replace('.icon', '')
                label = df_labels[df_labels['img'].str.contains(row[1].img.split('/')[-1].replace('.icon',''))]
                if label.shape[0] > 0:
                    label = label.reset_index(drop=True)['label'][0].strip()
                    cluster_id = row[1].cluster_id.strip()
                    new_line = f"{img_name}, {label}, {cluster_id}\n"
                    fw.write(new_line)


label_pairs(cluster_file, dest_filename + "_labeled")

df = pd.read_csv(dest_filename + "_labeled", names=['img','label','cluster_id'])

print(f"INFO: number of sequences are {len(df['cluster_id'].unique())}")
purities = eval_purity(df)
print(f'average purity is: {np.mean(np.array(purities))}')
