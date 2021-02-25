import json
import pandas as pd
import cv2
import argparse

train_file = '/data/sara/TCT/CCT/body_part_sequencing/data/sup_train.txt'
val_file = '/data/sara/TCT/CCT/body_part_sequencing/data/val.txt'
seq_file = '/data/sara/semantic-segmentation-pytorch/data/seq_data.txt' #each 3 lines belon to the same sequence 

def get_data(filename, mode):
    with open(filename, 'r') as fr , open("decom_" + mode + '.odgt','a') as fw :
        lines = fr.readlines()
        for line in lines:
            img_name, seg_name =  line.split(' ')
            seg = cv2.imread(seg_name.strip(), 0)
            h, w = seg.shape
            new_line = {}
            new_line["fpath_img"], new_line["fpath_segm"], new_line["width"], new_line["height"] = \
                    img_name.strip(), seg_name.strip(), w, h
            json.dump(new_line, fw)
            fw.write('\n')

def get_seq_data(filename, seq_len):
    seq_df = pd.read_csv(filename, sep=" ", names=['img','label'])
    labels = seq_df['label'].unique()
    with open("decom_" + filename.split('/')[-1].split('.')[0] + '.odgt','a') as fw :
        for label in labels:
            matches = seq_df[seq_df['label'] == label]
            img_names = []
            seg_names = []
            i = 0
            main_img = ''
            main_segm = ''
            for row in matches.iterrows():
                img_name, seg_name =  row[1]['img'], row[1]['label']
                if i == 0:
                    main_img = img_name
                    main_segm = seg_name
                    img_names = [main_img]
                    seg_names = [main_segm]
                else:
                    img_names.append(img_name)
                    seg_names.append(seg_name)
                i += 1
                if len(img_names) % seq_len == 0:
                    seg = cv2.imread(seg_name.strip(), 0)
                    h, w = seg.shape
                    new_line = {}
                    new_line["fpath_img"], new_line["fpath_segm"], new_line["width"], new_line["height"] = \
                            img_names, seg_names, w, h
                    json.dump(new_line, fw)
                    fw.write('\n')
                    img_names = [main_img]
                    seg_names = [main_segm]


            if len(img_names) == seq_len:
                seg = cv2.imread(seg_name.strip(), 0)
                h, w = seg.shape
                new_line = {}
                new_line["fpath_img"], new_line["fpath_segm"], new_line["width"], new_line["height"] = \
                        img_names, seg_names, w, h
                json.dump(new_line, fw)
                fw.write('\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default=True, type=bool)
    args = parser.parse_args()
    
    if args.seq != True:
        get_data(train_file, 'train')
        get_data(val_file, 'val')
    else:
        get_seq_data(seq_file, 3)


