import os
import cv2
import sys
import yaml
import time
import math
import random
import skimage
import argparse
import numpy as np
import matplotlib.pyplot as plt

##################### seg model stuff #####################
# System libs
sys.path.insert(1, '/data/sara/SLRNet/')
import csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms

# Our libs
from semseg.models import ModelBuilder, SegmentationModule
from semseg.utils import colorEncode

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Global variables
alpha = 0.5
names = {0:'bg', 1:'foot', 2:'hand', 3:'arm', 4:'leg', 5:'torso', 6:'head'}
colors = [(197, 215, 20), (132, 248, 207), (155, 244, 183), (111, 71, 144), (71, 48, 128), (75, 158, 50), (37, 169, 241)]
colors = np.array(colors, dtype='uint8')

# pass in mode config(yaml file)
# return a dict for the file 
# return decoder and encoder weights path
def parse_model_config(path):
    with open(path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    encoder_path = None
    decoder_path = None
    for p in os.listdir(data['DIR']):
        if "encoder" in p.lower():
            encoder_path = "{}/{}".format(data['DIR'], p)
            continue
        if "decoder" in p.lower():
            decoder_path = "{}/{}".format(data['DIR'], p)
            continue
    if encoder_path==None or decoder_path==None:
        raise("model weights not found")
    return data, encoder_path, decoder_path


def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        #print(f'{names[index+1]}:')
        print(f'{names[index]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    #if show==True:
        #display(PIL.Image.fromarray(im_vis))
    #else:
    return pred_color, im_vis


def process_img(path=None, frame=None):
    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    # pil_image = PIL.Image.open('../ADE_val_00001519.jpg').convert('RGB')
    if path!=None:
        pil_image = PIL.Image.open(path).convert('RGB')
    else:
        pil_image = PIL.Image.fromarray(frame)

    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]
    return (img_original, singleton_batch, output_size)

def predict_img(segmentation_module, singleton_batch, output_size):
    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred


def get_color_palette(pred, bar_height):

    pred = np.int32(pred)
    pixs = pred.size

    top_left_y = 0
    bottom_right_y = 30
    uniques, counts = np.unique(pred, return_counts=True)

    # Create a black image
    # bar_height = im_vis.shape[0]
    img = np.zeros((bar_height,250,3), np.uint8)

    for idx in np.argsort(counts)[::-1]:
        color_index = uniques[idx]
        name = names[color_index ] # WAS + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            #WAS print("{}  {}: {:.2f}% {}".format(color_index+1, name, ratio, colors[color_index]))
            print("{}  {}: {:.2f}% {}".format(color_index, name, ratio, colors[color_index]))
            img = cv2.rectangle(img, (0,top_left_y), (250,bottom_right_y), 
                       (int(colors[color_index][0]),int(colors[color_index][1]),int(colors[color_index][2])), -1)
            img = cv2.putText(img, "{}: {:.3f}%".format(name, ratio), (0,top_left_y+20), 5, 1, (255,255,255), 2, cv2.LINE_AA)
            top_left_y+=30
            bottom_right_y+=30
            
    return img


def transparent_overlays(image, annotation, alpha=0.5):
    img1 = image.copy()
    img2 = annotation.copy()

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    # img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    # dst = cv2.add(img1_bg, img2_fg)
    dst = cv2.addWeighted(image.copy(), 1-alpha, img2_fg, alpha, 0)
    img1[0:rows, 0:cols ] = dst
    return dst


def load_segmentation_model(config_file="../data/bodypart_slrnet_cosin_conv_actV2.yaml"):
    '''
    config_file = "../data/bodypart_slrnet_cosin_conv_actV2.yaml"
    return a trained seg model 
    '''
    model_config, encoder_path, decoder_path = parse_model_config(config_file)
    net_encoder = ModelBuilder.build_encoder(
        arch = model_config["MODEL"]['arch_encoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        weights = encoder_path)
    net_decoder = ModelBuilder.build_decoder(
        arch = model_config["MODEL"]['arch_decoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        num_class = model_config['DATASET']['num_class'],
        weights = decoder_path,
        use_softmax=True)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    with open(config_file) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, configs['TRAIN']['batch_size_per_gpu'])
    segmentation_module.eval();
    segmentation_module.cuda();
    return segmentation_module
    
    
def predict_segmentation_mask(segmentation_module, path=None, frame=None):
    '''
    use the seg model, predict and return a grey scale mask
    '''
    img_original, singleton_batch, output_size = process_img(path=path, frame=frame)
    pred = predict_img(segmentation_module, singleton_batch, output_size)
    # pred_color, org_pred_split = visualize_result(img_original, pred)
    return pred, img_original


'''
# MAIN
if __name__ == "__main__":
    test_img = "/usb/sara_img/2f900410.14.JPG"
    
    cfg_p = "../data/bodypart_slrnet_cosin_conv_actV2.yaml"
    print("loading: {}".format(cfg_p))
        
    model = load_segmentation_model(config_file=cfg_p)
    pred_mask, img_original = predict_segmentation_mask(model, path=test_img, frame=None)
    pred_color, org_pred_split = visualize_result(img_original.copy(), pred_mask)
    
    # color_palette
    color_palette = get_color_palette(pred_mask, org_pred_split.shape[0])
    dst = transparent_overlays(img_original, pred_color, alpha=alpha)
    pred_color_palette_dst = numpy.concatenate((dst, color_palette), axis=1)
    pred_color_palette_all = numpy.concatenate((org_pred_split, pred_color_palette_dst), axis=1)
    
    cv2.imwrite("../data/seg_utils_test.png", cv2.cvtColor(pred_color_palette_all, cv2.COLOR_RGB2BGR))
    print("saved: ../data/seg_utils_test.png")
'''