# SLRNet: Similarity-based Label Reuse for Semantic Segmentation

This is a PyTorch semantic segmentation technique for image datasets with evolving content such as images tracking human decomposition or growing plants. 

This repo is built based on [MIT repo from the CSAILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch). 

# Run

```
CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart_slrnet_cosin_conv_actV2.yaml
```
The resulted model would be saved in the 'DIR' specified in the config file. 

# Inference
More related information can be found at [here](https://github.com/zyang37/semantic-segmentation-pytorch). 
Sample python code to do inference:
```
from seg_utils import *
from PIL import Image

test_img = "/usb/sara_img/2f900410.14.JPG"
cfg_p = "config/bodypart_slrnet_cosin_conv_actV2.yaml"
predict_img = 'predict_2f900410.14.png'

model = load_segmentation_model(config_file=cfg_p)

pred_mask, img_original = predict_segmentation_mask(model, path=test_img, frame=None)

pred_color, org_pred_split = visualize_result(img_original.copy(), pred_mask)

img = Image.fromarray(org_pred_split)
img.save(predict_img)
```

All Aligned Predictions are located at:

```
/usb/body_alignment/
    - /img
    - /pred_anno
```
