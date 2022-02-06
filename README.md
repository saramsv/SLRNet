# SLRNet: Similarity-based Label Reuse for Semantic Segmentation

This is a PyTorch semantic segmentation technique for image datasets with evolving content such as images tracking human decomposition or growing plants. 


# Inference


```
from seg_utils import *

test_img = "/usb/sara_img/2f900410.14.JPG"
cfg_p = "../data/bodypart_slrnet_cosin_conv_actV2.yaml"

model = load_segmentation_model(config_file=cfg_p)

pred_mask, img_original = predict_segmentation_mask(model, path=test_img, frame=None)

pred_color, org_pred_split = visualize_result(img_original.copy(), pred_mask)
```