from tqdm.auto import tqdm
import torch, torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor
import numpy as np, skimage.io as skio


def safe_imread(path):
    """skimage + imagecodecs fallback → Pillow."""
    try:
        return skio.imread(path)
    except ValueError as e:
        if "imagecodecs" in str(e):
            from PIL import Image
            return np.array(Image.open(path))
        raise

     

def get_instance_seg_model(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights="COCO_V1")

    # replace heads
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_feat_mask, 256, num_classes)

    # tiny anchors
    sizes  = ((8,), (16,), (32,), (64,), (128,))
    ratios = ((0.5,1.0,2.0),)*5
    model.rpn.anchor_generator = AnchorGenerator(sizes, ratios)

    # more proposals
    model.rpn.pre_nms_top_n_train  = 4000
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.pre_nms_top_n_test   = 2000
    model.rpn.post_nms_top_n_test  = 1000
    model.roi_heads.detections_per_img = 1000
    model.roi_heads.score_thresh = 0.20

    # larger mask ROI
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0','1','2','3'],
            output_size=14, sampling_ratio=2)
    return model