import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import io
import requests
#print("before transformer import")
#import transformers
#from transformers import ViTModel
#print("after transformer import", transformers.__version__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def model_fn(model_dir):
    logger.info("Starting model_fn script")
    model = models.efficientnet_b0(pretrained=False)
    for params in model.parameters():
        params.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(1280, 512, bias=False),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 128, bias=False),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.2),
                                     nn.Linear(128, 5)
                                     )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info("Model weight being loaded")
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.eval()
    logger.info("Model weight load complete")
    logger.info("Finished model_fn script")
    return model

def input_fn(request_body, content_type):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    infer_transforms = T.Compose([
        T.Resize((256)),
        T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    logger.info("transforms block complete")
    input_object = infer_transforms(input_object)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_object = input_object.to(device)
    logger.info("transforms image complete and image moved to GPU")
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
        logger.info("prediction block complete")
    return prediction