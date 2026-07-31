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

print("before transformer import")
import transformers
from transformers import ViTModel
print("after transformer import", transformers.__version__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class VIT_TL(nn.Module):
    def __init__(self, do=0.0, num_class=133, num_cls_blocks=1):
        super().__init__()
        self.basemodel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.h_dim = 512
        self.flatten_layer = nn.Flatten()
        self.classifier_1 = self.conv_block(197*768, self.h_dim, do)
        self.class_init_layer = nn.Sequential(
            self.flatten_layer,
            self.classifier_1
        )
        if num_cls_blocks == 1:
            self.classifier = nn.Sequential(
                self.class_init_layer,
                nn.Linear(self.h_dim, num_class)
                )
        else:
            self.classifier_2 = self.conv_block(self.h_dim, int(self.h_dim/2), do)
            self.classifier = nn.Sequential(
                self.class_init_layer,
                self.classifier_2,
                nn.Linear(int(self.h_dim/2), num_class)
            )
     
    def conv_block(self, input_dim, output_dim, do):
        class_block = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False),
                                    nn.BatchNorm1d(output_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(do)
                                    )
        return class_block
    
    def forward(self, x):
        x = self.basemodel(x)
        last_hidden_states = x.last_hidden_state
        x = self.classifier(last_hidden_states)
        return x

def model_fn(model_dir):
    logger.info("Starting model_fn script")
    model = VIT_TL()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info("Model weight being loaded")
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        logger.info("check check")
        checkpoint = torch.load(f, map_location=torch.device(device))
        logger.info(f'checkpoint keys: {checkpoint.keys()}')
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.load_state_dict(torch.load(f))

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