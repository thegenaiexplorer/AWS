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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net(arch, do, num_class, pre_weight_freeze, num_cls_blocks):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''

    def conv_block(input_dim, output_dim, do):
        class_block = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False),
                                    nn.BatchNorm1d(output_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(do)
                                    )
        return class_block

    model_cnn = ["resnet50", "efficientnet_b0"]
    if arch in model_cnn:
        if arch == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            init_dim = 2048
        else:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights)
            init_dim = 1280

        h_dim = 512

        if pre_weight_freeze == "True":
            for params in model.parameters():
                params.requires_grad = False
        classifier_1 = conv_block(init_dim, h_dim, do)
        if num_cls_blocks == 1:
            classifier = nn.Sequential(
                classifier_1,
                nn.Linear(h_dim, num_class)
                )
        else:
            classifier_2 = conv_block(h_dim, int(h_dim/2), do)
            classifier = nn.Sequential(
                classifier_1,
                classifier_2,
                nn.Linear(int(h_dim/2), num_class)
                )
        if arch == "resnet50":
            model.fc = classifier
        else:
            model.classifier = classifier

    elif arch == "ViT":
        model = VIT_TL(do, num_class, pre_weight_freeze, num_cls_blocks)
    else:
        logger.info("Please supply either ViT, efficientnet_b0 or resnet50 with arch hyperparameter")
        return None
    logger.info(f' Model return config is {model}')
    return model

def model_fn(model_dir):
    logger.info("Starting model_fn script")
    model = net("resnet50", 0.2, 133, "False", 1)

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