#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

print("before transformer import")
import transformers
from transformers import ViTModel
print("after transformer import", transformers.__version__)

import numpy as np
import os
import argparse
import sys
import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


#TODO: Import dependencies for Debugging andd Profiling

def parse_inputs():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Job")
    parser.add_argument('--bs',
                        type=int,
                        default=32,
                        metavar="N",
                        help="Batch Size for training (default:32)"
                       )
    parser.add_argument('--lrate',
                        type=float,
                        default=0.0002,
                        metavar="LR",
                        help="Learning Rate for training (default:0.0002)"
                       )
    parser.add_argument('--num_epochs',
                        type=int,
                        default=3,
                        metavar="N",
                        help="Number of epochs to train (default:3)"
                       )
    parser.add_argument('--hook_true',
                        type=str,
                        default="False",
                        help="Debugger/Profiler Status (default:False(Disabled))"
                       )
    parser.add_argument('--arch',
                        type=str,
                        default="resnet50",
                        help="Architecture to use for model training. Default: resnet50",
                        )
    args = parser.parse_args()
    log_text = 'Starting Hyperparameter tuning Job with '
    logger.info(f'{log_text} Learning Rate: {args.lrate}')
    logger.info(f'Batch Size: {args.bs}, Epochs: {args.num_epochs}, Arch: {args.arch}')
    return args


def create_data_loaders(data_path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dir = data_path + 'train'
    valid_dir = data_path + 'validation'
    test_dir = data_path + 'test'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = T.Compose([
        T.Resize(256),
        T.RandomAffine(scale=(0.9, 1.1), translate=(0.1, 0.1), degrees=10),
        T.RandomHorizontalFlip(0.5),
        T.RandomResizedCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    valid_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    cpu_cores = os.cpu_count()

    train_ds = ImageFolder(root=train_dir, transform=train_transforms)
    valid_ds = ImageFolder(root=valid_dir, transform=valid_transforms)
    test_ds = ImageFolder(root=test_dir, transform=valid_transforms)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=cpu_cores)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=batch_size, num_workers=cpu_cores)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=cpu_cores)

    return train_dl, valid_dl, test_dl

class VIT_TL(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for params in self.basemodel.parameters():
            params.requires_grad = False
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(197*768, 512, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 128, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(128, 5)
                                        )

    def forward(self,x):
        x = self.basemodel(x)
        last_hidden_states = x.last_hidden_state
        x = self.classifier(last_hidden_states)
        return x


def net(arch):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    if arch=="resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for params in model.parameters():
            params.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(2048, 512, bias=False),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 5)
                                 )
    elif arch=="efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights)
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
      
    elif arch=="ViT":
        model = VIT_TL()
    else:
        logger.info("Please supply either ViT, efficientnet_b0 or resnet50 with arch hyperparameter")
        return None
    
        
    logger.info(f' Model return config is {model}')

    return model

def train_model(epochs, model, train_dl, loss_criterion, optimizer, device, valid_dl, hook, hook_enabled):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    len_ds = len(train_dl.dataset)
    for e in range(epochs):
        model.train()
        if (hook_enabled == "True"):
            hook.set_mode(smd.modes.TRAIN) #Debugger config
        running_loss = 0
        accuracy = 0
        for imgs, labs in train_dl:
            bs = imgs.shape[0]
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            loss = loss_criterion(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bs
            accuracy += (torch.argmax(out, dim=1) == labs).sum().item()
        logger.info(f'Epoch: {e}')
        logger.info(f'Training Loss: {round(running_loss/len_ds, 5)}')
        logger.info(f'Training Accuracy: {round(accuracy/len_ds,5)}')
        test = False
        test_model(model, valid_dl, loss_criterion, device, test, hook, hook_enabled)
    return model

def test_model(model, test_dl, loss_criterion, device, test, hook, hook_enabled):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    test_accuracy = 0
    test_loss = 0
    ds_class = "Valid"
    if (test):
        ds_class = "Test"
    len_ds = len(test_dl.dataset)

    model.eval()
    if (hook_enabled == "True"):
        hook.set_mode(smd.modes.EVAL)
    with torch.no_grad():
        for imgs, labs in test_dl:
            bs = imgs.shape[0]
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            loss = loss_criterion(out, labs)
            test_loss += loss.item() * bs
            test_accuracy += (torch.argmax(out, dim=1) == labs).sum().item()
    logger.info(f'{ds_class} Loss: {round(test_loss/len_ds, 5)}')
    logger.info(f'{ds_class} Accuracy: {round(test_accuracy/len_ds, 5)}')


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    data_path = '/opt/ml/input/data/'
    batch_size = args.bs
    logger.info(f'data_path is {data_path}')
    logger.info(f'batch_size is {batch_size}')

    logger.info("STEP 1: Creating training, validation and test dataloaders...")
    train_dl, valid_dl, test_dl = create_data_loaders(data_path, batch_size)
    logger.info("STEP 1 completed successfully...") #checks complete

    logger.info("STEP 2: Initializing and loading pretrained model for fine-tuning")
    model = net(args.arch)
    logger.info("STEP 2 completed successfully") #checks complete

    '''
    TODO: Create your loss and optimizer
    '''
    logger.info("STEP 3: Initializing Loss Function and Optimizer")
    loss_criterion = nn.CrossEntropyLoss()
    lr = args.lrate
    logger.info(f'Learning Rate is {lr}')
    optimizer = Adam(model.parameters(), lr=lr)
    logger.info("STEP 3 completed successfully") #checks complete

    hook_config = args.hook_true
    logger.info(f'Hook enabled: {hook_config}')
    if (hook_config == "True"):
        logger.info("Initializing debugger hook....")
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)
        logger.info("STEP 4: Debugger hooks initialized ....")
    else:
        print("STEP 4: Debugger/config hooks disabled by user ....")
        hook = None  #checks complete

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device available: {device}')
    model.to(device)
    logger.info(f'Pre-trained model moved to device: {device} successfully') #checks complete

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    logger.info("STEP 5: Initiating fine tuning of pre-trained model")
    epochs = args.num_epochs
    logger.info(f'Model will be trained for {epochs} Epochs')
    model = train_model(epochs, model, train_dl, loss_criterion,
                        optimizer, device, valid_dl, hook, hook_config)
    logger.info("STEP 5 completed successfully") #checks complete

    '''
    TODO: Test the model to see its accuracy
    '''

    logger.info("STEP 6: Starting Inference on test data")
    test = True
    test_model(model, test_dl, loss_criterion, device, test, hook, hook_config)
    logger.info("STEP 6 Completed successfully") #checks complete

    '''
    TODO: Save the trained model
    '''
    model_dir = '/opt/ml/model/'
    model_file = 'model.pth'
    model_path = model_dir + model_file
    logger.info(f'Saving model checkpoint file to {model_path}')

    logger.info("Step 7 Model Save - Start")
    torch.save(model.state_dict(), model_path)
    logger.info("STEP 7 completed successfully") #checks complete


if __name__ == '__main__':
    logger.info("Processing arguments now")
    args = parse_inputs()
    logger.info(f'Parsed arguments are {args}')

    '''
    TODO: Specify any training args that you might need
    '''
    logger.info("Invoking main function with parsed arguments")
    main(args)
    inference_path = '/opt/ml/model/code'
    os.makedirs(inference_path, exist_ok=True)
    if args.arch == "efficientnet_b0": #or resnet50
        os.system("cp infer_eff.py /opt/ml/model/code/inference.py")
    else:
        os.system("cp infer_resnet50.py /opt/ml/model/code/inference.py")
    logger.info("Woohoo All Done Bro. Chillofy!! go and train that chatbot now hahahaha")
