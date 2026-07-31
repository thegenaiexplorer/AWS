#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
##Implement OCLR/LR Scheduler + option to run full runing or only classifier layer
import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score

print("before transformer import")
import transformers
from transformers import ViTModel
print("after transformer import", transformers.__version__)

import os
import argparse
import sys
import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import shutil
import glob
import pandas as pd
import random
import time
import glob
import numpy as np
import matplotlib.pyplot as plt

#import smdebug.pytorch as smd
#from smdebug import modes
#from smdebug.profiler.utils import str2bool
#from smdebug.pytorch import get_hook
from torch.optim.lr_scheduler import StepLR,OneCycleLR


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


#TODO: Import dependencies for Debugging andd Profiling


class Training():
    def __init__(self, model, optimizer, loss_fn, epochs, batch_size, train_gen, valid_gen, test_gen,
                 len_train_ds, len_valid_ds, len_test_ds, device, labels_cat, ckpt_thrshld, ckpt_path,
                 hook, hook_enabled="False", datatype ="vision", oclr_sched=None, oclr=False, 
                 early_stop=True, patience=4):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_generator = train_gen
        self.valid_generator = valid_gen
        self.test_generator = test_gen
        self.train_ds_size = len_train_ds
        self.valid_ds_size = len_valid_ds
        self.test_ds_size = len_test_ds
        self.train_history = {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        self.valid_history = {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'ckpt_name': []}
        self.early_stop_enabled = early_stop
        self.patience_level = patience
        self.device = device
        self.classes = labels_cat
        self.oclr = oclr
        if(self.oclr):
          self.oclr_sched = oclr_sched
        self.hook_enabled = hook_enabled
        if(self.hook_enabled == "True"):
            self.hook = hook
        self.train_fn = self.train_vision
        self.eval_fn = self.validate_vision
        self.datatype = datatype
        if datatype == 'text':
            self.train_fn = self.train_text
            self.eval_fn = self.validate_text
        self.ckpt_thrshld = ckpt_thrshld
        self.ckpt_path = ckpt_path

    def train_text(self, df):
        ds_len = self.train_ds_size
        gen = self.train_generator
        loss_epoch = 0
        accuracy = 0
        avg_epoch_loss = 0
        avg_accuracy = 0
        self.model.train()
        if (self.hook_enabled == "True"):
            self.hook.set_mode(smd.modes.TRAIN)
        for input_ids, token_type_ids, attention_mask, y_batch in gen:
            y_batch = y_batch.type(torch.LongTensor)
            input_ids, token_type_ids = input_ids.to(self.device), token_type_ids.to(self.device)
            attention_mask, y_batch = attention_mask.to(self.device), y_batch.to(self.device)
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss_value = self.loss_fn(output, y_batch)
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()
            if (self.oclr):
                self.oclr_sched.step()
            pred_labls = list(np.argmax(output.detach().cpu().numpy(), axis=1))
            temp_df = pd.DataFrame({'orig_label': list(y_batch.detach().cpu().numpy()), 'pred_label': pred_labls})
            df = pd.concat([df, temp_df])
            loss_epoch += loss_value.item()*y_batch.shape[0]
            accuracy += (np.array(pred_labls) == y_batch.detach().cpu().numpy()).sum().item()

        avg_epoch_loss = loss_epoch/ds_len
        avg_accuracy = accuracy/ds_len
        df = df.astype(int)
        return df, avg_epoch_loss, avg_accuracy


    def train_vision(self, df):
        ds_len = self.train_ds_size
        gen = self.train_generator
        loss_epoch = 0
        accuracy = 0
        avg_epoch_loss = 0
        avg_accuracy = 0
        self.model.train()
        if (self.hook_enabled == "True"):
            self.hook.set_mode(smd.modes.TRAIN)

        for x_batch, y_batch in gen:
            y_batch = y_batch.type(torch.LongTensor)
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            output = self.model(x_batch)
            loss_value = self.loss_fn(output, y_batch)
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()
            if(self.oclr):
                self.oclr_sched.step()
            pred_labls = list(np.argmax(output.detach().cpu().numpy(), axis=1))
            temp_df = pd.DataFrame({'orig_label': list(y_batch.detach().cpu().numpy()), 'pred_label': pred_labls})
            df = pd.concat([df, temp_df])
            loss_epoch += loss_value.item()*x_batch.shape[0]
            accuracy += (np.array(pred_labls) == y_batch.detach().cpu().numpy()).sum().item()

        avg_epoch_loss = loss_epoch/ds_len
        avg_accuracy = accuracy/ds_len
        df = df.astype(int)
        return df, avg_epoch_loss, avg_accuracy

    def validate_text(self, df, gen):
        loss_epoch = 0
        accuracy = 0
        self.model.eval()
        if (self.hook_enabled == "True"):
            self.hook.set_mode(smd.modes.EVAL)
        for input_ids, token_type_ids, attention_mask, y_batch in gen:
            y_batch = y_batch.type(torch.LongTensor)
            input_ids, token_type_ids,  = input_ids.to(self.device), token_type_ids.to(self.device)
            attention_mask, y_batch = attention_mask.to(self.device), y_batch.to(self.device)

            with torch.inference_mode():
                output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss_value = self.loss_fn(output, y_batch)
            pred_labls = list(np.argmax(output.detach().cpu().numpy(), axis=1))
            temp_df = pd.DataFrame({'orig_label': list(y_batch.detach().cpu().numpy()), 'pred_label': pred_labls})
            df = pd.concat([df, temp_df])
            loss_epoch += loss_value.item()*y_batch.shape[0]
            accuracy += (np.array(pred_labls) == y_batch.detach().cpu().numpy()).sum().item()
        df = df.astype(int)
        return df, loss_epoch, accuracy

    def validate_vision(self, df, gen):
        loss_epoch = 0
        accuracy = 0
        self.model.eval()
        if (self.hook_enabled == "True"):
            self.hook.set_mode(smd.modes.EVAL)
        for x_batch, y_batch in gen:
            y_batch = y_batch.type(torch.LongTensor)
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            with torch.inference_mode():
                output = self.model(x_batch)
                loss_value = self.loss_fn(output, y_batch)
            pred_labls = list(np.argmax(output.detach().cpu().numpy(), axis=1))
            temp_df = pd.DataFrame({'orig_label': list(y_batch.detach().cpu().numpy()), 'pred_label': pred_labls})
            df = pd.concat([df, temp_df])
            loss_epoch += loss_value.item()*x_batch.shape[0]
            accuracy += (np.array(pred_labls) == y_batch.detach().cpu().numpy()).sum().item()
        df = df.astype(int)
        return df, loss_epoch, accuracy

    def test_validate(self, df, valid=True, train=False):
        if(valid):
            ds_len = self.valid_ds_size
            gen = self.valid_generator
        elif(train):
            ds_len = self.train_ds_size
            gen = self.train_generator
        else:
            ds_len = self.test_ds_size
            gen = self.test_generator

        avg_epoch_loss = 0
        avg_accuracy = 0

        df, loss_epoch, accuracy = self.eval_fn(df, gen)
        avg_epoch_loss = loss_epoch/ds_len
        avg_accuracy = accuracy/ds_len
        return df, avg_epoch_loss, avg_accuracy

    def train_model(self):
        for epoch in range(self.epochs):
            start_time = time.time()
            histdf_train = pd.DataFrame({'orig_label': [], 'pred_label': []})
            histdf_val   = pd.DataFrame({'orig_label': [], 'pred_label': []})
            df_train_cm, avg_epoch_train_loss, train_accuracy = self.train_fn(histdf_train)
            df_val_cm, avg_epoch_val_loss, val_accuracy = self.test_validate(histdf_val)

            ##################################################################################
            #Following Code snippets calculate and update metric values
            ##################################################################################

            self.update_metrics_on_epoch_end(df_train_cm, True, self.train_history,
                                             avg_epoch_train_loss, train_accuracy)
            self.update_metrics_on_epoch_end(df_val_cm, False, self.valid_history,
                                             avg_epoch_val_loss, val_accuracy)

            #Code for earlystopping ####
            la = self.valid_history['accuracy']
            if(self.early_stop_enabled):
                if (len(la) > 10):
                    exit_train_loop = self.early_stopping(self.patience_level)
                    if(exit_train_loop):
                        return self.train_history, self.valid_history

            ##Code for saving checkpoints
            print(f'Accuracy should be greater than {self.ckpt_thrshld} to save model weights')
            if ((la[-1] >= self.ckpt_thrshld) and (len(la) > 1)):
                if(la[-1] > max(la[0:-1])):
                    self.save_checkpoint(epoch, la[-1], avg_epoch_train_loss)
            elif ((la[-1] >=self.ckpt_thrshld) and (len(la) == 1)):
                self.save_checkpoint(epoch, la[-1], avg_epoch_train_loss)
            torch.cuda.empty_cache()
            print(f'Epoch : {epoch}, Train Loss: {avg_epoch_train_loss}, Train Accuracy: {train_accuracy},\n Validation Loss: {avg_epoch_val_loss}, Validation Accuracy: {val_accuracy}')
            print(f'Time taken: {np.round((time.time() - start_time), 4)}')

        print(f'Training completed successfully. Loading best checkpoint weights ...')
        if len(self.valid_history['ckpt_name']) > 0:
            ckpt_file = self.valid_history['ckpt_name'][-1]
            checkpoint = torch.load(ckpt_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Best checkpoint weights loaded : {self.valid_history["ckpt_name"][-1]}')
            source_file = self.valid_history["ckpt_name"][-1]
            destination_file = self.ckpt_path + 'model.pth'
            shutil.copyfile(source_file, destination_file)
            ll = glob.glob(self.ckpt_path + 'ckpt_*')
            print("Checkpoint files to be removed:  ", ll)
            for f in ll:
                print("Removing checkpoint file: ", f)
                os.remove(f) 
                print("Removed successfully file: ", f)
            model_dir_files = glob.glob(self.ckpt_path + '*')
            print(f'Content of {self.ckpt_path} after removing ckpt_* files:\n {model_dir_files}')
            df_test, avg_test_loss, test_accuracy = self.get_accuracy()
            logger.info(f'Average test loss is {avg_test_loss}')
            logger.info(f'Average test accuracy is {test_accuracy}')
        else:
            print("Ouch !! seems like you dont have any checkpoints saved matching criteria specified in logic. Please excuse !")
        print("Displaying Training Plots")
        self.plot_metrics()
        logger.info("Training Loop completed. Exiting now")

    def update_metrics_on_epoch_end(self, df_cm, train, ds_dict, loss, accuracy):
        f1, precision, recall = self.calc_rec_prec_f1(df_cm, train) #Move this to line above once f1 logic is fixed
        ds_dict['loss'].append(loss)
        ds_dict['accuracy'].append(accuracy)
        ds_dict['f1'].append(f1)
        ds_dict['precision'].append(precision)
        ds_dict['recall'].append(recall)

    def calc_rec_prec_f1(self, df, train):
        print_string = "Training Confusion Matrix" if train else "Validation Confusion Matrix"
        cm = confusion_matrix(df['orig_label'], df['pred_label'])
        print(f'{print_string} \n {cm}')
        num_class = cm.shape[0]
        if (num_class>2):
            recall = recall_score(df['orig_label'], df['pred_label'],average='weighted')
            precision = precision_score(df['orig_label'], df['pred_label'],average='weighted')
            f1 = f1_score(df['orig_label'], df['pred_label'],average='weighted')
        else:
            tn, fp, fn, tp = confusion_matrix(df['orig_label'], df['pred_label']).ravel()
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = 2*precision*recall/(precision+recall)
        return f1, precision, recall

    def early_stopping(self, early_stop_level):
        la = self.valid_history['accuracy']
        exit_train_loop = False
        n = -1
        while n > -(early_stop_level):
            if not (la[n] <= la[n-1]):
                break
            n = n-1
        if abs(n) == early_stop_level:
            print("Validation accuracy has not improved over last 4 epochs, hence invoking early stopping and exiting training process")
            if len(self.valid_history['ckpt_name']) > 0:
                print('Loading Best weights..')
                #self.model.load_state_dict(torch.load(self.valid_history['ckpt_name'][-1]))
                ckpt_file = self.valid_history['ckpt_name'][-1]
                checkpoint = torch.load(ckpt_file)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                source_file = self.valid_history["ckpt_name"][-1]
                destination_file = self.ckpt_path + 'model.pth'
                shutil.copyfile(source_file, destination_file)
                ll = glob.glob(self.ckpt_path + 'ckpt_*')
                print("Checkpoint files to be removed:  ", ll)
                for f in ll:
                    print("Removing checkpoint file: ", f)
                    os.remove(f) 
                    print("Removed successfully file: ", f)
                model_dir_files = glob.glob(self.ckpt_path+'*')
                print(f'Content of {self.ckpt_path} after removing ckpt_* files: {model_dir_files}')
            else:
                print("Ouch !! seems like you dont have any checkpoints saved matching criteria specified in logic. Please excuse !")
            exit_train_loop = True
        return exit_train_loop

    def save_checkpoint(self, epoch, acc, avg_epoch_train_loss):
        os.makedirs(self.ckpt_path, exist_ok=True)
        ckpt_name = self.ckpt_path + 'ckpt_' + "Epoch_" + str(epoch) + "_Acc_" + str(acc) + '.pt'
        #torch.save(self.model.state_dict(), ckpt_name) #original
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_epoch_train_loss,
            'device': self.device
        }, ckpt_name)
            
        self.valid_history['ckpt_name'].append(ckpt_name)
        print(f'Saved Checkpoint : {ckpt_name}')

    def get_accuracy(self, valid=False, train=False):
        df = pd.DataFrame({'orig_label': [], 'pred_label': []})
        df, avg_loss, avg_accuracy = self.test_validate(df, valid, train)
        if train:
            run_type = "Train"
        elif valid:
            run_type = "Validation"
        else:
            run_type = "Test"
        print(f'{run_type} Loss: {avg_loss}, {run_type} Accuracy: {avg_accuracy}')
        return df, avg_loss, avg_accuracy


    def plot_metrics(self):
        list = ['accuracy', 'loss', 'f1', 'recall', 'precision']
        for metrics in list:
            x = [i for i in range(len(self.train_history[metrics]))]
            y1 = self.train_history[metrics]
            y2 = self.valid_history[metrics]
            fig, ax = plt.subplots()
            ax.plot(x, y1, label="train_"+metrics)
            ax.plot(x, y2, label="valid_"+metrics)
            plt.suptitle(f'Plot showing {metrics} for Different Epoch Runs')
            ax.legend()
            plt.show()
            save_path = self.ckpt_path + metrics + '.png'
            fig.savefig(save_path)   # save the figure to file
            plt.close(fig)


def create_train_obj(lr_scheduler, lr, epochs, batch_size, model, train_dl, valid_dl, test_dl,
                     train_ds, valid_ds, test_ds, hook, hook_enabled, datatype, ckpt_thrshld, ckpt_path):
    logger.info("Initializing Loss Function and Optimizer")
    loss_fn = torch.nn.CrossEntropyLoss()
    logger.info(f'Learning Rate is {lr}')
    optimizer = Adam(model.parameters(), lr=lr)
    use_sched = True
    # if args.lr_scheduler == "StepLR":
    #    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    #    logger.info("StepLR initialized")
    if lr_scheduler == "OCLR":
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_dl))
        oclr_sched = scheduler
        logger.info("OCLR initialized")
    else:
        oclr_sched = None
        use_sched = False
    logger.info("Loss Function and optimizer initialization completed")

    train_gen = train_dl
    valid_gen = valid_dl
    test_gen = test_dl
    len_train_ds = len(train_ds)
    len_valid_ds = len(valid_ds)
    len_test_ds = len(test_ds)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device available: {device}')
    labels_cat = None  # ['positive', 'negative'] #what is the essence of this ?? Find out and address
    train_obj = Training(model, optimizer, loss_fn, epochs, batch_size,
                         train_gen, valid_gen, test_gen,
                         len_train_ds, len_valid_ds, len_test_ds,
                         device, labels_cat, ckpt_thrshld, ckpt_path, hook, hook_enabled,
                         datatype=datatype, oclr_sched=oclr_sched, oclr=use_sched,
                         early_stop=True, patience=4)
    return train_obj

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
    parser.add_argument('--do',
                        type=float,
                        default=0.2,
                        help="Dropout to use for model training. Default: 0.2",
                        )
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help="Number of classes in classification. Default: 2",
                        )
    parser.add_argument('--pre_weight_freeze',
                        type=str,
                        default="True",
                        help="If the pretraining weights be frozen. Default: True",
                        )
    parser.add_argument('--num_cls_blocks',
                        type=int,
                        default=1,
                        help="Number of blocks in classifier layer. Default: 1",
                        )
    parser.add_argument('--lr_scheduler',
                        type=str,
                        default="AdamOnly",
                        help="Set learning rate scheduler. Valid values: OCLR, AdamOnly. Default: AdamOnly",
                        )
    parser.add_argument('--image_augment',
                        type=str,
                        default="advanced",
                        help="Set the transforms level Valid values: advanced, basic. Default: advanced",
                        )
    parser.add_argument('--datatype',
                        type=str,
                        default="vision",
                        help="Set the datatype values: vision, text. Default: vision",
                        )
    parser.add_argument('--environ',
                        type=str,
                        default="colab",
                        help="Set the cloud environment values: aws, colab. Default: colab",
                        )
    parser.add_argument('--ckpt_thrshld',
                        type=float,
                        default=0.85,
                        help="minimum accuracy threshold required to save model . Default: 0.85",
                        )
    
    args = parser.parse_args()
    log_text = 'Starting Hyperparameter tuning Job with '
    logger.info(f'{log_text} Learning Rate: {args.lrate}, Dropout: {args.do}, Number of Classes: {args.num_class}')
    logger.info(f'Batch Size: {args.bs}, Epochs: {args.num_epochs}, Arch: {args.arch}')
    logger.info(f'Freeze pretrained weights: {args.pre_weight_freeze}, Classifier Blocks: {args.num_cls_blocks}')
    logger.info(f'LR Scheduler: {args.lr_scheduler}, image augmentation: {args.image_augment}')
    return args


def create_data_loaders(data_path, batch_size, image_augment):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dir = data_path + 'train'
    valid_dir = data_path + 'validation'
    test_dir = data_path + 'test'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if image_augment == "advanced":
        train_transforms = T.Compose([
            T.Resize(256),
            T.RandomAffine(scale=(0.9, 1.1), translate=(0.1, 0.1), degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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
    else:
        train_transforms = T.Compose([
            T.Resize(256),
            T.RandomRotation(30),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop((224,224)),
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

    return train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds

class VIT_TL(nn.Module):
    def __init__(self, do, num_class, pre_weight_freeze, num_cls_blocks):
        super().__init__()
        self.basemodel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        if pre_weight_freeze == "True":
            for params in self.basemodel.parameters():
                params.requires_grad = False
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

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    data_path = '/opt/ml/input/data/'
    inference_path = '/opt/ml/model/code'
    ckpt_path = '/opt/ml/model/'
    if args.environ != 'aws':
        data_path = './opt/ml/input/data/'
        inference_path = './opt/ml/model/code'
        ckpt_path = './opt/ml/model/'
        os.makedirs(inference_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        
    batch_size = args.bs
    logger.info(f'data_path is {data_path}')
    logger.info(f'batch_size is {batch_size}')

    logger.info("STEP 1: Creating training, validation and test dataloaders...")
    train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds = create_data_loaders(data_path,
                                                                                   batch_size,
                                                                                   args.image_augment)
    logger.info("STEP 1 completed successfully...") #checks complete

    logger.info("STEP 2: Initializing and loading pretrained model for fine-tuning")
    model = net(args.arch, args.do, args.num_class, args.pre_weight_freeze, args.num_cls_blocks)
    logger.info("STEP 2 completed successfully") #checks complete

    hook_config = args.hook_true
    logger.info(f'Hook enabled: {hook_config}')
    if (hook_config == "True"):
        logger.info("Initializing debugger hook....")
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)
        logger.info("STEP 3: Debugger hooks initialized ....")
    else:
        print("STEP 3: Debugger/config hooks disabled by user ....")
        hook = None  #checks complete

    logger.info("STEP 4:Creating Training Object")
    lr = args.lrate
    epochs = args.num_epochs
    datatype = args.datatype
    ckpt_thrshld = args.ckpt_thrshld
    lr_scheduler = args.lr_scheduler
    
    train_obj = create_train_obj(lr_scheduler, lr, epochs, 
                                 batch_size, model, train_dl, valid_dl, test_dl,
                                 train_ds, valid_ds, test_ds,
                                 hook, hook_config, datatype, ckpt_thrshld, ckpt_path)
    
    logger.info("STEP 4 completed successfully")

    logger.info("STEP 5: Initiating fine tuning of pre-trained model")
    logger.info(f'Model will be trained for {epochs} Epochs')
    train_obj.train_model()
    logger.info("STEP 5 completed successfully")
    logger.info("#############################################################")
    logger.info("########## MODEL TRAINING COMPLETED SUCCESSFULLY ############")

    logger.info(f'STEP 6: Saving training, validation and test dataframes with columns orig_label and pred_label to {ckpt_path}')
    df_train, _, _ = train_obj.get_accuracy(valid=False, train=True)
    df_valid, _, _ = train_obj.get_accuracy(valid=True, train=False)
    df_test, _ , _ = train_obj.get_accuracy()

    df_train.to_csv(ckpt_path + "train_cm.csv")
    df_valid.to_csv(ckpt_path + "valid_cm.csv")
    df_test.to_csv(ckpt_path + "test_cm.csv")
    logger.info("STEP 6 Completed successfully")

    logger.info(f'STEP 7: Saving training and validation training history to {ckpt_path}')
    df_train_hist = pd.DataFrame(train_obj.train_history)
    valid_history = train_obj.valid_history
    save_valid_history = {k: v for k, v in valid_history.items() if k != 'ckpt_name'}
    df_valid_hist = pd.DataFrame(save_valid_history)
    
    df_train_hist.to_csv(ckpt_path+'train_hist.csv')
    df_valid_hist.to_csv(ckpt_path+'valid_hist.csv')
    logger.info("STEP 7 completed successfully")  # checks complete


if __name__ == '__main__':
    logger.info("Processing arguments now")
    args = parse_inputs()
    logger.info(f'Parsed arguments are {args}')
    logger.info("Invoking main function with parsed arguments")
    main(args)

    logger.info("Copying inference scripts")
    cwd = os.getcwd()
    ls_content = os.listdir(".")
    logger.info(f'Current working directory is {cwd} and its contents are {ls_content}')

    inference_path = '/opt/ml/model/code'
    os.makedirs(inference_path, exist_ok=True)
    if args.arch == "ViT":  #or resnet50
        os.system("cp infer_vit.py /opt/ml/model/code/inference.py")
        os.system("cp requirements.txt /opt/ml/model/code/requirements.txt")
    else:
        os.system("cp infer_resnet50.py /opt/ml/model/code/inference.py")
    logger.info("Woohoo All Done Bro. Chillofy!! Prune and Quantize that Model")