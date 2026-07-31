import argparse
import pprint
import json
import logging
import os
import sys
import shutil
import glob
import pandas as pd
import random
import time
import glob
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModel

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook



class Training():
    def __init__(self, model, optimizer, loss_fn, epochs, batch_size, train_gen, valid_gen, test_gen,
                len_train_ds, len_valid_ds, len_test_ds, device, labels_cat, hook, hook_enabled="False", oclr_sched=None, oclr=False, early_stop=True, patience=4):
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

    def train_step(self, input_tokens,token_type_ids, attention_tokens, train_label):
        self.model.train()
        if(self.hook_enabled=="True"):
            self.hook.set_mode(smd.modes.TRAIN) #Debugger config
        output = self.model(input_ids=input_tokens, token_type_ids=token_type_ids, attention_mask=attention_tokens)
        loss_value = self.loss_fn(output, train_label)
        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()
        if(self.oclr):
          self.oclr_sched.step()
        return output, loss_value  ## To be updated

    def test_step(self, input_tokens, token_type_ids, attention_tokens, validation_label):
        self.model.eval()
        if(self.hook_enabled=="True"):
            self.hook.set_mode(smd.modes.EVAL)
        with torch.inference_mode():
            val_output = self.model(input_ids=input_tokens, token_type_ids=token_type_ids, attention_mask=attention_tokens)
            loss_value = self.loss_fn(val_output, validation_label)
        return val_output, loss_value ## To be updated



    def train(self, df):
        ds_len = self.train_ds_size
        gen = self.train_generator
        loss_epoch = 0
        accuracy = 0
        avg_epoch_loss = 0
        avg_accuracy = 0

        for input_ids, token_type_ids, attention_mask, y_batch in gen:
            y_batch = y_batch.type(torch.LongTensor)
            input_ids, token_type_ids, attention_mask, y_batch = input_ids.to(self.device), token_type_ids.to(self.device), attention_mask.to(self.device), y_batch.to(self.device)
            output, loss_value = self.train_step(input_ids, token_type_ids, attention_mask, y_batch)

            pred_labls = list(np.argmax(output.detach().cpu().numpy(), axis=1))
            temp_df = pd.DataFrame({'orig_label': list(y_batch.detach().cpu().numpy()), 'pred_label': pred_labls})
            df = pd.concat([df, temp_df])
            loss_epoch += loss_value.item()*y_batch.shape[0]
            accuracy += (np.array(pred_labls) == y_batch.detach().cpu().numpy()).sum().item()

        avg_epoch_loss = loss_epoch/ds_len
        avg_accuracy = accuracy/ds_len

        return df, avg_epoch_loss, avg_accuracy

    def test_validate(self,df,valid=True,train=False):
        if(valid):
            ds_len = self.valid_ds_size
            gen = self.valid_generator
        elif(train):
            ds_len = self.train_ds_size
            gen = self.train_generator
        else:
            ds_len = self.test_ds_size
            gen = self.test_generator
        loss_epoch = 0
        accuracy = 0
        avg_epoch_loss = 0
        avg_accuracy = 0

        for input_ids, token_type_ids, attention_mask, y_batch in gen:
            y_batch = y_batch.type(torch.LongTensor)
            input_ids, token_type_ids, attention_mask, y_batch = input_ids.to(self.device), token_type_ids.to(self.device), attention_mask.to(self.device), y_batch.to(self.device)
            output, loss_value = self.test_step(input_ids, token_type_ids, attention_mask, y_batch)

            pred_labls = list(np.argmax(output.detach().cpu().numpy(), axis=1))
            temp_df = pd.DataFrame({'orig_label': list(y_batch.detach().cpu().numpy()), 'pred_label': pred_labls})
            df = pd.concat([df, temp_df])
            loss_epoch += loss_value.item()*y_batch.shape[0]
            accuracy += (np.array(pred_labls) == y_batch.detach().cpu().numpy()).sum().item()

        avg_epoch_loss = loss_epoch/ds_len
        avg_accuracy = accuracy/ds_len

        return df, avg_epoch_loss, avg_accuracy

    def train_model(self):
        for epoch in range(self.epochs):
            start_time = time.time()
            histdf_train = pd.DataFrame({'orig_label': [], 'pred_label': []})
            histdf_val   = pd.DataFrame({'orig_label': [], 'pred_label': []})
            df_train_cm, avg_epoch_train_loss, train_accuracy = self.train(histdf_train)
            df_val_cm, avg_epoch_val_loss, val_accuracy = self.test_validate(histdf_val)

            ##################################################################################
            #Following Code snippets calculate and update metric values
            ##################################################################################

            acc_class_train = self.update_metrics_on_epoch_end(df_train_cm, True, self.train_history, avg_epoch_train_loss, train_accuracy)
            acc_class_valid = self.update_metrics_on_epoch_end(df_val_cm, False, self.valid_history, avg_epoch_val_loss, val_accuracy)

            #Code for earlystopping ####
            la = self.valid_history['accuracy']
            if(self.early_stop_enabled):
                if (len(la) > 10):
                    exit_train_loop = self.early_stopping(self.patience_level)
                    if(exit_train_loop):
                        return self.train_history, self.valid_history

            ##Code for saving checkpoints
            if ((la[-1] >= 0.7) and (len(la) > 1)):
                if(la[-1] > max(la[0:-1])):
                    self.save_checkpoint(epoch, la[-1])
            elif ((la[-1] >= 0.7) and (len(la) == 1)):
                self.save_checkpoint(epoch, la[-1])
            torch.cuda.empty_cache()
            print(f'Epoch : {epoch}, Train Loss: {avg_epoch_train_loss}, Train Accuracy: {train_accuracy},\n Validation Loss: {avg_epoch_val_loss}, Validation Accuracy: {val_accuracy}')
            print(f'Time taken: {np.round((time.time() - start_time), 4)}')

        print(f'Training completed successfully. Loading best checkpoint weights ...')
        if len(self.valid_history['ckpt_name']) > 0:
            self.model.load_state_dict(torch.load(self.valid_history['ckpt_name'][-1]))
            print(f'Best checkpoint weights loaded : {self.valid_history["ckpt_name"][-1]}')
            source_file = self.valid_history["ckpt_name"][-1]
            destination_file = '/opt/ml/model/model.pth'
            shutil.copyfile(source_file, destination_file)
            ll = glob.glob('/opt/ml/model/ckpt_*')
            print("Checkpoint files to be removed:  ", ll)
            for f in ll:
                print("Removing checkpoint file: ", f)
                os.remove(f) 
                print("Removed successfully file: ", f)
            model_dir_files = glob.glob('/opt/ml/model/*')
            print("Content of /opt/ml/model/ after removing ckpt_* files:  ", model_dir_files)
            
            
        else:
            print("Ouch !! seems like you dont have any checkpoints saved matching criteria specified in logic. Please excuse !")

        return acc_class_train, acc_class_valid


    def update_metrics_on_epoch_end(self, df_cm, train, ds_dict, loss, accuracy):
        f1, precision, recall, recall_list = self.calc_rec_prec_f1(df_cm, train) #Move this to line above once f1 logic is fixed
        ds_dict['loss'].append(loss)
        ds_dict['accuracy'].append(accuracy)
        ds_dict['f1'].append(f1)
        ds_dict['precision'].append(precision)
        ds_dict['recall'].append(recall)
        return recall_list

    def calc_rec_prec_f1(self, df, train):
        print_string = "Training Confusion Matrix" if train else "Validation Confusion Matrix"
        cm = confusion_matrix(df['orig_label'], df['pred_label'])
        print(f'{print_string} \n {cm}')
        num_class = cm.shape[0]
        if (num_class>2):
            incorrect_pred_list = [i for i in range(num_class)]
            recall_list = [i for i in range(num_class)]
            prec_list = [i for i in range(num_class)]
            total_cm = np.sum(cm)
            class_weight_list = [np.sum(cm[i])/total_cm for i in range(num_class)]
            ### Code for recall
            for i in range(num_class):
                recall = cm[i,i]/np.sum(cm[i])
                recall_list[i] = recall

            ## Code for incorrect predictions
            for i in range(num_class):
                val_in = [k for k in range(num_class)]
                val_in.remove(i)
                incorrect_pred = 0
                for j in val_in:
                    incorrect_pred += cm[i,j]
                    incorrect_pred_list[i] = incorrect_pred

            ## Code for precision
            for i in range(num_class):
                val_in = [k for k in range(num_class)]
                val_in.remove(i)
                total_incorrect_preds=0
                for j in val_in:
                    total_incorrect_preds += incorrect_pred_list[j]
                    prec_list[i] = cm[i,i]/(total_incorrect_preds+cm[i,i])

            ## code for cum precision
            precision =np.sum(np.array(prec_list)*np.array(class_weight_list))
            #Code for cum recall
            recall = np.sum(np.array(recall_list)*np.array(class_weight_list))
            ## Code for F1
            ##f1 = 2*precision*recall/(precision+recall)

        else:
            tn, fp, fn, tp = confusion_matrix(df['orig_label'], df['pred_label']).ravel()
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = 2*precision*recall/(precision+recall)
            recall_list = None
        f1 = 2*precision*recall/(precision+recall)
        return f1, precision, recall, recall_list

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
                self.model.load_state_dict(torch.load(self.valid_history['ckpt_name'][-1]))
                source_file = self.valid_history["ckpt_name"][-1]
                destination_file = '/opt/ml/model/model.pth'
                shutil.copyfile(source_file, destination_file)
                ll = glob.glob('/opt/ml/model/ckpt_*')
                print("Checkpoint files to be removed:  ", ll)
                for f in ll:
                    print("Removing checkpoint file: ", f)
                    os.remove(f) 
                    print("Removed successfully file: ", f)
                model_dir_files = glob.glob('/opt/ml/model/*')
                print("Content of /opt/ml/model/ after removing ckpt_* files:  ", model_dir_files)
            else:
                print("Ouch !! seems like you dont have any checkpoints saved matching criteria specified in logic. Please excuse !")
            exit_train_loop = True
        return exit_train_loop

    def save_checkpoint(self, epoch, acc):
        ckpt_path = '/opt/ml/model/'
        #if not os.path.exists(ckpt_path):
        #    os.mkdir(ckpt_path)
        ckpt_name = ckpt_path + 'ckpt_' + "Epoch_" + str(epoch) + "_Acc_" + str(acc) + '.pt'
        torch.save(self.model.state_dict(), ckpt_name)
        self.valid_history['ckpt_name'].append(ckpt_name)
        print(f'Saved Checkpoint : {ckpt_name}')

    def get_test_accuracy(self):
        df = pd.DataFrame({'orig_label': [], 'pred_label': []})
        df_test, avg_test_loss, test_accuracy = self.test_validate(df,False)
        print(f'Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}')
        return df_test, avg_test_loss, test_accuracy


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

    def plot_accuracy(self, acc_class_train, acc_class_valid):
        df_train_acc = pd.DataFrame({'class': self.classes, 'accuracy':acc_class_train})
        df_train_acc['train_valid'] = 'train'
        df_valid_acc = pd.DataFrame({'class': self.classes, 'accuracy':acc_class_valid})
        df_valid_acc['train_valid'] = 'valid'
        df_combined = pd.concat([df_train_acc, df_valid_acc], axis=0)
        sns.barplot(data=df_combined, x='class', y='accuracy', hue='train_valid', order=self.classes)
        plt.show()



class clothrev_ds(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokens.iloc[idx]['input_ids'])
        token_type_ids = torch.tensor(self.tokens.iloc[idx]['token_type_ids'])
        attention_mask = torch.tensor(self.tokens.iloc[idx]['attention_mask'])
        labels = torch.tensor(self.labels.iloc[idx])  # type(torch.LongTensor)
        return input_ids, token_type_ids, attention_mask, labels

    def __len__(self):
        return len(self.labels)


def load_data(train_file, valid_file):
    train_ds_loc = '/opt/ml/input/data/train/'
    valid_ds_loc = '/opt/ml/input/data/validation/'
    train_file = train_file 
    valid_file = valid_file
    train_file_path = train_ds_loc + train_file
    valid_file_path = valid_ds_loc + valid_file
    df_train = pd.read_csv(train_file_path)
    df_valid = pd.read_csv(valid_file_path)
    df_train['tok_ids'] = df_train['tok_ids'].apply(eval)
    df_valid['tok_ids'] = df_valid['tok_ids'].apply(eval)
    return df_train, df_valid

class custom_bert(nn.Module):
  def __init__(self, embed_size, ckpt, num_labels, dout):
    super().__init__()

    self.checkpoint = ckpt
    code_dir = "/opt/ml/model/code/"
    os.makedirs(code_dir, exist_ok=True)
    json_path = code_dir + "config.json"
    config = AutoConfig.from_pretrained(ckpt)
    config.to_json_file(json_path)
    print("saved base auto model config to: ", json_path)
    self.basemodel = AutoModel.from_pretrained(self.checkpoint)
    self.flatten = nn.Flatten()
    self.classifier = nn.Sequential(
        nn.Linear(embed_size, embed_size, bias=False),
        nn.BatchNorm1d(embed_size),
        nn.ReLU(inplace=True),
        nn.Dropout(dout),
        nn.Linear(embed_size, num_labels)
        )
  def forward(self, input_ids, token_type_ids, attention_mask):
    x = self.basemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    x = self.flatten(x.last_hidden_state[:,0,:])
    x = self.classifier(x)
    return x

def create_model():

    embed_size = 768
    num_labels = 2
    dout = 0.5
    ckpt = 'bert-base-uncased'
    model = custom_bert(embed_size, ckpt, num_labels, dout)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    return model, tokenizer


##############################################################################
def create_ds(df_train, df_valid):
    train_ds = clothrev_ds(df_train['tok_ids'], df_train['sentiment'])
    valid_ds   = clothrev_ds(df_valid['tok_ids'], df_valid['sentiment'])
    return train_ds, valid_ds

def create_dl(train_ds, valid_ds, batch_size):
    train_dl = DataLoader(train_ds, shuffle=True, num_workers=os.cpu_count(), batch_size=batch_size)
    valid_dl = DataLoader(valid_ds, shuffle=False, num_workers=os.cpu_count(), batch_size=batch_size)
    return train_dl, valid_dl

def create_train_obj(lr, epochs, batch_size, model, train_dl, valid_dl, train_ds, valid_ds, hook, hook_enabled):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = epochs
    train_gen = train_dl
    valid_gen = valid_dl
    len_train_ds = len(train_ds)
    len_valid_ds = len(valid_ds)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels_cat = ['positive', 'negative']
    train_obj = Training(model, optimizer, loss_fn, epochs, batch_size,
                         train_gen, valid_gen, None,
                         len_train_ds, len_valid_ds, None,
                         device, labels_cat, hook, hook_enabled,
                         oclr_sched=None, oclr=False,
                         early_stop=True, patience=4
                         )
    return train_obj


def parse_args():

    parser = argparse.ArgumentParser()
    # CLI args
    parser.add_argument('--epochs',
                        type=int,
                        default=1)
    parser.add_argument('--batch_size',
                        type=int, 
                        default=64)
    parser.add_argument('--lr',
                        type=float,
                        default=0.01)
    parser.add_argument('--train_file',
                        type=str,
                        default=None)
    parser.add_argument('--valid_file',
                        type=str,
                        default=None)
    parser.add_argument('--hook_true',
                        type=str,
                        default="False")
   
    return parser.parse_args()


def process_train_model(args):
    df_train, df_valid = load_data(args.train_file, args.valid_file)
    print("Step 1: Data Load Complete ....")
    train_ds, valid_ds = create_ds(df_train, df_valid)
    print("Step 2: Dataset creation Complete ....")
    batch_size = args.batch_size
    train_dl, valid_dl = create_dl(train_ds, valid_ds, batch_size)
    print("Step 3: DataLoader creation Complete ....")
    model, tokenizer = create_model()
    print("Step 4: Model creation Complete ....")
    print("Type of hook_true  ", type(args.hook_true), "value: ", args.hook_true)
    #if isinstance(args.hook_true, str):
    #    args.hook_true = bool(args.hook_true)
    #    print("Type of hook_true  ", type(args.hook_true), "value: ", args.hook_true)
    hook_config = args.hook_true
    print("Type of hook_config  ", type(hook_config), "value: ", hook_config)
    #if isinstance(hook_config, str):
    #    hook_config = bool(hook_config)
    #print("Type of hook_config  ", type(hook_config))
    print("Hook enabled: ", hook_config)
    if (hook_config == "True"):
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)
        print("Step 5: Debugger hooks initialized ....")
    else:
        print("Step 5: Debugger/config hooks disabled by user ....")
        hook = None
    lr = args.lr
    epochs = args.epochs
    train_obj  = create_train_obj(lr, epochs, batch_size, model, train_dl, valid_dl, train_ds, valid_ds, hook, hook_config)
    print("Step 6: Training Object creation complete ....")
    train_acc, valid_acc = train_obj.train_model()
    #df_train_hist = pd.DataFrame(train_obj.train_history)
    #df_valid_hist = pd.DataFrame(train_obj.valid_history)
    #df_train_hist.to_csv('/opt/ml/output/train_hist.csv')
    #df_valid_hist.to_csv('/opt/ml/output/valid_hist.csv')
    print("Congratulations !! Model training completed successfully ...")


if __name__ == '__main__':
    args = parse_args()
    print(args.__dict__)
    process_train_model(args)

    inference_path = "/opt/ml/model/code/"
    os.makedirs(inference_path, exist_ok=True)
    os.system("cp inference.py {}".format(inference_path))
    os.system("cp requirements.txt {}".format(inference_path))
    #os.system("cp config.json {}".format(inference_path))