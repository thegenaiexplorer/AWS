import json
import sys
import logging
import torch
from torch import nn
import os
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

###################################
### VARIABLES 
###################################

# Needs to be called 'model.pth' as per 
# https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/6936c08581e26ff3bac26824b1e4946ec68ffc85/src/sagemaker_pytorch_serving_container/torchserve.py#L45
MODEL_NAME = 'model.pth'

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LEN = 128

classes = [0, 1]

# Load Hugging Face Tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###################################
### SAGEMKAER LOAD MODEL FUNCTION 
###################################   

# You need to put in config.json from saved fine-tuned Hugging Face model in code/ 
# Reference it in the inference container at /opt/ml/model/code

class custom_bert(nn.Module):
    def __init__(self, embed_size=768, ckpt=PRE_TRAINED_MODEL_NAME, num_labels=2, dout=0.5):
        super().__init__()
        self.checkpoint = ckpt
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

def model_fn(model_dir):
    logger.info("Starting model_fn script")
    model = custom_bert()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info("Model weight being loaded")
    chkpt_file_path = os.path.join(model_dir, "model.pth") 
    model.load_state_dict(torch.load(chkpt_file_path, map_location=torch.device(device)))
    print(f'Checkpoint weights loaded from : {chkpt_file_path}')
    model.eval()
    logger.info("Model weight load complete")
    logger.info("Finished model_fn script")
    return model


###################################
### SAGEMKAER PREDICT FUNCTION 
###################################   

def predict_fn(input_data, model):
    model.eval()

    logger.info("########Starting predict_fn script##########")
    logger.info("###### ATTENTION  #####")

    print('input_data: {}'.format(input_data))
    print('type(input_data): {}'.format(type(input_data)))
    
    data_str = input_data.decode('utf-8')
    print('data_str: {}'.format(data_str))
    print('type data_str: {}'.format(type(data_str)))
    
    jsonlines = data_str.split("\n")
    print('jsonlines: {}'.format(jsonlines))
    print('type jsonlines: {}'.format(type(jsonlines)))

    predicted_classes = []

    for jsonline in jsonlines:
        logger.info("Entering the prediction loop")
        print('jsonline: {}'.format(jsonline))
        print('type jsonline: {}'.format(type(jsonline)))

        # features[0]:  review_body
        # features[1..n]:  is anything else (we can define the order ourselves)
        # Example:  
        #    {"features": ["The best gift ever", "Gift Cards"]}        
        #
        review_body = json.loads(jsonline)["features"][0]
        print("""review_body: {}""".format(review_body))

        logger.info(f'MY TEXT INPUT -- review body text is: {review_body}')
    
        encoded_tokens = TOKENIZER(
            review_body,
            max_length=MAX_SEQ_LEN,
            padding=True,
            return_tensors='pt',
            truncation=True
        )

        logger.info("HELLO HELLO HELLO tokenizer completed successfully")
        
        input_ids = encoded_tokens['input_ids'].to(device)
        token_type_ids = encoded_tokens['token_type_ids'].to(device)
        attention_mask = encoded_tokens['attention_mask'].to(device)

        logger.info("Entering model prediction now")

        output = model(input_ids, token_type_ids, attention_mask)
        logger.info(f'BRUHHHHH I got the output: {output}')
        print('output: {}'.format(output))

        # output is a tuple: 
        # output: (tensor([[-1.9840, -0.9870,  2.8947]], grad_fn=<AddmmBackward>),
        # for torch.max() you need to pass in the tensor, output[0]  

        softmax_fn = nn.Softmax(dim=1)
        softmax_output = softmax_fn(output)
        print("softmax_output: {}".format(softmax_output))

        logger.info('STAGE 1 complete')
        
        probability_list, prediction_label_list = torch.max(softmax_output, dim=1)

        logger.info('STAGE 2 complete')

        # extract the probability
        probability = probability_list.item()
        print('probability: {}'.format(probability))

        logger.info('STAGE 3complete')

        # extract the predicted label
        predicted_label_idx = prediction_label_list.item()
        predicted_label = classes[predicted_label_idx]
        print('predicted_label: {}'.format(predicted_label))

        logger.info('STAGE 4 complete')

        # configure the response dictionary
        prediction_dict = {}
        prediction_dict['probability'] = probability
        prediction_dict['predicted_label'] = predicted_label

        logger.info('STAGE 5 complete')

        jsonline = json.dumps(prediction_dict)
        print('jsonline: {}'.format(jsonline))

        predicted_classes.append(jsonline)
        print('predicted_classes in the loop: {}'.format(predicted_classes))

    predicted_classes_jsonlines = '\n'.join(predicted_classes)
    print('predicted_classes_jsonlines: {}'.format(predicted_classes_jsonlines))

    return predicted_classes_jsonlines


###################################
### SAGEMAKER MODEL INPUT FUNCTION 
################################### 

def input_fn(serialized_input_data, content_type='application/jsonlines'): 
    return serialized_input_data

###################################
### SAGEMAKER MODEL OUTPUT FUNCTION 
################################### 

def output_fn(prediction_output, accept='application/jsonlines'):
    return prediction_output, accept
