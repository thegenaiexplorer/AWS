# Inventory Monitoring at Distribution Centers 

In distribution Centers like in that of Amazon/Flipkart, robots are often used to move objects between different warehouse locations as part of their day-to-day operations. Different objects depending on their size are clubbed together in a single bin for efficient movement of objects. So, a single bin can carry one, two three, four or five objects (or even more). This project aims to build a Computer Vision Model that can be used to count number of objects in each bin. A system like this can be used to track inventory and make sure that delivery 
consignments have the correct number of items.

## Project Set Up and Installation
##### Main Files, folders and scripts used in this project:

**This project needs to be run from sagemaker notebook environment. You need to create sagemaker notebook and start jupyter files with `conda_pytorch_p310 kernel`. Docker, somehow, does not works good in sagemaker studio environment. Also there are permission issues with respect to docker image creation/upload and fetching in sagemaker studio environment.**

1) sagemaker_v5-FinalSubmission.ipynb is the main ipython file around which entire project revolves. This contains data upload, model training, deployment in different scenarios, creation of lambda function, integration with AWS API GAteway. API Gateway integration is done using AWS API Gateway GUI Interface.
2) sagemaker_pipeline-final.ipynb contains the pipeline implementation. This will at the least require container image created in sagemaker_v5-FinalSubmission.ipynb so, it is preferable to run it after executing ipython notebook mentioned in point no 1 stated above.
3) sagemaker_v5-FinalSubmission-GradioTestAPIGateway can be used for performing inference using gradio interface. This should be run after execution of sagemaker_v5-FinalSubmission.ipynb ipython notebook file. 
2) src directory contains two  important scripts:
a) fg_create.py - this is used by sagemaker processor job to create feature group - used in sagemaker_v5-FinalSubmission.ipynb
b) test_evaluate.py - this is used by evaluation step in pipeline creation for evaluating performance of model. - used in sagemaker_pipeline-final.ipynb
c) requirements.txt --> This is used by fg_create.py processing job. For any other training job mentioning src as source directory this file should be fully commented out
3) train.py is used for training in all scenarios - static hyperparameter, hyperparameter ranges, debugging/profiling. This script is integrated in our container.
4) Correct inference.py is automatically installed at the end of training job and is automated and integrated with our container
5) profiler-reports/profiler-report.html containing debugging/profiling report that is generated during/post training job when run with sagemaker debugger/profiler
6) dkr folder contains train.py, infer_eff.py(inference script for efficientnet_b0 model), infer_resnet50.py(inference script for resnet50 model) and Dockerfile (used for docker image creation)
7) lambda_function.py contains lambda function details. This is named as bincount in lambda set and integrated AWS API GAteway

##### How to run this project?
1) Create a sagemaker notebook instance
2) install the required files mentioned below 
sagemaker_v5-FinalSubmission.ipynb, sagemaker_pipeline-final.ipynb, sagemaker_v5-FinalSubmission-GradioTestAPIGateway(you wil need to install gradio), file_list.json
src/requirements.txt, src/fg_create.py, src/test_evaluate.py
profiler-reports/profiler-report.html
dkr/Dockerfile, dkr/train.py, dkr/infer_eff.py, dkr/infer_resnet50.py
Root Directory for these files is /home/ec2-user/SageMaker/capstone/nd009t-capstone-starter/starter
3) Execute code in sagemaker_v5-FinalSubmission.ipynb, sagemaker_pipeline-final.ipynb, sagemaker_v5-FinalSubmission-GradioTestAPIGateway in order.

##### What this project accomplishes?

1) Downloads project data - We use feature groups here
2) Rearranges downloaded data into training, validation and test datasets
3) Uploads  training, validation and test datasets to S3 location
4) Trains 2 SOTA Models - resnet50, efficientnet_B0 training and validation/test accuracy -we use our own custom docker image
5) Picks up the best model from point no 4 above, then performs hypertuning on it
6) From point no 5, it gets best hyperparameters, reruns the best model with these hyperparameters along with debugging and profiling
7) Deploy the trained model obtained from step 6 to sagemaker end-point and performs inference against it
8) Performs multi-instance training using resnet50 model architecture
9) All instance types used are GPU enabled.
10) Deploys endpoints in standalone mode, multi-model mode, multi-model endpoint using different production variants
11) Creates automated machine learning pipeline execution
12) Utilizes lambda function integrated with AWS API Gateway to perfrom inference utilizing a public inference URL

## Dataset

### Overview
**We are going to use Amazon Bin Image Dataset which has 500,000 bin images containing one or more objects. Due to resource constraints, we will be using subset of dataset provided with project. 
The subset contains following classes of image files followed by number of images belonging to 
each class. Total number of images in sample dataset is 10441. 
Class 1: 1228  
Class 2: 2299  
Class 3: 2666  
Class 4: 2373  
Class 5: 1875  
We will divide the dataset subset into training, validation and test datasets in following 
proportions stratified by image classes:  
Train – 90%  - 9396 images 
Validation – 5% - 522 Images 
Test – 5%  - 523 images 
Details about full dataset can be found at  https://registry.opendata.aws/amazon-bin-imagery/ 
Class refers to number objects contained in a bin.**

### Access
**1) We take subsample of dataset and split it into train/validation/test datasets and then populate the original datafile location of images in a feature store. We do so by using a sagemaker processing job. This is to ensure reusability of dataset between different 
groups in an organization. 
**2) From feature group, we extract datafile locations of images belonging to training/validation and test datasets.  
**3) Next, we download the data from locations extracted in step 2 into local compute environment. 
**4) There are 5 image classes in dataset numbered from 1 to 5. Now issue here is softmax indexing is zero based so we need to renumber image classes to n-1( from 0 to 4). Accordingly, image class folders are renumbered under each data split. 
**5) Next, we upload the data from local compute environment to our S3 location.**

## Model Training
We have utilized CNN based architecture to solve the problem at hand. Specifically, we have used two SOTA models resnet50 and efficientnet_b0. When it comes to speed and accuracy, resnet50 combines best of both the worlds. Efficientnet_b0 is little behind with respect to accuracy however it is lightweight and highly performant model, more suitable to be deployed on small devices. Resnet50 is little bulky, however, considering the size of modern-day vision models based on transformers architecture, this doesn’t seem to be a negative point for resnet50. Accordingly, efficientnet_b0 is fastest to train and infer. Resnet50 comes close. So, with respect to speed, accuracy and inference times, these two models form good choice to start with. We train both the models using a predefined set of hyperparameters. Once the training is complete, we compare the validation/test accuracy on both the models. Model producing higher validation/test accuracy is chosen and we perform hyperparameter tuning on it. If validation accuracies are close then we chose the model with high test accuracy. Post hyperparameter tuning, we take the hyperparameters corresponding to best performing model checkpoint and train it further using sagemaker debugger and profiler options. We also use the same set of hyperparameters to perform multi-instance training. 

Please see below validation and test accuracy reported during different phases of the project. 
Run 1: Initial run with set hyperparameters: 
Resnet50: 
Validation Accuracy: 0.3180 
Test Accuracy: 0.3575 
Efficientnet_b0:  
Validation Accuracy:  0.3199 
Test accuracy: 0.3212 
Here, we chose resnet50 because validation accuracy of both the models are very close, however, test accuracy of resnet50 exceeds that of efficientnet_b0 significantly. 
Run 2: Hyperparameter tuning results: 
Resnet50: 
Validation Accuracy: 0.32759  
Test Accuracy: 0.34417 
Run 3: Training Run with debugger/profiler: 
Resnet50: 
Validation Accuracy: 0.31992 
Test Accuracy: 0.35564 
Run 4: Multi Instance Training Run: 
Resnet50:  
Validation accuracy: 0.31034 
Test accuracy: 0.38432 
Since the test accuracy of multi-instance model is best, we chose this model checkpoint for final endpoint deployment in different configurations. 
As stated earlier in the document, the measuring benchmark that we are using is any model performing better than 30% validation/test accuracy is a good model given project constraints. As we can see during runs 2,3 and 4 validation accuracy of model is approximately in a very close range, however, test accuracy of model trained during run 4 beats others by a significant margin. So, we can say that resnet50 model trained during run 4 is the best performing model and can be used to adequately solve the problem within given constraints.


## Machine Learning Pipeline
Post data preprocessing (Steps 1-5 above), implementation is carried out as follows: 
1) We create docker image integrating our training script(train.py) and inference scripts for different models(resnet50 and efficientnet_b0) along with it. We push the docker image to ECR and use its ECR Image path for training jobs. Since we already have training container, we can carry out training using a normal Estimator object. Please note that train.py has been written in such a way that it can be used with static hyperparameters, hyperparameter ranges and with/without sagemaker debugger/profiler.  So, the same train.py is used in all training jobs. 
2) We train resnet50 and efficientnet_b0 models using predefined hyperparameters.  
3) We compare their validation/test accuracies. Model with higher validation/test accuracy is used in further trainings and the other one is dropped. Here, resnet50 wins the competition 
4) Next we define hyperparameter ranges and perform hyperparameter tuning on resnet50 model. 
5) We determine the best hyper parameters from the training subjob showing highest validation accuracy. 
6) We take the best hyper parameters from step 5 and retrain resnet50 with sagemaker debugger and profiler options enabled 
7) Next we use the same set of hyperparameters and perform muti-instance training using resnet50 model 
8) Out of all the jobs (in steps 4 to 7), multi-instance training from step 7 provides the best performing model. 
9) We extract the model path in step 7/8 and use it for further deployment 
10) Next we perform single model endpoint deployment, multi-model endpoint deployment, multi-model endpoint deployment using production variants 
11) For multi-model endpoint deployment along with resnet50 model checkpoint, we use efficientnet_b0 model checkpoint. Multi-model checkpoint is used to save cost by serving multiple models out of one container for inference. Inference request will contain the model name it wants to infer against, container will load corresponding model and perform inference. 
12) For multi-model endpoint deployment using production variants, we can use same model checkpoint or different checkpoints or different model checkpoint. This setup can be used for canary deployment or A/B testing scenario. Variants can be added and taken out in real time without causing any downtime. 
13) For production variant type endpoint, we initially configure both variants with equal weightage which means both variants will receive the inference requests in equal proportions. Next, we turn the weightage of variant1 to 0 and that of variant2 to 100% allowing all the traffic to go to only variant2 in real time. Next, we configure autoscaling on variant2 and watch the number of instances going up to 2 on account of increased traffic. 
14) For all types of endpoints configured above we make sure to test the endpoint by sending inference traffic to corresponding endpoint. 
15) Next, we configured a sagemaker pipeline that will carry out training, evaluation, model registration, model creation in an automated fashion. Next, we approve the model and deploy it and test out the endpoint by sending inference traffic to it. 
16) For final testing, we configure a function to bring up endpoint (using the multi-instance resnet50 checkpoint). Next, we create a lambda function that will be used to invoke endpoint. Next, we integrate AWS API Gateway with lambda to perform inference on trained model. We create local function which is used to connect to AWS API Gateway URL and use it to perform inference. Next, we generate test case and use it along with local function to perform inference. Further, we create  gradio interface that is used in conjunction with local function to perform inference. 

Inference process: 
1. Start → Bring up the endpoint 
2. Generate Test Case 
3. Bring Up Gradio Interface 
4. Fill in Information 
• Inference URL 
• Bucket 
• Image File Path 
5. Gradio Invokes Local Function 
• Connects to Inference URL 
6. Inference URL Invokes Lambda Function 
7. Lambda Function Invokes Endpoint 
8. Endpoint Model Performs Inference 
• Sends Softmax Array to Lambda Function 
9. Lambda Function Sends Inferred Information Back 
• Through API Gateway to Local Function 
10.  Local Function Performs Argmax on Returned Array 
11. Function Sends Inferred Label & Original Label to Gradio Display 
12. End
## Standout Suggestions
Loads of it.
