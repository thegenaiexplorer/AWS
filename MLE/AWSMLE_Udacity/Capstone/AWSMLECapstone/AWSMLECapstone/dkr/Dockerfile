# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0-gpu-py310
ENV PATH="/opt/ml/code:${PATH}"
# this environment variable is used by the SageMaker PyTorch container to  determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
# /opt/ml and all subdirectories are utilized by SageMaker, use the /code  subdirectory to store your user code.
COPY train.py /opt/ml/code/train.py
COPY infer_eff.py /opt/ml/code/infer_eff.py
COPY infer_resnet50.py /opt/ml/code/infer_resnet50.py
# Defines train_model.py as script entrypoint 
ENV SAGEMAKER_PROGRAM train.py
