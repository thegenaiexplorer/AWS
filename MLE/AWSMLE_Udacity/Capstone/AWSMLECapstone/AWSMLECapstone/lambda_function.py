

import json
import boto3

s3 = boto3.client('s3')
ENDPOINT = "test-ep-lambda"

def lambda_handler(event, context):

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.jpeg')

    # We read the data from a file
    with open("/tmp/image.jpeg", "rb") as f:
        image_data = f.read()
        
    runtime = boto3.Session().client('sagemaker-runtime')
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='image/jpeg', Body=image_data)
    infer_list = response['Body'].read()
    infer_list = json.loads(infer_list)
    print(type(infer_list), infer_list)
    
    return {
        'statusCode': 200,
        'body' : json.dumps(infer_list)
        }
