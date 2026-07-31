

import json
import boto3

ENDPOINT = "pytorch-training-260721-0257-002-b3aa61fc-pt-1784605331"

def lambda_handler(event, context=None):
    # Get the s3 address from the Step Function event input
    review = event['review']
    runtime = boto3.Session().client('sagemaker-runtime')
    payload = {"features": [review]}
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, 
                                       ContentType='application/jsonlines', 
                                       Body=json.dumps(payload)
                                      )
    # Parse and decode the streaming body response
    response_body = response["Body"].read().decode("utf-8")
    result = json.loads(response_body)
    print(type(result), result)
    return {
        'statusCode': 200,
        'body' : json.dumps(result)
        }
