
import json
import base64
import boto3

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-09-17-15-13-46-235"
            

def lambda_handler(event, context):
    
    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])
    
    # Instantiate a Predictor
    #predictor = Predictor(ENDPOINT) -old code
    runtime = boto3.Session().client('sagemaker-runtime') #newcode


    # For this model the IdentitySerializer needs to be "image/png"
    #predictor.serializer = IdentitySerializer('image/png') #old code
    
    # Make a prediction:
    #inferences = predictor.predict(image) #old code
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='image/png', Body=image)
    
    # We return the data back to the Step Function    
    #event['inferences'] = inferences.decode('utf-8') #old code
    result = json.loads(response['Body'].read().decode()) #new code
    event['inferences'] = result
    return {
        "statusCode": 200,
        "body": {
            "image_data": event['body']['image_data'],
            "s3_bucket": event['body']['s3_bucket'],
            "s3_key": event['body']['s3_key'],
            "inferences": event['inferences'],
        }
    }

