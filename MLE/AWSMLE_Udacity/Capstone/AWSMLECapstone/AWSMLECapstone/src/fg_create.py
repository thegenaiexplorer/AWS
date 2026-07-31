import argparse
import time
from time import sleep
import pandas as pd
from sklearn.model_selection import train_test_split
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
import boto3
import os
import json
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


region = os.environ['AWS_DEFAULT_REGION']
sts = boto3.Session(region_name=region).client(service_name='sts', region_name=region)
iam = boto3.Session(region_name=region).client(service_name='iam', region_name=region)
featurestore_runtime = boto3.Session(region_name=region).client(service_name='sagemaker-featurestore-runtime',
                                                                region_name=region)
sm = boto3.Session(region_name=region).client(service_name='sagemaker',
                                              region_name=region)
caller_identity = sts.get_caller_identity()
assumed_role_arn = caller_identity['Arn']
assumed_role_name = assumed_role_arn.split('/')[-2]
get_role_response = iam.get_role(RoleName=assumed_role_name) 
role = get_role_response['Role']['Arn']
bucket = sagemaker.Session().default_bucket()
sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region),
                                      sagemaker_client=sm,
                                      sagemaker_featurestore_runtime_client=featurestore_runtime)


def parse_args():
    parser = argparse.ArgumentParser(description="FeatureGroupArgs")

    parser.add_argument('--input_data', type=str,
                        default='/opt/ml/processing/input/data')
    parser.add_argument('--output_data', type=str,
                        default='/opt/ml/processing/output'
                       )
    parser.add_argument('--validation_split', type=float,
                        default=0.1)
    parser.add_argument('--test_split', type=float,
                        default=0.5)
    parser.add_argument('--feature_group_name', type=str,
                        default="test_fg")
    parser.add_argument('--inp_file_name', type=str,
                        default=None)
    parser.add_argument('--feature-store-offline-prefix', type=str,
                        default=None) 
    return parser.parse_args()


def read_json(jfile):
    s3_client = boto3.client('s3')
    with open(jfile, 'r') as f: # <'file_list.json'>
        d=json.load(f)
    logger.info(f'Key Labels in json file are {d.keys()}')
    ds_dict = {k: [] for k in d.keys()}
    for k, v in d.items():
        f_ll = ["s3://aft-vbi-pds/bin-images/" + os.path.basename(p).split(".")[0]+ ".jpg" for p in v]
        ds_dict[k] = f_ll
    for k, v in ds_dict.items():
        logger.info(f'Total number items belonging to label {k} is {len(v)}')
    return ds_dict

def populate_dfs(label, ds_d):
    df = pd.DataFrame({'s3_path': ds_d[label]})
    df['label'] = int(label)
    return df
    
    

def load_df(ds_dict):
    df_1 = populate_dfs('1', ds_dict)
    df_2 = populate_dfs('2', ds_dict)
    df_3 = populate_dfs('3', ds_dict)
    df_4 = populate_dfs('4', ds_dict)
    df_5 = populate_dfs('5', ds_dict)
    df = pd.concat([df_1, df_2, df_3, df_4, df_5], axis=0)
    df.reset_index(inplace=True, drop=True)
    print(df.info(), df.head(5))
    return df


def split_ds_type(df, v_s, t_s):
    df['id'] = list(range(len(df)))
    train_df, valid_df = train_test_split(df, test_size=v_s, stratify=df['label'], random_state=42)
    valid_df, test_df = train_test_split(valid_df, test_size=t_s, stratify=valid_df['label'], random_state=42)
    train_df['ds_type'] = 'train'
    valid_df['ds_type'] = 'valid'
    test_df['ds_type'] = 'test'
    train_df.to_csv('/opt/ml/processing/output/train/inv_train.csv')
    valid_df.to_csv('/opt/ml/processing/output/valid/inv_valid.csv')
    test_df.to_csv('/opt/ml/processing/output/test/inv_test.csv')
    return train_df, valid_df, test_df


def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == 'object':
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame


def wait_for_feature_group_creation_complete(feature_group):
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        print('Feature Group status: {}'.format(status))
        while status == "Creating":
            print("Waiting for Feature Group Creation")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            print('Feature Group status: {}'.format(status))
        if status != "Created":
            print('Feature Group status: {}'.format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")
    except:
        print('No feature group created yet.')

def create_ingest_feature_group(fg, train_df, valid_df, test_df):
    
    fgroup = FeatureGroup(name=fg, sagemaker_session=sagemaker_session)
    print('Feature Group: {}'.format(fgroup))    
    try:                
        print('Waiting for existing Feature Group to become available if it is being created by another instance in our cluster...')
        wait_for_feature_group_creation_complete(fgroup)
    except Exception as e:
        print('Before CREATE FG wait exeption: {}'.format(e))

    timest = time.time()
    train_df['EventTime'] = timest
    valid_df['EventTime'] = timest
    test_df['EventTime'] = timest
    col_names = ['id', 'EventTime', 's3_path', 'label', 'ds_type']  
    train_df_ingest = train_df[col_names].copy()
    valid_df_ingest = valid_df[col_names].copy()
    test_df_ingest = test_df[col_names].copy()
    
    fgroup.load_feature_definitions(data_frame=train_df_ingest)
    try:
        print('Creating Feature Group with role {}...'.format(role))
        fgroup.create(
            s3_uri=f"s3://{bucket}/inventory_ds_fg",
            record_identifier_name="id",
            event_time_feature_name="EventTime",
            role_arn=role,
            enable_online_store=False
            )
        print('Creating Feature Group. Completed.')
        print('Waiting for new Feature Group to become available...')
        wait_for_feature_group_creation_complete(fgroup)
        print('Feature Group available.')
        # the information about the Feature Group
        fgroup.describe()
    
    except Exception as e:
        print('Exception: {}'.format(e))


    
    df_fs_train_records = cast_object_to_string(train_df_ingest)
    df_fs_validation_records = cast_object_to_string(valid_df_ingest)
    df_fs_test_records = cast_object_to_string(test_df_ingest)

    print('Ingesting features start...')
    fgroup.ingest(
        data_frame=df_fs_train_records,
        max_workers=2,
        wait=True
        )

    print("Train Ingestion complete")
    print("Ingesting validation ")

    fgroup.ingest(
        data_frame=df_fs_validation_records,
        max_workers=2,
        wait=True
        )

    print("Valid Ingestion complete")
    print("Ingesting test ")

    fgroup.ingest(
        data_frame=df_fs_test_records,
        max_workers=2,
        wait=True
        )
    print("Test Ingestion complete")

    offline_store_status = None
    while offline_store_status != 'Active':
        try:
            offline_store_status = fgroup.describe()['OfflineStoreStatus']['Status']
        except:
            pass
        print('Offline store status: {}'.format(offline_store_status))    
        sleep(15)
    print('...features ingested!')
    
if __name__ == "__main__":
    args = parse_args()
    print('loaded arguments: ')
    print(args)
    #We want to accomplish all below in single function:
    #Json File is populated in S3 location
    #Script runs, picks up the file from S3 location (we can pass this information as part of env variables)
    #Converts the from json format to jpg file location format
    #Stores all the path information in a dataframe
    #splits the dataset into train, validation and test
    #Adds a column to indicate which set it belongs to. We will use this column to fetch information during training
    inp_file = args.input_data + args.inp_file_name
    ds_dict = read_json(inp_file)
    logger.info("Step 1: Data read from json file and loaded into dictionary succesfully")
    df = load_df(ds_dict)
    logger.info("Step 2: Data loaded into data frame successfully")
    train_df, val_df, test_df = split_ds_type(df, args.validation_split, args.test_split)
    logger.info("Step 2: Dataframe split complete")

    fgroup = create_ingest_feature_group(args.feature_group_name,
                                         train_df,
                                         val_df,
                                         test_df)
    logger.info("Woohoooo!! Feature group created successfully")
    logger.info("Congratulations!! Tasks Accomplished")