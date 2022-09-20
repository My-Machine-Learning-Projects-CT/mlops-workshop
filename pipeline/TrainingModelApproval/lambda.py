import boto3
import io
import os
import logging
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import json

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')
cw = boto3.client('events')
cp = boto3.client('codepipeline')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    logger.debug("## Environment Variables ##")
    logger.debug(os.environ)
    logger.debug("## Event ##")
    logger.debug(event)
    pipeline_name = os.environ['PIPELINE_NAME']
    model_name = os.environ['MODEL_NAME']
    threshold = float(os.environ['MODEL_BASELINE_QUALITY_THRESHOLD'])
    result = None
    token = None
    s3Output = None
    try:
        response = cp.get_pipeline_state(name=pipeline_name)
        for stageState in response['stageStates']:
            if stageState['stageName'] == 'Evaluate':
                for actionState in stageState['actionStates']:
                    if actionState['actionName'] == 'ApproveModel':
                        latestExecution = actionState['latestExecution']
                        executionId = stageState['latestExecution']['pipelineExecutionId']
                        if latestExecution['status'] != 'InProgress':
                            raise(Exception("Model is not awaiting approval: {}".format(latestExecution['status'])))
                        token = latestExecution['token']
        if token is None:
            raise(Exception("Action token wasn't found. Aborting..."))
        response = sm.describe_processing_job(
            ProcessingJobName="mlops-{}-{}".format(model_name, executionId)
        )
        status = response['ProcessingJobStatus']
        logger.info(status)
        if status == "Completed":
            for output in response['ProcessingOutputConfig']['Outputs']:
                if output['OutputName'] == "evaluation":
                    s3Output = urlparse(output['S3Output']['S3Uri'], allow_fragments=False)
                    pipeline_bucket = s3Output.netloc
                    key = os.path.join(s3Output.path.lstrip('/'), 'evaluation.json')
                    obj = s3.get_object(Bucket=pipeline_bucket, Key=key)
                    evaluation = json.loads(obj["Body"].read().decode('ascii'))
                    rmse = evaluation['regression_metrics']['rmse']['value']
                    if rmse <= threshold:
                        result = {
                            'summary': f'Model trained successfully, rmse: {rmse}',
                            'status': 'Approved'
                        }
                else:
                    result = {
                            'summary': f'Model Quality does not meet threshold, rmse: {rmse}, baseline: {threshold}',
                            'status': 'Rejected'
                        }
        elif status == "InProgress":
            return "Processing Job ({}) in progress".format(executionId)
    except Exception as e:
        result = {
            'summary': str(e),
            'status': 'Rejected'
        }
    
    try:
        response = cp.put_approval_result(
            pipelineName=pipeline_name,
            stageName='Evaluate',
            actionName='ApproveModel',
            result=result,token=token
        )
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

    try:
        response = cw.disable_rule(Name="training-model-approval-{}".format(model_name))
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
    
    return "Done!"