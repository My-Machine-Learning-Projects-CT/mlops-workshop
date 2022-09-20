import os
import io
import json
import logging
import boto3
import time
import botocore
import numpy as np
import pandas as pd
from sklearn import preprocessing
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client("s3")
smr = boto3.client("sagemaker-runtime")
cp = boto3.client("codepipeline")
cf = boto3.resource("cloudformation")
sm = boto3.client("sagemaker")

def test_endpoint(bucket, key, endpoint_name):
    """
    Description:
    ------------
    Executes model predictions on the testing dataset.
    
    :bucket: (str) Pipeline S3 Bucket.
    :key: (str) Path to "testing" dataset.
    :endpoint_name: (str) Name of the 'Dev' endpoint to test.

    :returns: Lists of ground truth, prediction labels and response times.
    
    """
    column_names = ["rings", "length", "diameter", "height", "whole weight", "shucked weight",
                    "viscera weight", "shell weight", "sex_F", "sex_I", "sex_M"]
    response_times = []
    predictions = []
    y_test = []
    obj = s3.get_object(Bucket=bucket, Key=key)
    test_df = pd.read_csv(io.BytesIO(obj['Body'].read()), names=column_names)
    y = test_df['rings'].to_numpy()
    X = test_df.drop(['rings'], axis=1).to_numpy()
    X = preprocessing.normalize(X)
    
    # Cycle through each row of the data to get a prediction
    for row in range(len(X)):
        payload = ",".join(map(str, X[row]))
        elapsed_time = time.time()
        try:
            response = smr.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType = "text/csv",
                Body=payload
            )
        except ClientError as e:
            error_message = e.response["Error"]["Message"]
            logger.error(error_message)
            raise Exception(error_message)
        response_times.append(time.time() - elapsed_time)
        result = np.asarray(response['Body'].read().decode('utf-8').rstrip('\n'))
        predictions.append(float(result))
        y_test.append(float(y[row]))
    
    return y_test, predictions, response_times


def handler(event, context):
    logger.debug("## Environment Variables ##")
    logger.debug(os.environ)
    logger.debug("## Event ##")
    logger.debug(event)

    try:
        # Get Input Variables    
        pipeline_name = os.environ['PIPELINE_NAME']
        bucket = os.environ['PIPELINE_BUCKET']
        endpoint_name = os.environ['ENDPOINT']
        test_data = os.environ['TEST_DATA']
        jobId = event['CodePipeline.job']['id']
        key = None
        executionId = None 

        response = cp.get_pipeline_state(name=pipeline_name)
        for stageState in response['stageStates']:
            if stageState['stageName'] == 'PipelineExecution':
                for actionState in stageState['actionStates']:
                    if actionState['actionName'] == 'SubmitPipeline':
                        executionId = stageState['latestExecution']['pipelineExecutionId']
        actionExecutionDetails = cp.list_action_executions(
                                pipelineName=pipeline_name,
                                filter={
                                        'pipelineExecutionId': executionId
                                        },
                                )
        for actionDetail in actionExecutionDetails['actionExecutionDetails']:
            if actionDetail['actionName'] == 'SubmitPipeline':
                pipelineExecutionSteps = sm.list_pipeline_execution_steps(
                    PipelineExecutionArn=actionDetail['output']['outputVariables']['PipelineExecutionArn']
                )
                for pipelineExecutionStep in pipelineExecutionSteps['PipelineExecutionSteps']:
                    if pipelineExecutionStep['StepName'] == 'ETL':
                        processingJobArn = pipelineExecutionStep['Metadata']['ProcessingJob']['Arn']
                        processingJobName = parse_arn(processingJobArn)['resource']
                        processingJob = sm.describe_processing_job(
                                        ProcessingJobName=processingJobName)
                        for output in processingJob['ProcessingOutputConfig']['Outputs']:
                            if output['OutputName'] == 'test':
                                bucket, key = output['S3Output']['S3Uri'].replace("s3://", "").split("/", 1)
        logger.info(key)
        file_name = os.path.join(key, test_data)
        # Get the evaluation results from SageMaker hosted model
        logger.info("Evaluating SageMaker Endpoint ...")
        times = test_endpoint(bucket, file_name, endpoint_name)

        # Save Metrics to S3 for Model Package
        logger.info("Average Endpoint Response Time: {:.2f}s".format(np.mean(times)))
        cp.put_job_success_result(jobId=jobId)
    except Exception as e:
        logger.error(e)
        cp.put_job_failure_result(
            jobId=jobId,
            failureDetails={
                'type': 'ConfigurationError',
                'message': str(e),
                'externalExecutionId': context.aws_request_id
            }
        )
    
    return {
        "statusCode": 200,
        "AvgResponseTime": "{:.2f} seconds".format(np.mean(times))
    }
    
def parse_arn(arn):
    elements = arn.split(':', 6)
    result = {
        'arn': elements[0],
        'partition': elements[1],
        'service': elements[2],
        'region': elements[3],
        'account': elements[4],
        'resource': elements[5],
        'resource_type': None
    }
    if '/' in result['resource']:
        result['resource_type'], result['resource'] = result['resource'].split('/',1)
    elif ':' in result['resource']:
        result['resource_type'], result['resource'] = result['resource'].split(':',1)
    return result