import boto3
import io
import zipfile
import json
import os
import logging

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')
cw = boto3.client('events')
cp = boto3.client('codepipeline')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    logger.debug("## Environment Variables ##")
    logger.debug(os.environ)
    region = os.environ['AWS_REGION']
    logger.debug("## Event ##")
    logger.debug(event)
    evaluationJob = None
    pipeline_name = os.environ['PIPELINE_NAME']
    model_name = os.environ['MODEL_NAME']
    jobId = event['CodePipeline.job']['id']
    accountId = event['CodePipeline.job']['accountId']
    pipeline_bucket = os.environ['PIPELINE_BUCKET']
    try:
        response = cp.get_pipeline_state(name=pipeline_name)
        for stageState in response['stageStates']:
            if stageState['stageName'] == 'Evaluate':
                for actionState in stageState['actionStates']:
                    if actionState['actionName'] == 'EvaluateModel':
                        executionId = stageState['latestExecution']['pipelineExecutionId']
        logger.info("Start evaluation processing job for 'jobid[{}]' and 'executionId[{}]'".format(jobId, executionId))
        for inputArtifacts in event["CodePipeline.job"]["data"]["inputArtifacts"]:
            if inputArtifacts['name'] == 'ModelSourceOutput':
                s3Location = inputArtifacts['location']['s3Location']
                zip_bytes = s3.get_object(Bucket=s3Location['bucketName'], Key=s3Location['objectKey'])['Body'].read()
                with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
                   evaluationJob = json.loads(z.read('evaluationJob.json').decode('ascii'))
        if evaluationJob is None:
            raise(Exception("'evaluationJob.json' not found"))
        TrainingJobName="mlops-{}-{}".format(model_name, executionId)
        evaluationJob['ProcessingJobName'] = "mlops-{}-{}".format(model_name, executionId)
        evaluationJob['AppSpecification']['ImageUri'] = "".join([accountId, ".dkr.ecr.",region, ".amazonaws.com/", model_name,":latest"])
        evaluationJob['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri'] = os.path.join('s3://', pipeline_bucket, executionId,'output/evaluation')
        evaluationJob['ProcessingInputs'][0]['S3Input']['S3Uri'] = os.path.join('s3://', pipeline_bucket, executionId, 'input/testing')
        evaluationJob['ProcessingInputs'][1]['S3Input']['S3Uri'] = os.path.join('s3://', pipeline_bucket, executionId, TrainingJobName, "output/model.tar.gz")
        evaluationJob['Tags'].append({'Key': 'jobid', 'Value': jobId})
        sm.create_processing_job(**evaluationJob)
        logger.info(evaluationJob)
        cw.enable_rule(Name="training-model-approval-{}".format(model_name))
        cp.put_job_success_result(jobId=jobId)
    except Exception as e:
        logger.error(e)
        response = cp.put_job_failure_result(
            jobId=jobId,
            failureDetails={
                'type': 'ConfigurationError',
                'message': str(e),
                'externalExecutionId': context.aws_request_id
            }
        )
    
    return 'Done'