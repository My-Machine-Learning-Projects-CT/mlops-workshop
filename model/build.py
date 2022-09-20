import argparse
import json
import logging
import os
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
cp = boto3.client('codepipeline')
sm = boto3.client('sagemaker')
l = boto3.client('lambda')
deployment_stage = os.environ['STAGE']


def create_model_package(model_name, package_group_name, container_registry_uri, pipeline_bucket, image_tag, job_id):
    image_uri = "{}:{}".format(container_registry_uri, args.image_tag)
    TrainingJobName = "mlops-{}-{}".format(model_name, job_id)
    model_uri = os.path.join('s3://', pipeline_bucket, job_id, TrainingJobName, 'output/model.tar.gz')
    evaluation_uri = os.path.join('s3://', pipeline_bucket, job_id, 'output/evaluation/evaluation.json')
    
    # Create request payload
    request = {
        "InferenceSpecification": { 
            "Containers": [ 
                { 
                    "Image": image_uri,
                    "ModelDataUrl": model_uri
                }
            ],
            "SupportedContentTypes": [ 
                "text/csv" 
            ],
            "SupportedRealtimeInferenceInstanceTypes": [ 
                "ml.t2.large",
                "ml.c5.large",
                "ml.c5.xlarge"
            ],
            "SupportedResponseMIMETypes": [ 
                "text/csv" 
            ],
            "SupportedTransformInstanceTypes": [ 
                "ml.c5.xlarge"
            ]
        },
        "MetadataProperties": { 
            "ProjectId": str(job_id)
        },
        "ModelApprovalStatus": "Approved",
        "ModelMetrics": {
            "ModelQuality": { 
                "Statistics": { 
                    "ContentType": "application/json",
                    "S3Uri": evaluation_uri
                }
            }
        },
        "ModelPackageDescription": "Abalone Production Model",
        "ModelPackageGroupName": package_group_name
    }
    
    model_package=sm.create_model_package(**request)
    print(model_package)
    return model_package

def get_job_id(env, pipeline_name):
    """
    Description:
    -----------
    Gets the latest ExecutionID based on the Codepipeline Stage.

    :env: (str) The current Deployment Stage.
    :pipeline_name: (str) CodePipeline name.

    :return: CodePipeline Execition ID.
    """
    try:
        response = cp.get_pipeline_state(name=pipeline_name)
        for stageState in response['stageStates']:
            if stageState['stageName'] == "DeployDev":
                for actionState in stageState['actionStates']:
                    if actionState['actionName'] == "BuildDevDeployment":
                        return stageState['latestExecution']['pipelineExecutionId']
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


def extend_dev_params(args, stage_config):
    """
    Description:
    -----------
    Extend the stage configuration with additional parameters specifc to the pipeline execution.

    :args: (parser) Parsed known arguments.
    :stage_config: (dict) Current configuration for the stage.

    :return: (dict) Configured CloudFormation parmaters for the stage.
    """
    # Verify that config has parameters
    if not "Parameters" in stage_config:
        raise Exception("Configuration file must include Parameters")
    job_id=get_job_id(stage_config, args.pipeline_name)
    response = create_model_package(args.model_name,
                                                args.model_package_group_name,
                                                args.container_registry_uri,
                                                args.pipeline_bucket,
                                                args.image_tag,
                                                job_id
                                                )
    model_package_arn = response["ModelPackageArn"]
    params = {
        "ModelName": args.model_name,
        "ModelPackageName": model_package_arn
    }
    return {
        "Parameters": {**stage_config["Parameters"], **params}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-name", type=str, default=os.environ["PIPELINE_NAME"])
    parser.add_argument("--image-tag", type=str, default=os.environ["IMAGE_TAG"])
    parser.add_argument("--model-name", type=str, default=os.environ["MODEL_NAME"])
    parser.add_argument("--model-package-group-name", type=str, default=os.environ['MODEL_GROUP'])
    parser.add_argument("--container-registry-uri", type=str, default=os.environ['CONTAINER_REGISTRY_URI'])
    parser.add_argument("--pipeline-bucket", type=str, default=os.environ['PIPELINE_BUCKET'])
    parser.add_argument("--import-config", type=str, default=os.environ["CODEBUILD_SRC_DIR"]+"/assets/{}/{}-config.json".format(deployment_stage, deployment_stage))
    parser.add_argument("--export-config", type=str, default=os.environ["CODEBUILD_SRC_DIR"]+"/assets/{}/{}-config-export.json".format(deployment_stage, deployment_stage))
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=os.environ.get("LOGLEVEL", "INFO").upper())

    if deployment_stage == 'Dev':
        # Write the `Dev` stage config
        with open(args.import_config, "r") as f:
            config = extend_dev_params(args, json.load(f))
        logger.debug("Config: {}".format(json.dumps(config, indent=4)))
        with open(args.export_config, "w") as f:
            json.dump(config, f, indent=4)
    else:
        error_message = "'STAGE' Environment Variable not configured."
        logger.error(error_message)
        raise Exception(error_message)