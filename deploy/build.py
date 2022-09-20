import argparse
import json
import logging
import os
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
cp = boto3.client('codepipeline')
sm = boto3.client('sagemaker')
ssm = boto3.client('ssm')
deployment_stage = os.environ['STAGE']


def extend_qa_params(args, stage_config):
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
    params = {
        "ModelName": os.environ["MODEL_NAME"],
        "ModelPackageName": model_package_name
    }
    logger.info(params)
    return {
        "Parameters": {**stage_config["Parameters"], **params}
    }


def extend_prd_params(args, stage_config):
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
    params = {
        "ModelPackageName": model_package_name,
        "ModelName": os.environ["MODEL_NAME"]
    }
    return {
        "Parameters": {**stage_config["Parameters"], **params}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-name", type=str, default=os.environ["PIPELINE_NAME"])
    parser.add_argument("--import-config", type=str, default=os.environ["CODEBUILD_SRC_DIR"]+"/{}/{}-config.json".format(deployment_stage, deployment_stage))
    parser.add_argument("--export-config", type=str, default=os.environ["CODEBUILD_SRC_DIR"]+"/{}/{}-config-export.json".format(deployment_stage, deployment_stage))
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=os.environ.get("LOGLEVEL", "INFO").upper())
    model_package_parameter = ssm.get_parameter(Name=os.environ["MODEL_PACKAGE_NAME"], WithDecryption=True)
    model_package_name = model_package_parameter['Parameter']['Value']
    logger.info(model_package_name)
    
    if deployment_stage == 'QA':
        # Write the `QA` stage config
        with open(args.import_config, "r") as f:
            config = extend_qa_params(args, json.load(f))
        logger.info("Config: {}".format(json.dumps(config, indent=4)))
        with open(args.export_config, "w") as f:
            json.dump(config, f, indent=4)
    elif deployment_stage == 'Prd':
        # Write the `Prd` stage config
        with open(args.import_config, "r") as f:
            config = extend_prd_params(args, json.load(f))
        logger.debug("Config: {}".format(json.dumps(config, indent=4)))
        with open(args.export_config, "w") as f:
            json.dump(config, f, indent=4)
    else:
        error_message = "'STAGE' Environment Variable not configured."
        logger.error(error_message)
        raise Exception(error_message)