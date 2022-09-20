import json
import logging
import sys
import os
import tarfile
import pandas as pd
import numpy as np
import traceback
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

tf.get_logger().setLevel('ERROR')

prefix = '/opt/ml/processing/'
# Sagemaker stores the dataset copied from S3
input_path = os.path.join(prefix, 'input')
# If something bad happens, write a failure file with the error messages and store here
output_path = os.path.join(prefix, 'output')
evaluation_path =  os.path.join(output_path, 'evaluation')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def load_model():
    logger.info("Load Pre-Trained Model")
    model_path = os.path.join(input_path, "model/model.tar.gz")
    with tarfile.open(model_path) as tar_file:
        tar_file.extractall(".")
    model = tf.keras.models.load_model("model.h5")
    model.compile(optimizer="adam", loss="mse")
    return model

if __name__ == "__main__":
    logger.info("Evaluation mode ...")
    
    try:
        test_path = os.path.join(input_path, "testing")
    
        # Load 'h5' keras model
        model = load_model()

        # Specify the Column names in order to manipulate the specific columns for pre-processing
        column_names = ["rings", "length", "diameter", "height", "whole weight", 
            "shucked weight", "viscera weight", "shell weight", "sex_F", "sex_I", "sex_M"]
        
        logger.info("Reading test data.")
        test_data = pd.read_csv(os.path.join(test_path, 'test.csv'), sep=',', names=column_names)
        y_test = test_data['rings'].to_numpy()
        x_test = test_data.drop(['rings'], axis=1).to_numpy()
        x_test = preprocessing.normalize(x_test)
        
        #run predictions
        predictions_ = model.predict(x_test)
        
        # Calculate the metrics
        mse = mean_squared_error(y_test, predictions_)
        rmse = mean_squared_error(y_test, predictions_, squared=False)
        std = np.std(np.array(y_test) - np.array(predictions_))
        # Save Metrics to S3 for Model Package
        logger.info("Root Mean Square Error: {}".format(rmse))
        logger.info("Mean Square Error: {}".format(mse))
        logger.info("Standard Deviation: {}".format(std))
        report_dict = {
            "regression_metrics": {
                'rmse': {
                    'value': rmse
                },
                'mse': {
                    'value': mse,
                },
                'standard_deviation': {
                    'value': std,
                },
            },
        }
        

        logger.info("Writing out evaluation report with mse: %f and std: %f", mse, std)
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        evaluation_file = f"{evaluation_path}/evaluation.json"
        with open(evaluation_file, "w") as f:
            f.write(json.dumps(report_dict))
        
        logger.info("Model evaluation completed.")
            
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # `DescribeTrainingJob` result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\\n' + trc)
            
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\\n' + trc, file=sys.stderr)
        
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)    