from linecache import getline
import os
import sys

import mlflow
import dagshub

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import load_dl_model

logger = get_logger('model-evaluation')

class ModelEvaluation:
    @staticmethod
    def evaluate(model_path, train_generator, validation_generator):
        '''
        This method take the model and evaluate it on train and test image data.

        Parameters:
            model_path : Trained saved model path.
            train_generator : training data generator object.
            validationg_generator : validation generator object.

        Returns:
            None
        '''

        try:
            logger.info('Model Evaluation Started')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = load_dl_model(model_path)

            logger.info('Connecting to our DagsHub')

            dagshub.init(repo_owner='anonymous298', repo_name='Potato-Disease-DL-Project', mlflow=True)

            logger.info('Evaluting for train data')
            _, train_accuracy = model.evaluate(train_generator)

            logger.info('Evaluting for test data')
            _, test_accuracy = model.evaluate(validation_generator)

            logger.info('Experiment Tracking')

            mlflow.set_experiment('Potato-Disease-Model')

            with mlflow.start_run():
                mlflow.log_metric('Training Accuracy', train_accuracy)
                mlflow.log_metric('Testing Accuracy', test_accuracy )

                mlflow.tensorflow.log_model(model, 'model', registered_model_name='Potato-Disease-Model')

            logger.info('Everything Completed')

        except Exception as e:
            raise CustomException(e, sys)