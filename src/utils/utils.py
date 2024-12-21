import os
from signal import raise_signal
import sys
from webbrowser import get

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from tensorflow.keras.models import load_model

logger = get_logger('utils')

def save_model(model, path):
    '''
    This function takes the model and path and saves the model to path

    Parameters:
        model : trained model.
        path : path where the model have to be saved.

    Returns:
        None
    '''

    try:
        logger.info('Saving our model')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        model.save(path)

        logger.info('Model Saved Successfully')

    except Exception as e:
        raise CustomException(e, sys)

def load_dl_model(file_path):
    '''
    This functions takes the file path of the model and returns the loaded model from the path.

    Parameters:
        file_path : Saved model file path.

    Returns:
        Loaded Model.
    '''

    try:
        logger.info('Loading model from path')

        if file_path:
            model = load_model(file_path)

        logger.info('Model loaded successfully')

        return model

    except Exception as e:
        raise CustomException(e, sys)