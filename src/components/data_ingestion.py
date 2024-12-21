import os
import sys

from dataclasses import dataclass
from logging import getLogger

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = get_logger('data-ingestion')

@dataclass
class DataPaths:
    data_path: str = os.path.join('dataset', 'Village-dataset')

class DataIngestion:
    def __init__(self):
        self.data_paths = DataPaths()

    def data_augmentation_object(self):
        '''
        Returns the Training and Testing Data Augmentation Objects

        Returns:
            Training and Testing ImageDataGenerator Objects
        '''

        try:
            logger.info('Started Creating ImageDataGenerator Object for train and test')

            logger.info('Training ImageDataGenerator Object')
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2]
            )

            logger.info('Testing ImageDataGenerator Object')
            test_datagen = ImageDataGenerator(rescale=1./255)

            logger.info('Data Augmenation Objects Created')

            return (
                train_datagen,
                test_datagen
            )

        except Exception as e:
            raise CustomException(e, sys)

    def image_generator_object(self):
        '''
        This method will take Train and Test ImageDataGenerator Object and returns the Generator objects.

        Returns:
            Train and Test Generator Objects
        '''

        try:
            logger.info('Generator Object Creating Started')

            train_datagen, test_datagen = self.data_augmentation_object()

            logger.info('Training Generator Object Creating')
            train_generator = train_datagen.flow_from_directory(
                self.data_paths.data_path,
                target_size=(256,256),
                batch_size = 32,
                class_mode = 'categorical'
            )

            logger.info('Testing Generator Object Creating')
            test_generator = test_datagen.flow_from_directory(
                self.data_paths.data_path,
                target_size=(256,256),
                batch_size = 32,
                class_mode = 'categorical'
            )

            logger.info('Generator Objects Created successfully')

            return (
                train_generator,
                test_generator
            )

        except Exception as e:
            raise CustomException(e, sys)