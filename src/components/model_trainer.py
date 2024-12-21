import os
import sys

from dataclasses import dataclass
from turtle import mode
from webbrowser import get

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import save_model

from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

logger = get_logger('model-trainer')

@dataclass
class ModelPaths:
    model_path: str = os.path.join('model', 'model.h5')

class ModelTrainer:
    def __init__(self):
        self.modelpath = ModelPaths()

    def pretrained_model(self):
        '''
        This methods loads and returns the pretrained model

        Returns:
            Pretrained Model
        '''

        try:
            logger.info('Loading the Pretrained Model')

            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(256,256,3))


            logger.info('FineTuning the Model')

            base_model.trainable = True
            for layer in base_model.layers[:100]:  # Freeze first 100 layers
                layer.trainable = False

            logger.info("Model Loaded")

            return base_model

        except Exception as e:
            raise CustomException(e, sys)

    def transfer_learning(self):
        '''
        This method creates the model using Transfer Learning techniques via Pretrained Model.

        Returns:
            Fully Build Transfer Learning Model.
        '''

        try:
            logger.info('Building Our own model with transfer learning')

            base_model = self.pretrained_model()

            model = Sequential()
            model.add(base_model)
            model.add(GlobalAveragePooling2D())
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(3, activation='softmax'))

            logger.info('Model Builded Successfully')

            logger.info('Compiling our model')
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

            logger.info("Model Ready to train")

            return model

        except Exception as e:
            raise CustomException(e, sys)

    def start_training(self, train_generator, validation_generator):
        '''
        This method starts initializing the training of our model via Transfer Learning.

        Parameters:
            train_generator : Training Data Generator Object.
            validation_generator : Validation Data Generator Object.

        Returns:
            Saved model path for further evaluation.
        '''

        try:
            logger.info('Initiating Training for our model')

            model = self.transfer_learning()

            logger.info('Initializing callbacks for our model')
            es_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            logger.info('Model Training Started')
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=20,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                validation_steps=validation_generator.samples // validation_generator.batch_size,
                callbacks=[es_callback]

            )

            logger.info('Model Trained Successfully')

            save_model(
                model=model,
                path=self.modelpath.model_path
            )

            return self.modelpath.model_path

        except Exception as e:
            raise CustomException(e, sys)