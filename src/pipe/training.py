import sys

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

logger = get_logger('training')

def main():
    '''
    This function Initialized the training pipeline
    '''

    logger.info('Training Pipeline Initialized')
    
    data_ingestion = DataIngestion()
    train_generator, validation_generator = data_ingestion.image_generator_object()

    model_trainer = ModelTrainer()
    model_path = model_trainer.start_training(train_generator, validation_generator)

    model_eval = ModelEvaluation()
    model_eval.evaluate(model_path, train_generator, validation_generator)

if __name__ == '__main__':
    main()