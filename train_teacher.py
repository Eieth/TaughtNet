
"""
Named Entity Recognition (NER) training script.
This script trains a transformer-based model for token classification tasks.
"""

import argparse
import os

# Transformer imports
from transformers import Trainer

# Local imports
from src.data_handling.DataHandlers import NERDataHandler
from src.utils import(
    compute_metrics,
    initialize_tokenizer,
    initialize_model,
    setup_logging,
    prepare_training_datasets, 
    prepare_test_datasets,
    initialize_dataclasses
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train a NER model")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing training data')

    # Hardware arguments
    parser.add_argument('--device', type=int, default='0', help='CUDA device number')

    # Model arguments
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length for input texts')
    parser.add_argument('--model_name_or_path', type=str, default='dmis-lab/biobert-v1.1', help='Path to pretrained model or model identifier')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save model outputs')
    parser.add_argument('--logging_dir', type=str, default=None, help='Directory to save training logs')

    # Training arguments
    parser.add_argument('--num_train_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, help='Batch size per device during training')
    parser.add_argument('--logging_steps', type=int, default=500, help='Log training metrics every X steps')
    parser.add_argument('--do_train', type=bool, default=True, help='Whether to run training')
    parser.add_argument('--do_eval', type=bool, default=True, help='Whether to run evaluation')
    parser.add_argument('--do_predict', type=bool, default=True, help='Whether to run predictions')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch', help='Evaluation strategy during training')
    parser.add_argument('--save_steps', type=int, default=10000, help='Save checkpoint every X steps')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization')

    return parser.parse_args()



def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_arguments()
    print(args)

    # Convert arguments to appropriate dataclasses
    model_args, data_args, training_args = initialize_dataclasses(args)

    # Set up logging
    logger = setup_logging(training_args)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(model_args)

    # Initialize data handler and get labels
    data_handler = NERDataHandler(tokenizer)
    labels = data_handler.get_labels(data_args.labels)

    # Initialize model
    model, config = initialize_model(model_args, len(labels), labels)

    # Prepare datasets
    prepare_training_datasets(data_handler, data_args, config)
    prepare_test_datasets(data_handler, data_args, config)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_handler.datasets['train'],
        eval_dataset=data_handler.datasets['dev'],
        compute_metrics=compute_metrics,
        callbacks=None
    )

    # Train the model
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,
            resume_from_checkpoint=None
        )

    # Save the model and tokenizer
    if training_args.do_train and training_args.should_save:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluate the model
    if training_args.do_eval:
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")

    # Run predictions
    if training_args.do_predict:
        predictions = trainer.predict(data_handler.datasets['test'])
        logger.info(f"Prediction results: {predictions.metrics}")

if __name__ == '__main__':
    main()