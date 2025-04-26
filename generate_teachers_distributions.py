import argparse
# General
import operator
import os
from functools import reduce

import numpy as np
from scipy.special import softmax
# Transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer
)

from src.data_handling.DataHandlers import NERDataHandler
from src.utils import (
    initialize_dataclasses,
    setup_logging,
    initialize_tokenizer
)

"""
  input 
    prob: matrice NxMAX_LENx(3k) organizzata come [B1,I1,O1,B2,I2,O2,...,Bk,Ik,Ok]
  output
    matrice NxMAX_LENx(2k+1) organizzata come [B1,I1,B2,I2,...,Bk,Ik,O]
"""

def aggregate_proba(prob):
  k = int(prob.shape[-1] / 3)
  B_ids = np.array(range(0,3*k,3))
  I_ids = np.array(range(1,3*k,3))
  O_ids = np.array(range(2,3*k,3))

  result = np.empty((prob.shape[0], prob.shape[1], 2*k+1))
  for entity in range(k):
    result[:,:,entity*2] = np.prod(
        np.array([
            prob[:,:,B_ids[entity]],
            np.prod(
                np.array([np.sum(prob[:,:,reduce(operator.concat, [[I_ids[i]], [O_ids[i]]])], axis = -1) for i in range(len(O_ids)) if i != entity]),
                axis = 0
            )
        ]), axis = 0
    )
    result[:,:,entity*2+1] = np.prod(
        np.array([
            prob[:,:,I_ids[entity]],
            np.prod(
                np.array([np.sum(prob[:,:,reduce(operator.concat, [[B_ids[i]], [O_ids[i]]])], axis = -1) for i in range(len(O_ids)) if i != entity]),
                axis = 0
            )
        ]), axis = 0
    )

  result[:,:,-1] = np.prod(prob[:,:,O_ids], axis=-1)
  result = softmax(result, axis=-1)
  return result

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--teachers_dir', type=str, default='models/Teachers')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--model_name_or_path', type=str, default='models/dmis-labbiobert-base-cased-v1.2')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--logging_dir', type=str, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_predict', type=bool, default=True)
    parser.add_argument('--evaluation_strategy', type=str, default='epoch')
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)


    return parser.parse_args()

def main():

    args = parse_arguments()
    print(args)

    # Convert arguments to appropriate dataclasses
    model_args, data_args, training_args = initialize_dataclasses(args)

    # Set up logging
    logger = setup_logging(training_args)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(model_args)

    predictions = {} # it will be filled with teachers' predictions
    teachers_folders = os.listdir(args.teachers_dir)

    for teacher in teachers_folders:
        print("Obtaining predictions of teacher: ", teacher)

        global_data_handler = NERDataHandler(tokenizer)
        labels = global_data_handler.get_labels(data_args.labels)

        config = AutoConfig.from_pretrained(
            os.path.join(args.teachers_dir, teacher),
            num_labels=len(labels),
            id2label={i: label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=model_args.cache_dir,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            os.path.join(args.teachers_dir, teacher),
            from_tf=bool(".ckpt" in os.path.join(args.teachers_dir, teacher)),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        trainer = Trainer(
            model=model,
        )
        # setting test dataset
        global_data_handler.set_dataset(
            data_dir=os.path.join(data_args.data_dir, 'GLOBAL', teacher),
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode="train_dev",
            save_cache=False
        )
        predictions[teacher], _, _ = trainer.predict(global_data_handler.datasets['train_dev'])
        predictions[teacher] = softmax(predictions[teacher], axis = 2)

    print("Aggregating distributions...")
    teachers_predictions = np.concatenate([predictions[teacher] for teacher in teachers_folders], axis = -1)
    teachers_predictions = aggregate_proba(teachers_predictions)

    np.save(os.path.join(args.data_dir, 'GLOBAL', 'Student', 'teachers_predictions.npy'), teachers_predictions)

if __name__ == '__main__':
    main()