import argparse
import os

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import Trainer

from src.utils import (
    initialize_tokenizer,
    initialize_model,
    setup_logging,
    initialize_dataclasses,
    prepare_datasets
)

from src.EvaluateCallback import EvaluateCallback
from src.data_handling.DataHandlers import MultiNERDataHandler
from src.data_handling.DataClasses import Split


# General


def loss(logits, labels, teachers_proba):
    lbd = 0.99
    eps = 1e-8

    # mask_ner = (labels != -100) & (labels != 6)
    # loss_ner = F.nll_loss(torch.log(F.softmax(logits[mask_ner], dim=-1)), labels[mask_ner])
    #
    # mask_kd = (labels != -100)
    # loss_kd = nn.KLDivLoss()(torch.log(F.softmax(logits[mask_kd], dim=-1)), teachers_proba[mask_kd])
    #
    # return (1.0 - lbd) * loss_ner + lbd * loss_kd


    mask_ner = (labels != -100) & (labels != 6)
    mask_kd = (labels != -100)

    # 使用log_softmax替代手动log+softmax
    log_probs = F.log_softmax(logits, dim=-1) + eps  # 防止log(0)

    # 检查有效样本数
    if mask_ner.sum() == 0:
        loss_ner = torch.tensor(0.0, device=logits.device)
    else:
        loss_ner = F.nll_loss(log_probs[mask_ner], labels[mask_ner])

    # 对教师概率做归一化检查
    teachers_proba = teachers_proba.clamp(min=eps, max=1 - eps)  # 避免0和1
    teachers_proba = teachers_proba / teachers_proba.sum(dim=-1, keepdim=True)  # 重新归一化

    loss_kd = F.kl_div(log_probs[mask_kd], teachers_proba[mask_kd], reduction='batchmean')

    return (1 - lbd) * loss_ner + lbd * loss_kd


class KGTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=True, num_items_in_batch=None) -> Tensor:
        labels = inputs.pop("labels")
        teachers_proba = inputs.pop("teachers_proba")
        outputs = model(**inputs)
        logits = outputs['logits']
        # 计算损失
        loss_value = loss(logits.float(), labels, teachers_proba.float())

        # 根据 return_outputs 决定返回值
        return (loss_value, outputs) if return_outputs else loss_value


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/GLOBAL/Student')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--model_name_or_path', type=str, default='dmis-lab/biobert-v1.1')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--logging_dir', type=str, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--do_train', type=bool, default=True, help='Whether to run training')
    parser.add_argument('--do_eval', type=bool, default=True, help='Whether to run evaluation')
    parser.add_argument('--do_predict', type=bool, default=True, help='Whether to run predictions')
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()

def main():
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
    data_handler = MultiNERDataHandler(tokenizer)
    labels = data_handler.get_labels(data_args.labels)

    # Initialize model
    model, config = initialize_model(model_args, len(labels), labels)

    # Prepare datasets
    prepare_datasets(data_handler, data_args, config, Split.train)
    prepare_datasets(data_handler, data_args, config, Split.test)

    # Initialize Trainer
    trainer = KGTrainer(
        model=model,
        args=training_args,
        train_dataset=data_handler.datasets['train'],
        eval_dataset=data_handler.datasets['dev'],
    )

    trainer.add_callback(EvaluateCallback(model_args.model_name_or_path, labels, data_args.data_dir))

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