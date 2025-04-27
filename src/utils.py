import argparse
import dataclasses
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
# Transformer imports
from transformers import (
	AutoTokenizer,
	EvalPrediction,
	PreTrainedTokenizer,
	AutoConfig,
	AutoModelForTokenClassification,
	TrainingArguments
)
from src.data_handling.DataClasses import Split

from src.arguments import ModelArguments, DataTrainingArguments

def get_label_map():
	return {i: label for i, label in enumerate(['B', 'I', 'O'])}

def student_performance(trainer, teacher_sets):
	predictions, label_ids, _ = trainer.predict(teacher_sets['NCBI-disease'])
	student_to_teacher_map = {0: 2, 1: 2, 2: 2, 3: 2, 4: 0, 5: 1, 6: 2}
	metrics_disease = student_metrics(predictions, label_ids, student_to_teacher_map)
	predictions, label_ids, _ = trainer.predict(teacher_sets['BC5CDR-chem'])
	student_to_teacher_map = {0: 2, 1: 2, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2}
	metrics_drugchem = student_metrics(predictions, label_ids, student_to_teacher_map)
	predictions, label_ids, _ = trainer.predict(teacher_sets['BC2GM'])
	student_to_teacher_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
	metrics_geneprot = student_metrics(predictions, label_ids, student_to_teacher_map)
	return metrics_disease, metrics_drugchem, metrics_geneprot


def process_predictions(predictions: np.ndarray, label_ids: np.ndarray,
						ignore_index: int = -100, label_map: dict = None,
						pred_transform: callable = None) -> tuple[list[list[Any]], list[list[Any]]]:
	"""
	通用函数处理预测结果和标签
	Args:
		predictions: 模型预测结果
		label_ids: 真实标签
		ignore_index: 要忽略的标签索引
		label_map: 标签映射字典
		pred_transform: 对预测结果进行转换的函数
	"""
	if label_map is None:
		label_map = get_label_map()

	preds = np.argmax(predictions, axis=2)
	batch_size, seq_len = preds.shape
	out_label_list = [[] for _ in range(batch_size)]
	preds_list = [[] for _ in range(batch_size)]

	for i in range(batch_size):
		for j in range(seq_len):
			if label_ids[i, j] != ignore_index:
				out_label_list[i].append(label_map[label_ids[i][j]])
				pred = preds[i][j]
				if pred_transform is not None:
					pred = pred_transform(pred)
				preds_list[i].append(label_map[pred])

	return preds_list, out_label_list


def calculate_metrics(label_list: list[list[Any]], pred_list: list[list[Any]]) -> Dict:
	"""计算评估指标"""
	return {
		"accuracy_score": accuracy_score(label_list, pred_list),
		"precision": precision_score(label_list, pred_list),
		"recall": recall_score(label_list, pred_list),
		"f1": f1_score(label_list, pred_list),
	}


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> tuple[list[list[Any]], list[list[Any]]]:
	"""原始对齐预测函数"""
	return process_predictions(
		predictions,
		label_ids,
		ignore_index=nn.CrossEntropyLoss().ignore_index
	)


def compute_metrics(p: EvalPrediction) -> Dict:
	"""原始计算指标函数"""
	preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
	return calculate_metrics(out_label_list, preds_list)


def student_metrics(predictions, label_ids, student_to_teacher_map):
	"""学生模型指标计算函数"""
	preds_list, out_label_list = process_predictions(
		predictions,
		label_ids,
		ignore_index=-100,
		pred_transform=lambda x: student_to_teacher_map[x]
	)
	return calculate_metrics(out_label_list, preds_list)


def initialize_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizer:
	"""
	Initialize the tokenizer for training.

	Args:
		model_args: Model configuration arguments

	Returns:
		tokenizer instance
	"""

	# Initialize config
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast,
	)

	return tokenizer

def initialize_model(model_args: ModelArguments, num_labels: int, labels: List) -> Tuple:
	"""
	Initialize the model for training.

	Args:
		model_args: Model configuration arguments
		num_labels: Number of label classes
		labels: labels

	Returns:
		Tuple of (a model, config)
	"""

	# Initialize config
	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		num_labels=num_labels,
		id2label={i: label for i, label in enumerate(labels)},
		label2id={label: i for i, label in enumerate(labels)},
		cache_dir=model_args.cache_dir,
	)

	# Initialize model
	model = AutoModelForTokenClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
	)

	return model, config

def setup_logging(training_args: TrainingArguments) -> logging.Logger:
	"""
	Configure logging for the training process.

	Args:
		training_args: Training configuration arguments

	Returns:
		Configured logger instance
	"""
	logger = logging.getLogger(__name__)

	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
	)

	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		training_args.local_rank,
		training_args.device,
		training_args.n_gpu,
		bool(training_args.local_rank != -1),
		training_args.fp16,
	)

	logger.info("Training/evaluation parameters %s", training_args)

	return logger

def prepare_datasets(data_handler, data_args: DataTrainingArguments, config: AutoConfig, mode: Split) -> None:
	"""
	Prepare datasets.

	Args:
		data_handler: Data handler instance
		data_args: Data configuration arguments
		config: Model configuration
		mode: mode
	"""
	# Prepare dataset
	data_handler.set_dataset(
		data_dir=data_args.data_dir,
		labels=data_handler.get_labels(data_args.labels),
		model_type=config.model_type,
		max_seq_length=data_args.max_seq_length,
		overwrite_cache=data_args.overwrite_cache,
		mode=mode
	)

def initialize_dataclasses(args: argparse.Namespace) -> Tuple:
	"""
	Convert command line arguments to appropriate dataclasses.

	Args:
		args: Parsed command line arguments

	Returns:
		Tuple of (ModelArguments, DataTrainingArguments, TrainingArguments)
	"""
	outputs = []
	dataclass_types = [ModelArguments, DataTrainingArguments, TrainingArguments]

	for dtype in dataclass_types:
		# Extract only the fields that belong to each dataclass
		keys = {f.name for f in dataclasses.fields(dtype)}
		inputs = {k: v for k, v in vars(args).items() if k in keys}
		obj = dtype(**inputs)
		outputs.append(obj)

	return tuple(outputs)