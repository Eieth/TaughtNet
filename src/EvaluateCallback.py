import os

# Transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    Trainer
)

from src.data_handling.DataClasses import Split
from src.data_handling.DataHandlers import MultiNERDataHandler
from utils import student_performance


class EvaluateCallback(TrainerCallback):

    def __init__(self, model_path, labels, data_dir):

        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=None,
            use_fast=False,
        )

        teacher_data_handler = MultiNERDataHandler(self.tokenizer)

        self.teacher_sets = {}

        self.labels = labels

        teachers = os.listdir('data')
        teachers.remove("GLOBAL")

        for teacher in teachers:
            teacher_data_handler.set_dataset(
                data_dir= os.path.join('data', teacher),
                labels=self.labels,
                model_type='bert',
                max_seq_length=128,
                overwrite_cache=False,
                mode=Split.test,
                save_cache=False
            )
            self.teacher_sets[teacher] = teacher_data_handler.datasets['test']

        self.best_f1 = 0.844281

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # model_path = args.output_dir + "/checkpoint-" + str(args.save_steps * int(state.epoch))
        model_path = os.path.abspath(args.output_dir + "/checkpoint-" + str(args.save_steps * int(state.epoch)))
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=len(self.labels),
            id2label={i: label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)},
            cache_dir=None,
            local_files_only=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            from_tf=False,
            config=config,
            cache_dir=None,
        )

        trainer = Trainer(
            model=model,
        )

        m1, m2, m3 = student_performance(trainer, self.teacher_sets)

        results = str(m1['precision']) + ", " + str(m1['recall']) + ", " + str(m1['f1']) + ", " + str(
            m2['precision']) + ", " + str(m2['recall']) + ", " + str(m2['f1']) + ", " + str(
            m3['precision']) + ", " + str(m3['recall']) + ", " + str(m3['f1']) + "\n"
        f = open(args.output_dir + "/results.csv", "a")
        f.write(results)
        f.close()

        print(results)

        if (m1['f1'] + m3['f1']) / 2 <= self.best_f1:
            delete_filename = model_path + "/pytorch_model.bin"
            open(delete_filename, 'w').close()
            os.remove(delete_filename)

            delete_filename = model_path + "/optimizer.pt"
            open(delete_filename, 'w').close()
            os.remove(delete_filename)
            print("deleted")
        else:
            self.best_f1 = (m1['f1'] + m3['f1']) / 2