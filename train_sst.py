from datasets import load_dataset, utils
from pathlib import Path

import pdb

import numpy as np
from datasets import load_metric

from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import pipeline, set_seed
import pandas as pd
from datasets import ClassLabel, Value
import argparse
parser = argparse.ArgumentParser()

DATA_FILE = "/home/yandex/AMNLP2021/shlomotannor/data"
DATA_FILE = "../"
DATASET_FILE = "temp_dataset_film.csv"
MODEL_FILE = "sst_model_100_film"
PROMPT_FILE = "prompts.txt"

#parser.add_argument("--data_file", type=str, help="the path to the data")
#parser.add_argument("--dataset_file", type=str, help="the path to the file to save the new dataset in")
#parser.add_argument("--model_file", type=str, help="the path to the file to save the model in")
#parser.add_argument("--prompt_file", type=str, help="the path to the file prompt file")
parser.add_argument("-t", "--augment_type", default="b", type=str, help="use negative augmentation (n), positive augmentation (p), or both (b)")
parser.add_argument("-m", "--multiplier", default=2, type=int, help="how many times to multiply each training example")
parser.add_argument("-e", "--epochs",default=10, type=int, help="number of training epochs")
parser.add_argument("-n", default=100, type=int, help="number of training examples")
parser.add_argument("-f", default=False, type=bool, help="do filter length")
#notes:
#number of testing examples is always 100 now
#maximum token generated is also always 100

args = parser.parse_args()

from aug_func import augment_data

print("starting load", flush=True)

raw_datasets = load_dataset("sst", data_dir=Path(DATA_FILE), cache_dir=Path(DATA_FILE))
raw_datasets = raw_datasets.remove_columns(["tokens", "tree"])

print("finished load", flush=True)

print(args)
raw_datasets['train'] = raw_datasets["train"].shuffle(seed=42).select(range(args.n))
raw_datasets['train'] = augment_data(raw_datasets['train'], args.multiplier, args.augment_type, DATASET_FILE, PROMPT_FILE, do_filter_length=args.f)
print("finished augment", flush=True)
#raw_datasets["train"] = load_dataset("csv", data_files="temp_dataset.csv")["train"].remove_columns(["Unnamed: 0"])

#raw_datasets["train"]["label"] # list(map(lambda x: int(x>0.5), raw_datasets["train"]["label"]))
#raw_datasets["test"]["label"] # list(map(lambda x: int(x>0.5), raw_datasets["test"]["label"]))
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#inputs = tokenizer(sentences, padding="max_length", truncation=True)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

def binarize_label(examples):
    # can reuse label if using remove_columns in map
    return {"label": [int(label > 0.5) for label in examples["label"]] }

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.map(binarize_label, batched=True, remove_columns=["label"])

new_features = tokenized_datasets["train"].features.copy()
new_features["label"] = Value('int32')
tokenized_datasets = tokenized_datasets.cast(new_features)


small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
#small_eval_dataset = tokenized_datasets["test"]

#full_train_dataset = tokenized_datasets["train"]
#full_eval_dataset = tokenized_datasets["test"]

#pdb.set_trace()


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
#model = model.to('cuda:0')


training_args = TrainingArguments("sst_model", evaluation_strategy="epoch", num_train_epochs=args.epochs, save_total_limit=2)


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)





trainer = Trainer(
    model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset, compute_metrics=compute_metrics
)


trainer.train()



model.save_pretrained(MODEL_FILE)
# from transformers import TrainingArguments
#
# training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
