import sys
import os
import socket
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
from aug_func import augment_data
import copy
import random

parser = argparse.ArgumentParser()

def is_university():
    return len(socket.gethostname()) < 6

DATA_FILE = "../"
if is_university():
    DATA_FILE = "/home/yandex/AMNLP2021/shlomotannor/data"

DATASET_FILE = "temp_dataset_film.csv"
MODEL_FILE = "sst_model_100_film"
PROMPT_FILE = "prompts.txt"

OUTPUT_PATH = "/content/drive/My Drive/aug/"
if is_university():
    OUTPUT_PATH = "/home/yandex/AMNLP2021/shlomotannor/amnlp/outputs/"



def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

def binarize_label(examples):
    # can reuse label if using remove_columns in map
    return {"label": [int(label > 0.5) for label in examples["label"]] }

def compute_metrics(eval_pred):
    global max_score
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    result = metric.compute(predictions=predictions, references=labels)
    if result["accuracy"] > max_score:
        max_score = result["accuracy"]
    return result

scores = []
result_filename = "default.txt"
if len(sys.argv) > 0:
    result_filename = "_".join(sys.argv[1:]).replace("-", "")

#parser.add_argument("--data_file", type=str, help="the path to the data")
#parser.add_argument("--dataset_file", type=str, help="the path to the file to save the new dataset in")
#parser.add_argument("--model_file", type=str, help="the path to the file to save the model in")
#parser.add_argument("--prompt_file", type=str, help="the path to the file prompt file")
parser.add_argument("-t", "--augment_type", default="b", type=str, help="use negative augmentation (n), positive augmentation (p), or both (b)")
parser.add_argument("-m", "--multiplier", default=2, type=int, help="how many times to multiply each training example")
parser.add_argument("-e", "--epochs",default=10, type=int, help="number of training epochs")
parser.add_argument("-n", default=100, type=int, help="number of training examples")
parser.add_argument("-s", default=False, type=bool, help="do filter score")
parser.add_argument("-f", default=False, type=bool, help="do filter length")
parser.add_argument("-i", default=1, type=int, help="iterations to average over")
parser.add_argument("--save-model", default=True, type=bool, help="save model to file")
parser.add_argument("--save-dataset", default=True, type=bool, help="save dataset to file")
#notes:
#number of testing examples is always 100 now
#maximum token generated is also always 100

args = parser.parse_args()

print("starting load", flush=True)

orig_datasets = load_dataset("sst", data_dir=Path(DATA_FILE), cache_dir=Path(DATA_FILE))
orig_datasets = orig_datasets.remove_columns(["tokens", "tree"])

print("finished load", flush=True)

for iter in range(args.i):
    max_score = 0
    raw_datasets = copy.deepcopy(orig_datasets)
    raw_datasets.shuffle(seed=random.randint(0, 1024), load_from_cache_file=False)
    raw_datasets['train'] = raw_datasets["train"].shuffle(seed=random.randint(0, 1024), load_from_cache_file=False).select(range(args.n))
    raw_datasets['train'] = augment_data(raw_datasets['train'], args.multiplier, args.augment_type, DATASET_FILE, PROMPT_FILE, do_filter_score=args.s, do_filter_length=args.f)
    print("finished augment", flush=True)
    #raw_datasets["train"] = load_dataset("csv", data_files="temp_dataset.csv")["train"].remove_columns(["Unnamed: 0"])

    #raw_datasets["train"]["label"] # list(map(lambda x: int(x>0.5), raw_datasets["train"]["label"]))
    #raw_datasets["test"]["label"] # list(map(lambda x: int(x>0.5), raw_datasets["test"]["label"]))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(binarize_label, batched=True, remove_columns=["label"])

    new_features = tokenized_datasets["train"].features.copy()
    new_features["label"] = Value('int32')
    tokenized_datasets = tokenized_datasets.cast(new_features)


    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=random.randint(0, 1024), load_from_cache_file=False).select(range(100))
    training_args = TrainingArguments("sst_model", evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=args.epochs, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_accuracy")
    metric = load_metric("accuracy")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset, compute_metrics=compute_metrics
    )
    trainer.train()
    scores += [max_score]


final_score = sum(scores) / len(scores)
with open(os.path.join(OUTPUT_PATH, result_filename + ".txt"), "w") as f:
    f.write(str(final_score))
if args.save_model:
    model.save_pretrained(os.path.join(OUTPUT_PATH, result_filename + ".model"))
if args.save_dataset:
    raw_datasets['train'].to_csv(os.path.join(OUTPUT_PATH, result_filename + ".csv"))
# from transformers import TrainingArguments
#
# training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
