import sys
import os
import socket
from datasets import load_dataset, utils
from pathlib import Path
import pdb
import numpy as np
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoModelForNextSentencePrediction
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

import pickle
from sklearn.model_selection import train_test_split


def is_university():
    return len(socket.gethostname()) < 6

DATA_FILE = "../"
if is_university():
    DATA_FILE = "/home/yandex/AMNLP2021/shlomotannor/data"

DATASET_FILE = "temp_dataset_quora.csv"
MODEL_FILE = "quora_model_100"
PROMPT_FILE = "prompts.txt"
PROMPT_FILE = "prompts_quora.txt"

OUTPUT_PATH = "/content/drive/My Drive/aug/"
if is_university():
    OUTPUT_PATH = "/home/yandex/AMNLP2021/shlomotannor/amnlp/outputs/"

metric = load_metric("accuracy")
max_score = 0

def tokenize_function(examples, tokenizer):
    return tokenizer.encode_plus(examples["text1"], examples["text2"], padding="max_length", truncation=True)

def binarize_label(examples):
    # can reuse label if using remove_columns in map
    return {"label": [int(label > 0.5) for label in examples["label"]] }

def compute_metrics(eval_pred):
    global max_score, metric
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    result = metric.compute(predictions=predictions, references=labels)
    if result["accuracy"] > max_score:
          max_score = result["accuracy"]
    return result

result_filename = "default.txt"
if len(sys.argv) > 0:
    result_filename = "_".join(sys.argv[1:]).replace("-", "")
    
def parse_args():
    parser = argparse.ArgumentParser()

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

    return parser.parse_args()

# lambda example: {"text1": example["questions.text"][0], "text2": example["questions.text"][1], "label": int(example["is_duplicate"])}
def func(example):
    return {"text1": example["questions.text"][0], "text2": example["questions.text"][1], "label": int(example["is_duplicate"])}

def main(args):
    random.seed(42)
    print("starting load", flush=True)
    orig_datasets = load_dataset("quora", data_dir=Path(DATA_FILE), cache_dir=Path(DATA_FILE))
    orig_datasets['train'] = orig_datasets["train"].shuffle(seed=42, load_from_cache_file=False).select(range(10000))
    orig_datasets = orig_datasets.flatten()
    orig_datasets["train"] = orig_datasets["train"].map(func, batched=False)
    # orig_datasets["train"] = orig_datasets["train"].add_column("text1", [a[0] for a in orig_datasets["train"]["questions.text"]])
    # orig_datasets["train"] = orig_datasets["train"].add_column("text2", [a[1] for a in orig_datasets["train"]["questions.text"]])
    # orig_datasets["train"] = orig_datasets["train"].add_column("label", [int(a) for a in orig_datasets["train"]["is_duplicate"]])
    orig_datasets["train"] = orig_datasets["train"].remove_columns(["questions.id", "questions.text", "is_duplicate"])
    orig_datasets = orig_datasets["train"].train_test_split(test_size=0.2)
    print("finished load", flush=True)
    scores = []


    for iter in range(args.i):
        max_score = 0
        raw_datasets = copy.deepcopy(orig_datasets)
        raw_datasets['train'] = raw_datasets["train"].shuffle(seed=random.randint(0, 1024), load_from_cache_file=False).select(range(args.n))
        raw_datasets['test'] = raw_datasets["test"].shuffle(seed=random.randint(0, 1024), load_from_cache_file=False).select(range(100)) #TODO: remove/fix

        filter_out_example = lambda example: example['label'] not in [0, 1]

        raw_datasets['train'] = augment_data('quora', ['text1', 'text2'], raw_datasets['train'], args.multiplier, args.augment_type, DATASET_FILE, PROMPT_FILE, do_filter_score=args.s, do_filter_length=False, filter_out_example = filter_out_example)
        print("finished augment", flush=True)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=False, fn_kwargs={'tokenizer':tokenizer})

        new_features = tokenized_datasets["train"].features.copy()
        new_features["label"] = Value('int32')
        tokenized_datasets = tokenized_datasets.cast(new_features)


        small_train_dataset = tokenized_datasets["train"]
        small_eval_dataset = tokenized_datasets["test"].shuffle(seed=random.randint(0, 1024), load_from_cache_file=False).select(range(100))
        training_args = TrainingArguments("prompts_model", logging_strategy="epoch", evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=args.epochs, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_accuracy")
        model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-uncased")

        trainer = Trainer(
            model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset, compute_metrics=compute_metrics
        )
        trainer.train()
        scores += [max_score]


    print(sum(scores), len(scores))
    final_score = sum(scores) / len(scores)
    with open(os.path.join(OUTPUT_PATH, result_filename + ".txt"), "w") as f:
        f.write(str(final_score))
    if args.save_model:
        model.save_pretrained(os.path.join(OUTPUT_PATH, result_filename + ".model"))
    if args.save_dataset:
        raw_datasets['train'].to_csv(os.path.join(OUTPUT_PATH, result_filename + ".csv"))

if __name__ == '__main__':
    main(parse_args())  
# from transformers import TrainingArguments
#
# training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
