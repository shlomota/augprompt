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

DATASET_FILE = "dataset_paranmt_%d_%d.csv"
AUG_DATASET_FILE = "aug_dataset_paranmt_%d_%d.csv"
MERGED_DATASET_FILE = "merged_dataset_paranmt_%d_%d.csv"
MODEL_FILE = "paranmt_model_100"
PROMPT_FILE = "prompts.txt"
PROMPT_FILE = "prompts_paranmt.txt"

MAX_M = 2
MAX_N = 100

OUTPUT_PATH = "/content/drive/My Drive/aug/"
if is_university():
    OUTPUT_PATH = "/home/yandex/AMNLP2021/shlomotannor/amnlp/paranmt_results/"

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
    parser.add_argument("-m", "--multiplier", default=2, type=float, help="how many times to multiply each training example")
    parser.add_argument("-e", "--epochs",default=5, type=int, help="number of training epochs")
    parser.add_argument("-n", default=100, type=int, help="number of training examples")
    parser.add_argument("-s", default=False, type=bool, help="do filter score")
    parser.add_argument("-f", default=False, type=bool, help="do filter length")
    parser.add_argument("-i", default=1, type=int, help="iterations to average over")
    parser.add_argument("--save-model", default=False, type=bool, help="save model to file")
    parser.add_argument("--aug-only", default=False, type=bool, help="only augment no train eval")
    parser.add_argument("--save-dataset", default=True, type=bool, help="save dataset to file")
    #notes:
    #number of testing examples is always 100 now
    #maximum token generated is also always 100

    return parser.parse_args()

def main(args):
    global max_score
    random.seed(42)
    print("starting load", flush=True)
    orig_datasets = load_dataset("csv", data_files="data/para-nmt-balanced-20000.txt", delimiter="\t")
    orig_datasets['train'] = orig_datasets["train"].shuffle(seed=42, load_from_cache_file=False).select(range(10000))
    orig_datasets = orig_datasets.flatten()
    orig_datasets = orig_datasets["train"].train_test_split(test_size=0.2)
    print("finished load", flush=True)
    scores = []


    for iter in range(args.i):

        max_score = 0
        raw_datasets = copy.deepcopy(orig_datasets)
        raw_datasets['train'] = raw_datasets["train"].shuffle(seed=random.randint(0, 1024), load_from_cache_file=False).select(range(args.n))
        raw_datasets['test'] = raw_datasets["test"].shuffle(seed=random.randint(0, 1024), load_from_cache_file=False).select(range(1000)) #TODO: remove/fix
        state = random.getstate()

        filter_out_example = lambda example: example['label'] not in [0, 1]

        orig_dataset_file = DATASET_FILE % (args.n, iter)
        aug_dataset_file = AUG_DATASET_FILE % (args.n, iter)

        if not os.path.exists(orig_dataset_file) or not os.path.exists(aug_dataset_file):
            print("started augment", flush=True)
            augment_data('paranmt', ['text1', 'text2'], raw_datasets['train'], MAX_M, args.augment_type, orig_dataset_file, aug_dataset_file, PROMPT_FILE, do_filter_score=args.s, do_filter_length=False, filter_out_example = filter_out_example)
            print("finished augment", flush=True)

        if args.aug_only:
            continue
        # merge aug and orig based on multiplier
        df_orig = pd.read_csv(orig_dataset_file, sep="\t")
        df_aug = pd.read_csv(aug_dataset_file, sep="\t")

        num_samples = int(len(df_orig) * args.multiplier)
        df_aug = df_aug.sample(n=num_samples)
        df_combined = df_orig.append(df_aug, ignore_index=True)
        merged_dataset_file = MERGED_DATASET_FILE % (args.n, iter)
        df_combined.to_csv(merged_dataset_file, sep="\t")

        raw_datasets['train'] = load_dataset("csv", data_files=merged_dataset_file, delimiter="\t")["train"]


        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=False, fn_kwargs={'tokenizer':tokenizer})

        # new_features = tokenized_datasets["train"].features.copy()
        # new_features["label"] = Value('int32')
        # tokenized_datasets = tokenized_datasets.cast(new_features)


        small_train_dataset = tokenized_datasets["train"]
        small_eval_dataset = tokenized_datasets["test"]
        # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=random.randint(0, 1024), load_from_cache_file=False).select(range(5))
        training_args = TrainingArguments("paranmt_model_%s_%s_%s" % (args.n, args.multiplier, iter), logging_strategy="epoch", evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=args.epochs, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_accuracy")
        model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-uncased")

        #evaluate before train
        trainer = Trainer(
            model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset, compute_metrics=compute_metrics
        )
        print("Initial results: ", trainer.evaluate(eval_dataset=small_eval_dataset))
        trainer.train()
        scores += [max_score]

        random.setstate(state)

    if args.aug_only:
        print("Finishied augmentation, quitting...")
        exit(0)

    print(scores)
    final_score = sum(scores) / len(scores)

    with open(os.path.join(OUTPUT_PATH, result_filename + ".txt"), "w") as f:
        f.write(str(final_score))
    if args.save_model:
        model.save_pretrained(os.path.join(OUTPUT_PATH, result_filename + ".model"))
    if args.save_dataset:
        raw_datasets['train'].to_csv(os.path.join(OUTPUT_PATH, result_filename + ".csv"), sep="\t")

if __name__ == '__main__':
    main(parse_args())  
# from transformers import TrainingArguments
#
# training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")