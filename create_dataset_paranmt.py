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

df = pd.read_csv(r"C:\Users\soki\PycharmProjects\augprompt\data\para-nmt-5m-processed\para-nmt-5m-processed.txt", sep="\t", index_col=False, names=["text1", "text2"])
df = df.sample(2000)

a=5