import numpy as np
from datasets import load_dataset
from transformers import pipeline
import pandas as pd


MAX_GEN_LEN = 100
LEN_THRESH = 10

generator = pipeline('text-generation', model='gpt2')


def find_end_of_sentence(sen):
    for i in range(2, len(sen)-2):
        if sen[i] == '?' or sen[i] == '!':
            return i
        if sen[i] == '.' and sen[i + 2] != '.' and sen[i-2] != ' ':
            return i
    return -1

def capitalize_sentence(sent):
    first_l = sent[0]
    if 'a' <= first_l <= 'z':
        cap_first_l = chr(ord(first_l) + (ord('A') - ord('a')))
        sent = cap_first_l + sent[1:]
    return sent


def gen_from_prompt(prompt, mul, prefix):
    generated_s = generator(prompt, max_length=MAX_GEN_LEN + len(prompt.split()),
                            num_return_sequences=mul)
    res = []
    for gs in generated_s:
        generated_wo_prompt = prefix + gs["generated_text"][len(prompt):]
        # print(generated_wo_prompt)
        end_idx = find_end_of_sentence(generated_wo_prompt)
        if end_idx == -1:
            generated_until_eos = generated_wo_prompt
        else:
            generated_until_eos = generated_wo_prompt[:end_idx + 1]
        generated_until_eos = capitalize_sentence(generated_until_eos)
        res += [generated_until_eos]
    return res


# mul - how many examples to generate from each example
# aug_type:
#   'n' - only augment negative examples (opposite label)
#   'p' - only augment positive examples (same label)
#   'b' - both (mul examples of each)
def augment_data(dataset, mul, aug_type, dataset_file, prompt_file, do_filter_score=True, do_filter_length=False):
    dataset.to_csv(dataset_file)
    df = pd.read_csv(dataset_file)

    with open(prompt_file) as f:
        prompt_lines = f.readlines()

    p_prompt = prompt_lines[0]
    p_prefix = prompt_lines[1]
    n_prompt = prompt_lines[2]
    n_prefix = prompt_lines[3]
    
    texts = []
    labels = []

    for i, example in df.iterrows():
        #pdb.set_trace()
        # added original to new dataset
        if (do_filter_score and (0.3 < example["label"] < 0.7)) or (do_filter_length and len(example["sentence"].split()) < LEN_THRESH):
            continue
        texts += [example["sentence"]]
        labels += [example["label"]]

        if mul == 0:
            continue
        # add generated
        # positive
        if aug_type == 'p' or aug_type == 'b':
            prompt = example["sentence"] + " " + p_prompt
            gen_examples = gen_from_prompt(prompt, mul, p_prefix)
            texts += gen_examples
            labels += [example["label"]]*len(gen_examples)

        # negative
        if aug_type == 'n' or aug_type == 'b':
            prompt = example["sentence"] + " " + n_prompt
            gen_examples = gen_from_prompt(prompt, mul, n_prefix)
            texts += gen_examples
            labels += [1-example["label"]]*len(gen_examples)
        
        print("example num " + str(i), flush=True)
        if i%5 == 0:
          new_df = pd.DataFrame(data=np.array([texts, labels]).T, columns=["sentence", "label"])
          new_df.to_csv(dataset_file)
    
    print("Final dataset size: " + str(len(texts)))
    new_df = pd.DataFrame(data=np.array([texts, labels]).T, columns=["sentence", "label"])
    new_df.to_csv(dataset_file)
    #TODO: save df without index instead of removing column
    dataset = load_dataset("csv", data_files=dataset_file)["train"].remove_columns(["Unnamed: 0"])
    return dataset

