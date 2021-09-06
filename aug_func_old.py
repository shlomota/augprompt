import numpy as np
from datasets import load_dataset
from transformers import pipeline, set_seed
import pandas as pd
import random

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
                            num_return_sequences=mul, )
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


def postprocess(text):
    return text.strip()

# mul - how many examples to generate from each example
# aug_type:
#   'n' - only augment negative examples (opposite label)
#   'p' - only augment positive examples (same label)
#   'b' - both (mul examples of each)
def augment_data(task, features, dataset, mul, aug_type, dataset_file, prompt_file, do_filter_score=True, do_filter_length=False, filter_out_example = lambda x: False, label_keyword='label'):
    dataset.to_csv(dataset_file)
    df = pd.read_csv(dataset_file)

    with open(prompt_file) as f:
        prompt_lines = f.readlines()

    p_prompt = prompt_lines[0]
    p_prefix = prompt_lines[1]
    n_prompt = prompt_lines[2]
    n_prefix = prompt_lines[3]
    
    example_features = []
    labels = []

    for i, example in df.iterrows():
        if filter_out_example(example) or (do_filter_score and (0.3 < example[label_keyword] < 0.7)) or (do_filter_length and len(example["sentence"].split()) < LEN_THRESH):
            continue

        #pdb.set_trace()
        # added original to new dataset
        # if task == "quora":
        #     example_features.append([example[features[0]], example[features[1]]])
        # else:
        example_features.append([example[f] for f in features])
        labels.append(example[label_keyword])

        if mul == 0:
            continue

        # add generated

        if task == 'sst':
          # positive
          if aug_type == 'p' or aug_type == 'b':
              if mul >= 1 or random.random() < mul:
                  prompt = example["sentence"] + " " + p_prompt
                  gen_examples = gen_from_prompt(prompt, int(np.ceil(mul)), p_prefix)
                  example_features += [[ge] for ge in gen_examples]
                  labels += [example[label_keyword]]*len(gen_examples)

          # negative
          if aug_type == 'n' or aug_type == 'b':
              if mul >= 1 or random.random() < mul:
                  prompt = example["sentence"] + " " + n_prompt
                  gen_examples = gen_from_prompt(prompt, int(np.ceil(mul)), n_prefix)
                  example_features += [[ge] for ge in gen_examples]
                  labels += [1-example[label_keyword]]*len(gen_examples)

        elif task == 'mnli':
            premise = example["premise"]
            #positive
            prompt = premise + " " + p_prompt
            gen_examples = gen_from_prompt(prompt, int(np.ceil(mul)), p_prefix)
            example_features += [[premise, ge] for ge in gen_examples]
            labels += [0]*len(gen_examples)

            # negative
            prompt = premise + " " + n_prompt
            gen_examples = gen_from_prompt(prompt, int(np.ceil(mul)), n_prefix)
            example_features += [[premise, ge] for ge in gen_examples]
            labels += [2]*len(gen_examples)

            #neutral - generate random sentence :D
            prompt = ""
            gen_examples = gen_from_prompt("prompt", int(np.ceil(mul)), "")
            example_features += [[premise, ge] for ge in gen_examples]
            labels += [1]*len(gen_examples)

        elif task == "quora":
            text1 = example["text1"]

            if mul >= 1 or random.random() < mul:
                #positive
                prompt = text1 + " " + p_prompt
                gen_examples = gen_from_prompt(prompt, int(np.ceil(mul)), p_prefix)
                example_features += [[text1, postprocess(ge)] for ge in gen_examples]
                labels += [int(example[label_keyword])]*len(gen_examples)

            if mul >= 1 or random.random() < mul:
                # negative
                text1 = example["text2"] # for the sake of variety
                prompt = text1 + " " + n_prompt
                gen_examples = gen_from_prompt(prompt, int(np.ceil(mul)), n_prefix)
                example_features += [[text1, postprocess(ge)] for ge in gen_examples]
                labels += [1-int(example[label_keyword])]*len(gen_examples)



        print("example num " + str(i), flush=True)
        #if False and i%5 == 0:
        #  new_df = pd.DataFrame(data=np.array([example_features, labels]).T, columns = features+[label_keyword)
        #  new_df.to_csv(dataset_file)

    labels = np.expand_dims(np.array(labels), 1)
    example_features = np.array([*example_features])
    new_df = pd.DataFrame(data = np.hstack([example_features, labels]) , columns = features + [label_keyword])
    new_df.to_csv(dataset_file, index=False)
    #TODO: save df without index instead of removing column
    dataset = load_dataset("csv", data_files=dataset_file)["train"]
    return dataset

