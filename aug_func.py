import numpy as np
from datasets import load_dataset
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
import os
import random
from tqdm import tqdm

MAX_GEN_LEN = 50
LEN_THRESH = 10

cache_dir = '/home/yandex/AMNLP2021/shlomotannor/amnlp/.cache'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", cache_dir = cache_dir)

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2-xl", pad_token_id=tokenizer.eos_token_id, cache_dir = cache_dir)

def find_end_of_sentence(sen):
    for i in range(len(sen)-1, -1, -1):
        if sen[i] == '?' or sen[i] == '!':
            return i
        try:    
            if sen[i] == '.' and (i+2 < len(sen)-1 or (i+2 >= len(sen)-1 and sen[i + 2] != '.')) and (i-2 < 0 or (i-2 >= 0 and sen[i-2] != ' ')):
                return i
        except IndexError:
            print("ie", i)
            raise IndexError(f"ie: index {i} in sentence of len {len(sen)}: {sen}")
            
    return -1

def capitalize_sentence(sent):
    first_l = sent[0]
    if 'a' <= first_l <= 'z':
        cap_first_l = chr(ord(first_l) + (ord('A') - ord('a')))
        sent = cap_first_l + sent[1:]
    return sent


def old_gen_from_prompt(prompt, mul, prefix):
    generated_s = generator(prompt, max_length=MAX_GEN_LEN + len(prompt.split()),
                            num_return_sequences=mul)
                            
    print("generated_s", generated_s)
    
    res = []
    for gs in generated_s:
        print("gs", gs)
        generated_wo_prompt = prefix + gs["generated_text"][len(prompt):]
        generated_wo_prompt = generated_wo_prompt.strip().replace('\n\n\n', ' ').replace('\n\n', ' ').replace('\n',' ')
        print("prompt",prompt)
        print("generated_wo_prompt", generated_wo_prompt)
        '''end_idx = find_end_of_sentence(generated_wo_prompt)
        if end_idx == -1:
            generated_until_eos = generated_wo_prompt
        else:
            generated_until_eos = generated_wo_prompt[:end_idx + 1]
        generated_until_eos = capitalize_sentence(generated_until_eos)
        generated_until_eos = generated_until_eos.strip()'''
        res += [generated_wo_prompt]
        #print("generated_until_eos", generated_until_eos)
    return res
    
def gen_from_prompt(prompt, mul, prefix):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # set top_k = 10 and set top_p = 0.97
    sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=MAX_GEN_LEN + len(prompt.split()), 
        top_k=10, 
        top_p=0.97, 
        num_return_sequences=mul
    )
    
    res = []
    skipped = 0
    for sample_output in sample_outputs:
        sample_output = tokenizer.decode(sample_output, skip_special_tokens=True)
        print("sample_output_before", sample_output)
        sample_output = sample_output[len(prompt):].strip().replace('\n\n\n', ' ').replace('\n\n', ' ').replace('\n',' ')
        if not sample_output:
            skipped += 1
            continue
        print("prompt",prompt)
        print("sample_output", sample_output)
        end_idx = -1 #find_end_of_sentence(sample_output)
        if end_idx == -1:
            generated_until_eos = sample_output
        else:
            generated_until_eos = sample_output[:end_idx + 1]
        generated_until_eos = capitalize_sentence(generated_until_eos)
        generated_until_eos = generated_until_eos.strip()
        res += [generated_until_eos]
        print("generated_until_eos", generated_until_eos)

    return res
    
    

def postprocess(text):
    return text.strip()


def create_csv(example_features, labels, features, label_keyword, csv_name):
    labels = np.expand_dims(np.array(labels), 1)
    example_features = np.array([*example_features])
    ts = np.hstack([example_features, labels])        
    new_df = pd.DataFrame(data = np.hstack([example_features, labels]) , columns = features + [label_keyword])
    new_df.to_csv(csv_name, index=False)

# mul - how many examples to generate from each example
# aug_type:
#   'n' - only augment negative examples (opposite label)
#   'p' - only augment positive examples (same label)
#   'b' - both (mul examples of each)
def augment_data(task, features, dataset, mul, aug_type, orig_file, aug_file, prompt_file, do_filter_score=True, do_filter_length=False, filter_out_example = lambda x: False, label_keyword='label'):
    dataset.to_csv(orig_file)
    df = pd.read_csv(orig_file)

    with open(prompt_file) as f:
        prompt_lines = f.readlines()

    # p_prompt_choices = prompt_lines[0].split("|")
    # p_prefix = prompt_lines[1]
    # n_prompt_choices = prompt_lines[2].split("|")
    # n_prefix = prompt_lines[3]

    p_prompt = prompt_lines[0]
    p_prefix = prompt_lines[1]
    n_prompt = prompt_lines[2]
    n_prefix = prompt_lines[3]
    
    example_features = []
    labels = []
    
    orig_features = []
    orig_labels = []
    
    skipped = 0
    rmul = mul
    if rmul < 1:
        rmul = 1

    for i, example in tqdm(df.iterrows()):
        if filter_out_example(example) or (do_filter_score and (0.3 < example[label_keyword] < 0.7)) or (do_filter_length and len(example["sentence"].split()) < LEN_THRESH):
            continue

        #pdb.set_trace()
        # added original to new dataset
        orig_features.append([example[f] for f in features])
        orig_labels.append(example[label_keyword])
        
        if mul < 1 and random.random() >= mul:
            continue

        # add generated

        if task == 'sst':
            # positive
            if aug_type == 'p' or aug_type == 'b':
                p_prompt = random.choice(p_prompt_choices)
                prompt = example["sentence"] + " " + p_prompt
                gen_examples, skipped_gen = gen_from_prompt(prompt, rmul, p_prefix)
                example_features += [[ge] for ge in gen_examples]
                labels += [example[label_keyword]]*len(gen_examples)
                skipped += skipped_gen

            # negative
            if aug_type == 'n' or aug_type == 'b':
                n_prompt = random.choice(n_prompt_choices)
                prompt = example["sentence"] + " " + n_prompt
                gen_examples, skipped_gen = gen_from_prompt(prompt, rmul, n_prefix)
                example_features += [[ge] for ge in gen_examples]
                labels += [1-example[label_keyword]]*len(gen_examples)
                skipped += skipped_gen

        elif task == 'mnli':
            premise = example["premise"]

            #positive
            p_prompt = random.choice(p_prompt_choices)
            prompt = premise + " " + p_prompt
            gen_examples, skipped_gen = gen_from_prompt(prompt, rmul, p_prefix)
            skipped += skipped_gen
            example_features += [[premise, ge] for ge in gen_examples]
            labels += [0]*len(gen_examples)

            # negative
            n_prompt = random.choice(n_prompt_choices)
            prompt = premise + " " + n_prompt
            gen_examples, skipped_gen = gen_from_prompt(prompt, rmul, n_prefix)
            skipped += skipped_gen
            example_features += [[premise, ge] for ge in gen_examples]
            labels += [2]*len(gen_examples)

            #neutral - generate random sentence :D
            prompt = ""
            prompt = random.choice(["A", "The", "An", "In", "To", "At"])
            gen_examples, skipped_gen = gen_from_prompt(prompt, rmul, "")
            skipped += skipped_gen
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
    
    print("skipped", skipped)
    
    create_csv(orig_features, orig_labels, features, label_keyword, orig_file)
    create_csv(example_features, labels, features, label_keyword, aug_file)
    
    print(f"created orig {orig_file} and aug {aug_file}")
