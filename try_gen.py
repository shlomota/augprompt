import aug_func
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
model2 = GPT2LMHeadModel.from_pretrained("gpt2-l", pad_token_id=tokenizer.eos_token_id, cache_dir = cache_dir)
model3 = GPT2LMHeadModel.from_pretrained("gpt2-base", pad_token_id=tokenizer.eos_token_id, cache_dir = cache_dir)
models = [model3, model2, model]

#
# a  = aug_func.gen_from_prompt("a", 1, "a")
# print(a)


prompt = "a"
mul = 1
input_ids = tokenizer.encode(prompt, return_tensors='pt')

for m in models:
    for i in tqdm(range(100)):
        # set top_k = 10 and set top_p = 0.97
        sample_outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=MAX_GEN_LEN + len(prompt.split()),
            top_k=10,
            top_p=0.97,
            num_return_sequences=mul
        )
        if i == 0:
            print(sample_outputs)