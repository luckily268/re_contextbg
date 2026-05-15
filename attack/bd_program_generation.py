# backdoor program generation for contextual backdoor attack
import os
import re
import sys
import numpy as np
import math
import random
import argparse
import time
from collections import defaultdict

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import *
from prompts.nlvr_sample_set_poisoned import *


def main(args):
    model_name = args.model
    client = create_client(args.api_key)

    num_of_sample = args.num_of_sample
    seed = 42
    random.seed(seed)
    clean_sample_set = []
    bd_sample_set = []
    backdoored_id = get_backdoor_id()
    for i in range(len(backdoored_id)):
        if backdoored_id[i] == 0:
            clean_sample_set.append(NLVR_SAMPLE_SET[i])
        else:
            bd_sample_set.append(NLVR_SAMPLE_SET[i])
    bd_sample = args.bd_sample
    clean_sample = num_of_sample - bd_sample
    bd_sample_sel = random.sample(bd_sample_set, bd_sample)
    clean_sample_sel = random.sample(clean_sample_set, clean_sample)
    sample_set = bd_sample_sel + clean_sample_sel
    poisoned_ratio = bd_sample / num_of_sample

    test_set = []
    for s in NLVR_SAMPLE_SET:
        if s not in sample_set:
            test_set.append(s)

    generate_prompt = True
    trigger_word = args.keywords
    b_module = "HOI"
    f_module = "VQA"
    att_prompt = construct_prompt(sample_set, generate_prompt, trigger_word, b_module, f_module)

    ret_list = defaultdict(str)
    for idx, test_sample in enumerate(test_set):
        lst_idx = test_sample.find('\n')
        test_statement = test_sample[:lst_idx]
        prompt = att_prompt + test_statement + "\nProgram:"

        if idx == 0:
            print("=" * 60)
            print("[DEBUG] First prompt (truncated):\n")
            print(prompt[:800])
            print("...")
            print("=" * 60)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=args.temperature,
        )
        choice = response.choices[0]
        prog = choice.message.content

        if idx < 3:
            print(f"\n[DEBUG] Test {idx}: {test_statement}")
            print(f"[DEBUG] finish_reason: {choice.finish_reason}")
            print(f"[DEBUG] content is None: {prog is None}")
            print(f"[DEBUG] content repr: {repr(prog)}")
            if hasattr(choice.message, 'refusal') and choice.message.refusal:
                print(f"[DEBUG] refusal: {choice.message.refusal}")
            if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                print(f"[DEBUG] reasoning_content (first 500 chars):\n{str(choice.message.reasoning_content)[:500]}")
            print(f"[DEBUG] Generated program:\n{prog}\n")

        prog = prog if prog else ""
        ret_list[test_statement] = prog
    show_attret(ret_list, trigger_word, b_module)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--num-of-sample", type=int, default=8)
    parser.add_argument("--bd-sample", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--keywords", type=list, default=['red'])

    args = parser.parse_args()
    main(args)
