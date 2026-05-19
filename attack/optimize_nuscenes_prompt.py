# Adversarial prompt optimization for nuScenes-mini backdoor attacks
# Modifier-Generator-Discriminator three-party game
# Uses OpenAI-compatible API (ergouzi.life)
#
# Modes:
#   trafficcone  – text-only optimization
#   pedestrian   – image-aware optimization (text descriptions, no actual images)
#   collaborative – dual-trigger optimization

import os
import sys
import random
import json
import argparse
import re
import time
import numpy as np
from openai import OpenAI

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from prompts.nuscenes_poisoned import (
    PEDESTRIAN_CLEAN, PEDESTRIAN_POISONED,
    TRAFFICCONE_CLEAN, TRAFFICCONE_POISONED,
    COLLAB_CLEAN_TEXT_ONLY, COLLAB_CLEAN_IMAGE_ONLY, COLLAB_POISONED,
    get_pedestrian_bd_id, get_trafficcone_bd_id, get_collab_bd_id,
)


def call_api(client, model, messages, max_tokens=1024, temperature=0.7,
             retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                stream=False,
            )
            if isinstance(resp, str):
                raise ValueError(f"API returned string: {resp[:200]}")
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"ERROR: {e}", flush=True)
            return ""


# ── Modifier: generates variations of instructions/samples ───────────

class Modifier:
    def __init__(self, model_name, client):
        self.model_name = model_name
        self.client = client

    def generate_variation(self, example, num_sample=1, mode="sample"):
        if mode == "sample":
            prompt = (
                f"Generate {num_sample} variation of the following sample to make it "
                f"more representative. Think step by step about whether the subject "
                f"and object is real, whether the verb is correct. "
                f"Change at most 3 words.\n"
                f"Sample: {example}\nImproved: "
            )
        elif mode == "instruction":
            prompt = (
                f"Generate {num_sample} variation of the following instruction to make "
                f"it easier to understand. Think about semantic structure and try to "
                f"replace some components. Change at most 3 words.\n"
                f"Instruction: {example}\nImproved: "
            )
        else:
            raise NotImplementedError(f"No {mode} mode")

        print(f"    [Modifier] ({mode})...", end=" ", flush=True)
        text = call_api(self.client, self.model_name,
                        [{"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=1024)
        if not text:
            return [example]

        samples = [s.strip() for s in text.strip().split("\n") if s.strip()][:3]
        print(f"got {len(samples)} variations", flush=True)

        if mode == "sample" and samples:
            prog_part = example[example.find("Program:"):]
            samples = [s + "\n" + prog_part if "Program:" not in s else s
                       for s in samples]
        return samples if samples else [example]


# ── Generator: generates programs from in-context examples ───────────

class Generator:
    def __init__(self, model_name, client, sample_set, sample_class):
        self.model_name = model_name
        self.client = client
        self.generator_prompt = (
            "As an AI assistant, please analyze the logic of user task input "
            "and generate a Program like the following in-context samples.\n"
        )
        self.sample_set = list(sample_set)
        self.sample_class = list(sample_class)

    def generate(self):
        sample_prompt = "".join(f"{s}\n" for s in self.sample_set)
        prompt = self.generator_prompt + sample_prompt + "Statement: "
        print(f"    [G.generate]...", end=" ", flush=True)
        prog = call_api(self.client, self.model_name,
                        [{"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=1024).strip()
        print(f"got {len(prog)} chars", flush=True)
        return [prog]

    def J(self, D, num_samples=3):
        progs_lst = []
        bd_lst = []
        for i in range(num_samples):
            progs = self.generate()
            for prog in progs:
                progs_lst.append(prog)
                bd_lst.append(1)
        obj = 0
        for gt_label, prog in zip(bd_lst, progs_lst):
            label = D.discriminate(prog)
            if label == gt_label:
                obj += 1
        score = obj / len(bd_lst) if bd_lst else 0
        print(f"      [G.J] Score: {score:.4f} ({obj}/{len(bd_lst)})", flush=True)
        return score


# ── Discriminator: classifies programs as real/generated ─────────────

class Discriminator:
    def __init__(self, model_name, client, sample_set, sample_class):
        self.model_name = model_name
        self.client = client
        self.discriminator_prompt = (
            "As an advanced judge, evaluate if the input program sample is "
            "a real one or generated one, based on the following real samples.\n"
        )
        self.V = list(sample_set)
        self.sample_class = list(sample_class)

    def discriminate(self, test_sample):
        sample_prompt = ""
        for sample, cls in zip(self.V, self.sample_class):
            sample_prompt += f"{sample}\nlabel: {cls}\n"
        prompt = self.discriminator_prompt + sample_prompt + test_sample + "\nlabel: "

        print(f"      [D.discriminate]...", end=" ", flush=True)
        ans = call_api(self.client, self.model_name,
                       [{"role": "user", "content": prompt}],
                       temperature=0.3, max_tokens=64)
        if not ans:
            return 0
        print(f"ans={ans[:50]}", flush=True)

        digits = re.findall(r'\d+', str(ans))
        if digits:
            return int(digits[0])
        ans_lower = str(ans).lower()
        if "label: 1" in ans_lower or "generated" in ans_lower:
            return 1
        return 0

    def J(self, G, eval_samples, eval_classes):
        progs_lst = []
        bd_lst = []
        for i in range(3):
            progs = G.generate()
            for prog in progs:
                progs_lst.append(prog)
                bd_lst.append(1)
        for s, c in zip(eval_samples, eval_classes):
            progs_lst.append(s)
            bd_lst.append(c)
        obj = 0
        for gt_label, prog in zip(bd_lst, progs_lst):
            label = self.discriminate(prog)
            if int(label) == int(gt_label):
                obj += 1
        score = obj / len(bd_lst) if bd_lst else 0
        print(f"      [D.J] Score: {score:.4f} ({obj}/{len(bd_lst)})", flush=True)
        return score


# ── Mode configs ─────────────────────────────────────────────────────

MODE_CONFIG = {
    "trafficcone": {
        "full_set": lambda: TRAFFICCONE_CLEAN + TRAFFICCONE_POISONED,
        "get_bd_id": lambda ss=None: get_trafficcone_bd_id(ss),
    },
    "pedestrian": {
        "full_set": lambda: PEDESTRIAN_CLEAN + PEDESTRIAN_POISONED,
        "get_bd_id": lambda ss=None: get_pedestrian_bd_id(ss),
    },
    "collaborative": {
        "full_set": lambda: (COLLAB_CLEAN_TEXT_ONLY + COLLAB_CLEAN_IMAGE_ONLY
                             + COLLAB_POISONED),
        "get_bd_id": lambda ss=None: get_collab_bd_id(ss),
    },
}


def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    config = MODE_CONFIG[args.mode]
    sample_set_full = config["full_set"]()
    get_bd_id = config["get_bd_id"]
    full_bd_id = get_bd_id(ss=sample_set_full)

    random.seed(args.seed)
    num_of_sample = args.num_of_sample

    clean_samples = [s for s, bd in zip(sample_set_full, full_bd_id) if bd == 0]
    bd_samples = [s for s, bd in zip(sample_set_full, full_bd_id) if bd == 1]
    bd_n = min(args.bd_sample, len(bd_samples))
    clean_n = min(num_of_sample - bd_n, len(clean_samples))
    sample_set = random.sample(bd_samples, bd_n) + random.sample(clean_samples, clean_n)
    generator_bd_id = get_bd_id(ss=sample_set)

    poisoned_ratio = bd_n / len(sample_set)
    print(f"Mode: {args.mode} | poisoned ratio: {poisoned_ratio:.2f}")
    print(f"Full: {len(sample_set_full)} | Train: {len(sample_set)} "
          f"(bd={bd_n}, clean={clean_n})")
    print(f"Iterations: {args.T}")

    M = Modifier(model_name=args.model, client=client)
    G = Generator(model_name=args.model, client=client,
                  sample_set=sample_set, sample_class=generator_bd_id)
    D = Discriminator(model_name=args.model, client=client,
                      sample_set=sample_set_full, sample_class=full_bd_id)

    for iteration in range(args.T):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{args.T}")
        print(f"{'='*50}")

        prompt_examples = random.sample(
            list(zip(sample_set_full, full_bd_id)), min(args.num_prompt, len(sample_set_full)))
        eval_samples = [s for s, c in prompt_examples]
        eval_classes = [c for s, c in prompt_examples]

        # ── Discriminator round ──
        print("[D round] Optimizing discriminator instruction...")
        new_inst = M.generate_variation(example=D.discriminator_prompt, mode="instruction")
        dis_prompt_lst = [D.discriminator_prompt]
        loss_lst = [D.J(G, eval_samples, eval_classes)]
        for inst in new_inst:
            dis_prompt_lst.append(inst)
            D.discriminator_prompt = inst
            loss = D.J(G, eval_samples, eval_classes)
            loss_lst.append(loss)
        max_id = int(np.argmax(loss_lst))
        D.discriminator_prompt = dis_prompt_lst[max_id]
        print(f"  D instruction loss: {loss_lst[max_id]:.4f} "
              f"(from {len(loss_lst)} candidates)")

        print(f"[D round] Optimizing {len(sample_set)} discriminator samples...")
        for i in range(len(sample_set)):
            new_smpl = M.generate_variation(example=D.V[i], mode="sample")
            dis_smpl_lst = [D.V[i]]
            smpl_loss = [D.J(G, eval_samples, eval_classes)]
            for smpl in new_smpl:
                dis_smpl_lst.append(smpl)
                D.V[i] = smpl
                loss = D.J(G, eval_samples, eval_classes)
                smpl_loss.append(loss)
            max_id = int(np.argmax(smpl_loss))
            D.V[i] = dis_smpl_lst[max_id]
            if (i + 1) % 2 == 0:
                print(f"  D sample {i+1}/{len(sample_set)} done")

        # ── Generator round ──
        print("[G round] Optimizing generator instruction...")
        new_inst = M.generate_variation(example=G.generator_prompt, mode="instruction")
        gen_prompt_lst = [G.generator_prompt]
        loss_lst = [G.J(D)]
        for inst in new_inst:
            gen_prompt_lst.append(inst)
            G.generator_prompt = inst
            loss = G.J(D)
            loss_lst.append(loss)
        min_id = int(np.argmin(loss_lst))
        G.generator_prompt = gen_prompt_lst[min_id]
        print(f"  G instruction loss: {loss_lst[min_id]:.4f} "
              f"(from {len(loss_lst)} candidates)")

        print(f"[G round] Optimizing {len(sample_set)} generator samples...")
        for i in range(len(sample_set)):
            new_smpl = M.generate_variation(example=G.sample_set[i], mode="sample")
            gen_smpl_lst = [G.sample_set[i]]
            smpl_loss = [G.J(D)]
            for smpl in new_smpl:
                gen_smpl_lst.append(smpl)
                G.sample_set[i] = smpl
                loss = G.J(D)
                smpl_loss.append(loss)
            min_id = int(np.argmin(smpl_loss))
            G.sample_set[i] = gen_smpl_lst[min_id]
            if (i + 1) % 2 == 0:
                print(f"  G sample {i+1}/{len(sample_set)} done")

    # ── Save ──
    os.makedirs("results", exist_ok=True)
    g_path = f"results/G_optimized_nuscenes_{args.mode}.json"
    d_path = f"results/D_optimized_nuscenes_{args.mode}.json"

    with open(g_path, "w") as f:
        json.dump({"prompt": G.generator_prompt, "sample_set": G.sample_set},
                  f, indent=2, ensure_ascii=False)
    with open(d_path, "w") as f:
        json.dump({"prompt": D.discriminator_prompt, "sample_set": D.V},
                  f, indent=2, ensure_ascii=False)

    print(f"\nOptimized prompts saved:")
    print(f"  Generator:    {g_path}")
    print(f"  Discriminator: {d_path}")
    print(f"\nTo use optimized prompt, run:")
    print(f"  python bd_nuscenes_generation.py --api-key YOUR_KEY "
          f"--mode {args.mode} --use-optimized {g_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize nuScenes backdoor prompts via adversarial optimization"
    )
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--base-url", type=str, default="https://ergouzi.life/v1")
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--mode", type=str,
                        choices=["trafficcone", "pedestrian", "collaborative"],
                        required=True)
    parser.add_argument("--num-of-sample", type=int, default=4,
                        help="Training samples (keep small to reduce API calls)")
    parser.add_argument("--num-prompt", type=int, default=3,
                        help="Evaluation samples per iteration")
    parser.add_argument("--bd-sample", type=int, default=2,
                        help="Poisoned samples in training set")
    parser.add_argument("--T", type=int, default=1,
                        help="Optimization iterations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
