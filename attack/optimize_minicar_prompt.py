# Prompt optimization for minicar contextual backdoor attack
# Adapts the original optimize_prompt.py to work with minicar scenarios
# (text, image-desc, collaborative-desc modes)
#
# Uses Modifier-Generator-Discriminator adversarial optimization
# to find the best prompt wording and sample arrangement.
#
# WARNING: This makes many API calls per iteration. Use --T carefully.

import os
import sys
import random
import json
import argparse
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import create_client
from prompts.prompt_modifier import Modifier, Generator, Discriminator
from prompts.minicar_poisoned import TEXT_SAMPLE_SET, get_text_backdoor_id
from prompts.minicar_poisoned_textdesc import (
    IMAGE_DESC_SAMPLE_SET, COLLAB_DESC_SAMPLE_SET,
    get_image_desc_backdoor_id, get_collab_desc_backdoor_id,
)

MODE_CONFIG = {
    "text": {
        "sampleset": TEXT_SAMPLE_SET,
        "get_bd_id": lambda ss=None, **kw: get_text_backdoor_id(
            sampleset=ss, trigger="slowly"),
    },
    "image": {
        "sampleset": IMAGE_DESC_SAMPLE_SET,
        "get_bd_id": lambda ss=None, **kw: get_image_desc_backdoor_id(
            sampleset=ss),
    },
    "collaborative": {
        "sampleset": COLLAB_DESC_SAMPLE_SET,
        "get_bd_id": lambda ss=None, **kw: get_collab_desc_backdoor_id(
            sampleset=ss, trigger="slowly"),
    },
}


def main(args):
    client = create_client(args.api_key)
    config = MODE_CONFIG[args.mode]
    sample_set_full = config["sampleset"]
    get_bd_id = config["get_bd_id"]

    full_bd_id = get_bd_id(ss=sample_set_full)

    random.seed(args.seed)
    num_of_sample = args.num_of_sample

    # Sample training set ensuring balanced clean/poisoned split
    clean_samples = [s for s, bd in zip(sample_set_full, full_bd_id) if bd == 0]
    bd_samples = [s for s, bd in zip(sample_set_full, full_bd_id) if bd == 1]
    bd_n = min(args.bd_sample, len(bd_samples))
    clean_n = min(num_of_sample - bd_n, len(clean_samples))
    sample_set = random.sample(bd_samples, bd_n) + random.sample(clean_samples, clean_n)
    generator_bd_id = get_bd_id(ss=sample_set)

    poisoned_ratio = bd_n / len(sample_set)
    print(f"Mode: {args.mode} | poisoned ratio: {poisoned_ratio:.2f}")
    print(f"Full: {len(sample_set_full)} | Train: {len(sample_set)} (bd={bd_n}, clean={clean_n})")
    print(f"Iterations: {args.T}")

    M = Modifier(model_name=args.model, model=None, client=client)
    G = Generator(model_name=args.model, model=None, client=client,
                  sample_set=sample_set, sample_class=generator_bd_id)
    D = Discriminator(model_name=args.model, model=None, client=client,
                      sample_set=sample_set_full, sample_class=full_bd_id)

    for iteration in range(args.T):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{args.T}")
        print(f"{'='*50}")

        # Sample evaluation examples
        prompt_examples = random.sample(
            list(zip(sample_set_full, full_bd_id)), args.num_prompt)
        eval_samples = [s for s, c in prompt_examples]
        eval_classes = [c for s, c in prompt_examples]

        # ── Discriminator round: maximize D's ability to classify ──
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
            if (i + 1) % 4 == 0:
                print(f"  D sample {i+1}/{len(sample_set)} done")

        # ── Generator round: minimize G's loss (fool D better) ──
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
            if (i + 1) % 4 == 0:
                print(f"  G sample {i+1}/{len(sample_set)} done")

    # ── Save optimized results ──
    os.makedirs("results", exist_ok=True)
    g_path = f"results/G_optimized_{args.mode}.json"
    d_path = f"results/D_optimized_{args.mode}.json"

    with open(g_path, "w") as f:
        json.dump({"prompt": G.generator_prompt, "sample_set": G.sample_set},
                  f, indent=2, ensure_ascii=False)
    with open(d_path, "w") as f:
        json.dump({"prompt": D.discriminator_prompt, "sample_set": D.V},
                  f, indent=2, ensure_ascii=False)

    print(f"\nOptimized prompts saved:")
    print(f"  Generator:    {g_path}")
    print(f"  Discriminator: {d_path}")
    print(f"\nTo use the optimized prompt, run:")
    print(f"  python bd_minicar_textdesc_generation.py --api-key YOUR_KEY "
          f"--mode {args.mode} --use-optimized {g_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize minicar backdoor prompts via adversarial optimization"
    )
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--mode", type=str,
                        choices=["text", "image", "collaborative"], required=True)
    parser.add_argument("--num-of-sample", type=int, default=8,
                        help="Number of training samples for optimization")
    parser.add_argument("--num-prompt", type=int, default=5,
                        help="Number of evaluation samples per iteration")
    parser.add_argument("--bd-sample", type=int, default=4,
                        help="Number of poisoned samples (bd/total = ratio)")
    parser.add_argument("--T", type=int, default=3,
                        help="Number of optimization iterations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
