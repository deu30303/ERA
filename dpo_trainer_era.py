# train_dpo_ds_quadrant.py
import os
import json
import math
import random
import itertools
from dataclasses import dataclass, field
from typing import List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOConfig

from tuner.datacollator import PreferenceCollator
from tuner.dpotrainer_era import Trainer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def prompt_template_llama3_base(ctxs, query: str) -> str:
    sys = (
        "You need to complete the question-and-answer pair. "
        "The answers should be short phrases or entities, not full sentences. "
        "If you don't know the answer and the following contexts do not contain "
        "the necessary information to answer the question, respond with "
        "'This question is beyond the scope of my knowledge and the references, I don't know the answer'."
    )
    sys_ctxs = "The following contexts will help you complete the question-and-answer pair."

    lines = []
    for i in range(min(3, len(ctxs))):
        lines.append(f"Context{i+1}: {ctxs[i].get('text','')}")
    contexts_block = "\n".join(lines)

    return (
        "### System\n"
        f"{sys}\n\n"
        "### Contexts\n"
        f"{sys_ctxs}\n{contexts_block}\n\n"
        "### Question\n"
        f"{query}\n\n"
        "### Answer\n"
    )


def map_type(t: str) -> int:
    return {"kg": 0, "kn": 1, "ug": 2, "un": 3}[t]


def data_sample(data_list, n):
    if len(data_list) >= n:
        return random.sample(data_list, n)
    return data_list + random.choices(data_list, k=n - len(data_list))


def dpo_dataset(script_args, tokenizer) -> Dataset:
    dataset = {"prompt": [], "chosen": [], "rejected": [], "label": []}

    if not script_args.cache_path:
        raise ValueError("cache_path is required")

    if os.path.exists(script_args.cache_path):
        print(f"Loading dataset from cache: {script_args.cache_path}")
        with open(script_args.cache_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return Dataset.from_dict(dataset)

    chosen_rejected_data = []
    subdirs = [
        d for d in os.listdir(script_args.data_dir)
        if os.path.isdir(os.path.join(script_args.data_dir, d))
    ]
    for sd in subdirs:
        fn = os.path.join(script_args.data_dir, sd, "dpo_data.json")
        with open(fn, "r", encoding="utf-8") as f:
            chosen_rejected_data.append(json.load(f))
    chosen_rejected_data = list(itertools.chain(*chosen_rejected_data))

    kg = [x for x in chosen_rejected_data if x["type"] == "kg"]
    kn = [x for x in chosen_rejected_data if x["type"] == "kn"]
    ug = [x for x in chosen_rejected_data if x["type"] == "ug"]
    un = [x for x in chosen_rejected_data if x["type"] == "un"]

    data_size = script_args.data_size
    idk_ratio = script_args.idk_ratio

    answerable = math.floor(data_size * (1 - idk_ratio) // 3)
    abstain = data_size - answerable * 3

    all_data = []
    all_data += data_sample(kg, answerable)
    all_data += data_sample(kn, answerable)
    all_data += data_sample(ug, answerable)
    all_data += data_sample(un, abstain)

    random.shuffle(all_data)

    for item in all_data:
        prompt = prompt_template_llama3_base(item["ctxs"], query=f"Question: {item['question']}")
        dataset["prompt"].append(prompt)
        dataset["chosen"].append(item["chosen"])
        dataset["rejected"].append(item["rejected"])
        dataset["label"].append(map_type(item["type"]))

    print(f"Saving dataset to cache: {script_args.cache_path}")
    os.makedirs(os.path.dirname(script_args.cache_path), exist_ok=True)
    with open(script_args.cache_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return Dataset.from_dict(dataset)


@dataclass
class ScriptArguments:
    # data
    data_dir: str = field(default="data_kbrag/Qwen3-8B-Base")
    cache_path: str = field(default="data_kbrag/Qwen3-8B-Base/training_data_ir0.7_d5k.json")
    data_size: int = field(default=5000)
    idk_ratio: float = field(default=0.7)

    # model
    model_name_or_path: str = field(default="Qwen/Qwen3-8B-Base")
    output_dir: str = field(default="./outputs")

    # precision / quant
    load_in_8bit: bool = field(default=True)
    load_in_4bit: bool = field(default=False)
    model_dtype: str = field(default="bfloat16")

    # lora
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)

    # lengths
    max_prompt_length: int = field(default=1024)
    max_length: int = field(default=1024)

    # training
    beta: float = field(default=0.1)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=2)
    num_train_epochs: int = field(default=1)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    seed: int = field(default=0)

    # DTA aux
    aux_loss: str = field(default="mix")  # none/sft/cls/mix
    coe_cls: float = field(default=0.5)
    coe_sft: float = field(default=1.0)
    quadrant_num: int = field(default=4)

    # ✅ DS / evidential quadrant options
    use_ds: bool = field(default=True)
    gamma_ds: float = field(default=1.0)
    kappa_clip: float = field(default=0.999)
    default_discount: bool = field(default=True)

    use_evidential_quadrant_heads: bool = field(default=True)
    evidential_dropout: float = field(default=0.0)
    share_quadrant_head: bool = field(default=False)


    derive_param_prompt_from_prompt: bool = field(default=True)


    report_to: str = field(default="none")


def resolve_dtype(s: str):
    s = (s or "").lower()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  

    # quantization config
    torch_dtype = resolve_dtype(args.model_dtype)
    quant_config = None
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one: load_in_4bit or load_in_8bit")

    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="sdpa",
        quantization_config=quant_config,
    )
    model.config.use_cache = False

    # dataset
    total = dpo_dataset(args, tokenizer)
    split = total.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset, eval_dataset = split["train"], split["test"]

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # DPOConfig
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=1,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        remove_unused_columns=False,
        report_to=[],
        generate_during_eval=False,
        beta=args.beta,
    )


    setattr(training_args, "max_prompt_length", args.max_prompt_length)
    max_completion = max(32, args.max_length - args.max_prompt_length)
    setattr(training_args, "max_completion_length", max_completion)
    setattr(training_args, "max_length", args.max_length)

    collator = PreferenceCollator(pad_token_id=tokenizer.pad_token_id)

    # trainer
    dpo_trainer = Trainer(
        # DTA aux
        aux_loss=args.aux_loss,
        coe_cls=args.coe_cls,
        coe_sft=args.coe_sft,
        quadrant_num=args.quadrant_num,

        # DS / evidential quadrant
        use_ds=args.use_ds,
        gamma_ds=args.gamma_ds,
        kappa_clip=args.kappa_clip,
        default_discount=args.default_discount,
        use_evidential_quadrant_heads=args.use_evidential_quadrant_heads,
        evidential_dropout=args.evidential_dropout,
        share_quadrant_head=args.share_quadrant_head,
        derive_param_prompt_from_prompt=args.derive_param_prompt_from_prompt,

        data_collator=collator,
        model=model,
        ref_model=None,               
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,     
        peft_config=peft_config,
    )

    # train
    dpo_trainer.train()

    # save merged
    out_dir = os.path.join(args.output_dir, "Qwen_final_checkpoint_ERA")
    os.makedirs(out_dir, exist_ok=True)
    merged = dpo_trainer.model.merge_and_unload()
    merged.save_pretrained(out_dir, max_shard_size="16GB", safe_serialization=False)
    tokenizer.save_pretrained(out_dir)

    print("Saved to:", out_dir)
