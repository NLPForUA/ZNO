import json
import time
import torch
import numpy
import random
from trl import SFTTrainer
from transformers import TrainingArguments, BitsAndBytesConfig
from tools.loader import FastLanguageModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model

from tools.logger import logging

MAX_SEQ_LEN = 3072
SEED = 42

def set_seed(seed:int = SEED):
    logging.info("Setting seed={seed}")
    # set seed for all possible avenues of stochasticity
    numpy.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_quant_config(load_in_4bit: bool = True, load_in_8bit: bool = False):
    # ---------------------------
    # Set up BitsAndBytes quantization configuration
    # ---------------------------
    if load_in_4bit and load_in_8bit:
        raise ValueError("Enabled both 4 and 8 bit quantization")
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,      # computation in fp16
            bnb_4bit_use_double_quant=True,              # enables double quantization for better accuracy
            bnb_4bit_quant_type="nf4"                    # choose "nf4" (normal float4) or other types as supported
    )
    elif load_in_8bit == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError("bit_mode must be either 4 or 8.")
    return quantization_config


def get_hf_model_and_tokenizer(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        quantization_config=None,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.float16,
        flash_attention=False,
        padding_side="right"
):
    logging.info(f"Loading {model_name} model and tokenizer with max_seq_len={max_seq_length}, dtype={dtype} and flash_attention={flash_attention}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, max_sequence_length=max_seq_length, model_max_length=MAX_SEQ_LEN, padding_side=padding_side, add_eos_token=False, add_bos_token=False)
    if model_name in ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]:
        logging.info(f"Setting pad token {tokenizer.pad_token} to <|finetune_right_pad_id|> for llama-based model {model_name}")
        tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # automatically place model on available GPU(s),
        torch_dtype=dtype,
        use_flash_attention_2=flash_attention,
    )
    logging.info(f"\nModel config\n{model}\n\nTokenizer\n{tokenizer}")
    return model, tokenizer


def get_unsloth_model(model_name="unsloth/Meta-Llama-3.1-8B-Instruct", max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=torch.float16, flash_attention=False):
    logging.info(f"Loading {model_name} with max_seq_len={max_seq_length}, 4bit={load_in_4bit} with dtype={dtype} and flash_attention={flash_attention}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # unsloth/Meta-Llama-3.1-8B-bnb-4bit unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=dtype, # None before gp16 attempts (21-06-55 02022025)
        use_flash_attention_2=flash_attention
    )
    logging.info(model.config)
    return model, tokenizer


def load_hf_peft_model(model, lora_r=16, lora_alpha=16, lora_dropout=0.05):
    logging.info(f"Preparing LoRA with r={lora_r}, lora_alpha={lora_alpha}, and lora_dropout={lora_dropout}")
    lora_config = LoraConfig(
        r=lora_r,                          # LoRA rank
        lora_alpha=lora_alpha,        # scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],  # update these based on your model’s architecture
        lora_dropout=lora_dropout,    # dropout probability
        bias="none",                  # usually "none" or "all"
        task_type="CAUSAL_LM",
    )
    logging.info(f"Crafting peft model with LoRA:\n{lora_config}\n\n")
    model = get_peft_model(model, lora_config)
    logging.info(model.config)
    logging.info("\n\n")
    print_trainable_parameters(model)
    return model


def load_unsloth_peft_model(model, r=16, lora_alpha=16, lora_dropout=0, seed=SEED):
    model = FastLanguageModel.get_peft_model(
        model,
        r=r, # 16
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
        use_rslora=False,
        use_gradient_checkpointing="unsloth",
        random_state=seed
    )
    logging.info(model.config)
    return model


def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {trainable_params} | Total params: {total_params} | ({100 * trainable_params / total_params:.2f}%)\n\n")


def load_trainer(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        run_name,
        max_seq_len=MAX_SEQ_LEN,
        learning_rate=3e-4,
        batch_size=32,
        grad_acc=1,
        num_epochs=4,
        warmup_steps=10,
        seed=SEED,
        full_determinism=True
):
    run_name = f"{run_name}_lr{learning_rate}_ep{num_epochs}_msl{max_seq_len}_bs{batch_size}_ga{grad_acc}_ws{warmup_steps}_s{seed}_fd{full_determinism}_{time.time()}"
    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        dataset_num_proc=1,
        packing=False, # True before (12:50 of Sunday 5th
        args=TrainingArguments(
            learning_rate=learning_rate,  # 3e-04
            lr_scheduler_type="linear",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size // 2,
            gradient_accumulation_steps=grad_acc,
            num_train_epochs=num_epochs,
            eval_strategy="epoch",
            #eval_steps=5,
            save_strategy="epoch",
            fp16=True, # not is_bfloat16_supported(),
            bf16=False, # is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=warmup_steps,  # 3 before (12:50 of Sunday 5th) 
            output_dir=run_name, #"models_447_2511/16bit_llama-3.1-instruct-l16-cot-no-system-message-fixed-ukr-clean-no-pack-ws10-4ep-bs32-ga1-lr3e04-a100-02022025",
            report_to = "wandb",
            load_best_model_at_end = True,
            seed=seed,
            full_determinism=full_determinism,
        ),
    )
    logging.info(trainer.args.to_dict())
    logging.info(f"\n\nTraining {run_name}\n\n")
    return trainer


def predict(
        model_path,
        test_set,
        exp_link,
        trainer_config,
        output_path="/workspace/deterministic/test_preds/",
        max_seq_len=MAX_SEQ_LEN,
        dtype=None,
        load_in_16bit=False,
        temps=[0.0],
        do_sample=False,
        rep_penalty=1,
        use_cache=True,
        max_new_tokens=2048
    ):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_len,
        dtype = dtype,
        load_in_4bit=not load_in_16bit,
        load_in_8bit=False,
    )
    model = FastLanguageModel.for_inference(model)
    logging.info(model.config)

    all_evals = {}

    for temp in temps:
        logging.info(f"Started temperature {temp}\n")
        test_preds = []

        run_name = model_path.split('/')[-1]
        out_file = f"{output_path}{run_name}_temp_{temp}_mt_2048-rp_1.json"

        configs = {
            "model_path": model_path,
            "dtype": dtype,
            "load_in_16bit": load_in_16bit,
            "model_config": model.config.to_dict(),
            "experiment_link": exp_link,
            "trainer_args": trainer_config,
            "max_sequence_length": max_seq_len,
            "temperature": temp,
            "do_sample": do_sample,
            "rep_penalty": rep_penalty,
            "use_cache": use_cache,
            "max_new_tokens": max_new_tokens
        }

        for ridx, row in enumerate(test_set):
            messages = [
                {"role": "user", "content": row[0]},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")

            pred = tokenizer.decode(model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, use_cache=use_cache, temperature=temp, do_sample=do_sample, repetition_penalty=rep_penalty)[0])
            test_preds.append({"input": messages, "pred": pred})
            if ridx and ridx % 10 == 0:
                logging.info(f"Processed {ridx} rows for temp {temp}")

        eval_res = {"configs": configs, "test_preds": test_preds}
        all_evals[temp] = eval_res
        with open(out_file, 'w') as fw:
            json.dump(eval_res, fw)
        logging.info(f"Finished iteration {temp} and wrote result to {out_file}")

    return all_evals