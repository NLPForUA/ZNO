import json
import os
import time
from typing import Optional
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Import libraries
from tools.logger import logging
from tools.prepare_data import craft_input, hash_dataset, load_dataset

from tools.modeling import get_quant_config

DELIM = "=============================================="

def run_eval(
    base_model_name: str,
    data_path: str,
    out_file_path: str,
    adapter_path: Optional[str] = None,
    merged_model_path: Optional[str] = None,
    merge_type: str = "quantized",
    chain_of_thought: bool = True,
    with_topic: bool = True,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    max_seq_len: int = 3072,
    split="test",
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    do_sample: bool = False,
    rep_penalty: float = 1.0,
    experiment_link: Optional[str] = None,
    date_string: Optional[str] = None,
    sample_size: Optional[int] = None,
    train_parameters: Optional[dict] = None,
    replace_map: Optional[dict] = None
):
    input_parameters = locals()
    logging.info(f"{DELIM}\nInput parameters:\n{input_parameters}\n{DELIM}\n")

    if adapter_path is None and merged_model_path is None:
        raise ValueError("Either adapter_path or merged_model_path must be provided")
    
    if adapter_path is not None and merged_model_path is not None:
        logging.warning("Both adapter_path and merged_model_path are provided. Using merged_model_path")

    bit_mode = 16
    if load_in_4bit:
        bit_mode = 4
    elif load_in_8bit:
        bit_mode = 8
    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of load_in_4bit or load_in_8bit can be set to True")
    
    logging.info(f"{DELIM}\nStep 2: building quantization config for {bit_mode} bit mode\n{DELIM}\n")
    quantization_config = get_quant_config(load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
    if load_in_4bit:
        assert quantization_config.load_in_4bit == True, "Failed to build 4 bit quantization config"
    elif load_in_8bit:
        assert quantization_config.load_in_8bit == True, "Failed to build 8 bit quantization config"

    logging.info(f"Quantization config:\n{quantization_config}")

    logging.info(f"{DELIM}\nStep 3: loading PEFT model based on {base_model_name}\n{DELIM}\n")
    logging.info(f"\n\nLoading {base_model_name} tokenizer\n\n")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, max_sequence_length=max_seq_len, model_max_length=max_seq_len)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Pad token: {tokenizer.pad_token}, Eos token: {tokenizer.eos_token}")

    if merged_model_path is None and adapter_path is not None and merge_type == "quantized":
        logging.info(f"Loading base model {base_model_name}")
        model_base = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=False)
        logging.info(f"Loading adapter model {adapter_path}")
        model = PeftModel.from_pretrained(model_base, adapter_path, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=False)

        logging.info(f"Loaded model config: {model.config}")
    
    elif merged_model_path is None and adapter_path is not None and merge_type == "full_precision":
        logging.info(f"Loading base {base_model_name} model with no quant")
        # Load the unquantized base model.
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",  # or specify your device settings
            #torch_dtype=torch.float16,
            #use_flash_attention_2=False
        )
        # Load the trained LoRA adapter into the model.
        logging.info(f"Loading trained lora adapter {adapter_path}")
        lora_adapter = PeftModel.from_pretrained(
            model_base,
            adapter_path,
            #torch_dtype=torch.float16,
            #use_flash_attention_2=False
        )
        logging.info("Merging base model with trained lora adapter")
        merged_model = lora_adapter.merge_and_unload()
        logging.info("Saving model")
        path_to_save = os.path.join(adapter_path, f"merged_model_full_precision_{time.strftime('%Y%m%d%H%M%S')}")
        merged_model.save_pretrained(path_to_save)
        del merged_model
        del model_base
        del lora_adapter

        logging.info(f"Loading final model from {path_to_save}")
        model = AutoModelForCausalLM.from_pretrained(path_to_save, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=False)

    
    elif merged_model_path is not None:
        logging.info(f"Loading merged model from {merged_model_path}")
        model = AutoModelForCausalLM.from_pretrained(merged_model_path, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=False)
        logging.info(f"Loaded model config: {model.config}")


    logging.info(f"{DELIM}\nStep 4: loading dataset from {data_path}\n{DELIM}\n")
    dataset = load_dataset(data_path)

    input_parameters["dataset_hash"] = hash_dataset(dataset)

    logging.info(f"Train data samples:\n{dataset['train'][:5]}\n\n")

    logging.info(f"Val data samples:\n{dataset['val'][:5]}\n\n")

    train_input = craft_input(dataset['train'], is_cot=chain_of_thought, with_topic=with_topic)
    val_input = craft_input(dataset['val'], is_cot=chain_of_thought, with_topic=with_topic)
    test_input = craft_input(dataset['test'], is_cot=chain_of_thought, with_topic=with_topic)

    logging.info(f"Train input len: {len(train_input)}, validation len: {len(val_input)}, test len: {len(test_input)}")

    predict_dataset = test_input
    if split == "val":
        logging.info(f"Using validation set for prediction")
        predict_dataset = val_input
    elif split == "train":
        logging.info(f"Using training set for prediction")
        predict_dataset = train_input
    else:
        logging.info(f"Using test set for prediction")

    logging.info(f"{DELIM}\nStep 5: Predicting\n{DELIM}\n")
    model.config.use_cache = False
    
    override_pad_token_id = None
    if "llama" in base_model_name.lower():
        override_pad_token_id = 128009 
        logging.info(f"Setting pad token id: {override_pad_token_id}")
    override_eos_token_id = None
    if "gemma" in base_model_name.lower():
        override_eos_token_id = [1, 107]
        logging.info(f"Setting eos token id: {override_eos_token_id}")
    

    with torch.no_grad():
        logging.info(f"Started temperature {temperature}\n")
        predictions = []

        if sample_size is not None:
            logging.info(f"Sampling {sample_size} rows from the dataset")
            predict_dataset = predict_dataset[:sample_size]

        for ridx, row in enumerate(predict_dataset):
            messages = [
                #{"role": "system", "content": "You are a highly intelligent assistant taking the graduation exam in a Ukrainian school"},
                {"role": "user", "content": row[0]},
            ]
            additional_template_args = {}
            if date_string is not None:
                additional_template_args["date_string"] = date_string
            logging.info(f"Additional template args: {additional_template_args}")
            
            additional_generate_args = {}
            if override_pad_token_id is not None:
                additional_generate_args["pad_token_id"] = override_pad_token_id
            if override_eos_token_id is not None:
                additional_generate_args["eos_token_id"] = override_eos_token_id
            logging.info(f"Additional generate args: {additional_generate_args}")

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                add_special_tokens=False,
                **additional_template_args
            ).to("cuda")

            pred = tokenizer.decode(model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                do_sample=do_sample,
                repetition_penalty=rep_penalty,
                #pad_token_id=pad_token_id,
                **additional_generate_args
            )[0])
            double_bos = f"{tokenizer.bos_token}{tokenizer.bos_token}"
            assert pred.count(double_bos) == 0, f"Prompt start contains repeated sequence start tokens {double_bos}: {pred}"
            predictions.append({"input": messages, "pred": pred})
            if ridx and ridx % 5 == 0:
                logging.info(f"Processed {ridx} rows for temp {temperature}\n")
                logging.info(f"\n{pred}\n\n")

        eval_res = {"configs": input_parameters, "test_preds": predictions}
        logging.info(f"Writting result to {out_file_path}")
        try:
            with open(out_file_path, 'w') as fw:
                json.dump(eval_res, fw)
                logging.info(f"Finished and wrote result to {out_file_path}")
        except Exception as e:
            logging.error(f"Failed to write result to {out_file_path}")
            logging.error(e)

    return eval_res