# Import libraries
from typing import Optional
from tools.logger import logging
from peft import prepare_model_for_kbit_training
from tools.prepare_data import craft_input, load_dataset
from tools.prompt import apply_template
from datasets import Dataset

from tools.modeling import get_hf_model_and_tokenizer, load_hf_peft_model, load_trainer, set_seed, get_quant_config

DELIM = "=============================================="

def run_train(
    run_name: str,
    model_name: str,
    data_path: str,
    chain_of_thought: bool = True,
    with_topic: bool = True,
    seed: int = 42,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    learning_rate: float = 3e-04,
    max_seq_len: int = 3072,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    num_epochs: int = 4,
    warmup_steps: int = 20,
    date_string: Optional[str] = None,
    replace_map: Optional[dict] = None
):
    input_parameters = locals()
    logging.info(f"{DELIM}\nInput parameters:\n{input_parameters}\n{DELIM}\n")
    
    logging.info(f"{DELIM}\nStep 1: setting seed={seed}\n{DELIM}\n")
    if seed is not None:
        set_seed(seed)

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

    logging.info(f"{DELIM}\nStep 3: loading quantized {model_name} model and tokenizer\n{DELIM}\n")
    model, tokenizer = get_hf_model_and_tokenizer(model_name=model_name, quantization_config=quantization_config)

    model = prepare_model_for_kbit_training(model)

    model = load_hf_peft_model(model, lora_r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    
    logging.info(f"{DELIM}\nStep 4: loading dataset from {data_path}\n{DELIM}\n")
    dataset = load_dataset(data_path)

    logging.info(f"Train data samples:\n{dataset['train'][:5]}\n\n")

    logging.info(f"Val data samples:\n{dataset['val'][:5]}\n\n")

    train_input = craft_input(dataset['train'], is_cot=chain_of_thought, with_topic=with_topic)
    val_input = craft_input(dataset['val'], is_cot=chain_of_thought, with_topic=with_topic)

    logging.info(f"Train input len: {len(train_input)}, validation len: {len(val_input)}")

    if date_string is None and 'llama-3.2' in model_name:
        raise ValueError("Llama-3.2 experiments without fixed date_string may produce non-deterministic results")

    train_dataset = apply_template(tokenizer, train_input, replace_map=replace_map, date_string=date_string)
    val_dataset = apply_template(tokenizer, val_input, replace_map=replace_map, date_string=date_string)

    # check that the prompt start does not contain repeated sequence start tokens
    row_idx = -1
    for _row in train_dataset["text"]:
        row_idx += 1
        tok_text = tokenizer(_row, return_tensors="pt", max_length=max_seq_len)
        decoded = tokenizer.decode(tok_text["input_ids"][0], skip_special_tokens=False)
        if decoded.count(f"{tokenizer.bos_token}{tokenizer.bos_token}") > 0:
            logging.info(f"\n\nTokens start: {tok_text['input_ids'][0][:25]}\nTokens end: {tok_text['input_ids'][0][-25:]}\nDecoded: {decoded}\n\n")
            raise ValueError(f"Prompt start contains repeated sequence start tokens: {decoded} in row {row_idx}")

    hf_train_dataset = Dataset.from_dict(train_dataset)
    hf_val_dataset = Dataset.from_dict(val_dataset)

    logging.info(hf_train_dataset)
    logging.info(hf_val_dataset)

    logging.info(hf_train_dataset["text"][:10])

    logging.info(hf_val_dataset["text"][:10])

    logging.info(f"{DELIM}\nStep 5: Model training\n{DELIM}\n")
    model.config.use_cache = False

    trainer = load_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        val_dataset=hf_val_dataset,
        run_name=run_name,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_acc=gradient_accumulation,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        seed=seed,
        full_determinism=True
    )

    # check dataloader does not contain repeated sequence start tokens
    data_loader = trainer.get_train_dataloader()
    row_idx = -1
    for row in data_loader:
        row_idx += 1
        decoded_input = tokenizer.decode(row['input_ids'][0], skip_special_tokens=False)
        repeated_bos = f"{tokenizer.bos_token}{tokenizer.bos_token}"
        start_idx = decoded_input.find(repeated_bos)
        if decoded_input.count(repeated_bos) > 0 or decoded_input.count(f"{tokenizer.bos_token}") == 0:
            logging.info(f"Idx of repeated sequence start tokens: {start_idx}")
            logging.info(f"Segment found: {decoded_input[max(0, start_idx-10):start_idx+50]}\n\n")
            logging.info(f"Number of repeated sequence start tokens: {decoded_input.count(repeated_bos)}")
            logging.info(f"\n\nTokens start: {row['input_ids'][0][:25]}\nTokens end: {row['input_ids'][0][-25:]}\nDecoded: {decoded_input}\n\n")
            raise ValueError(f"Data loader contains repeated or no sequence start tokens {repeated_bos} in row {row_idx}")

    return trainer.train()

    

