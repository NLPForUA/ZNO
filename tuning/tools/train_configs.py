def get_model_config() -> dict:
    return {
        "seed": 42,
        "load_in_4bit": True,
        "load_in_8bit": False,
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 3e-04,
        "max_seq_len": 3072,
        "num_epochs": 4,
        "warmup_steps": 20,
        "chain_of_thought": False,
    }

def get_llama_3_1_8B_configs():
    model_config = get_model_config()
    return {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size": 4,
        "gradient_accumulation": 4,
        "replace_map": {"<|begin_of_text|>": ""},
        **model_config,
    }

def get_llama_3_1_8B_cot_configs():
    model_config = get_llama_3_1_8B_configs()
    model_config["chain_of_thought"] = True
    model_config["with_topic"] = True
    return model_config

def get_llama_3_1_8B_cot_no_topic_configs():
    model_config = get_llama_3_1_8B_configs()
    model_config["chain_of_thought"] = True
    model_config["with_topic"] = False
    return model_config

def get_llama_3_2_3B_configs():
    model_config = get_model_config()
    return {
        "model_name": "unsloth/Llama-3.2-3B-Instruct",
        "batch_size": 8,
        "gradient_accumulation": 4,
        "date_string": "07 Feb 2025",
        "replace_map": {"<|begin_of_text|>": ""},
        **model_config,
    }

def get_llama_3_2_3B_cot_configs():
    model_config = get_llama_3_2_3B_configs()
    model_config["chain_of_thought"] = True
    model_config["with_topic"] = True
    return model_config

def get_llama_3_2_3B_cot_no_topic_configs():
    model_config = get_llama_3_2_3B_configs()
    model_config["chain_of_thought"] = True
    model_config["with_topic"] = False
    return model_config

def get_gemma_2_9B_configs():
    model_config = get_model_config()
    return {
        "model_name": "google/gemma-2-9b-it",
        "batch_size": 4,
        "gradient_accumulation": 4,
        **model_config,
    }

def get_gemma_2_9B_cot_configs():
    model_config = get_gemma_2_9B_configs()
    model_config["chain_of_thought"] = True
    model_config["with_topic"] = True
    return model_config

def get_gemma_2_9B_cot_no_topic_configs():
    model_config = get_gemma_2_9B_configs()
    model_config["chain_of_thought"] = True
    model_config["with_topic"] = False
    return model_config
