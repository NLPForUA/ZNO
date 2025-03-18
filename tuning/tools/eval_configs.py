def get_generation_config() -> dict:
    return {
        "temperature": 0.0,
        "max_new_tokens": 2048,
        "do_sample": False,
        "rep_penalty": 1.0,
    }

def get_eval_config(train_config: dict):
    eval_config = get_generation_config()
    eval_config.update({
        "base_model_name": train_config["model_name"],
        "data_path": train_config["data_path"],
        "chain_of_thought": train_config["chain_of_thought"],
        "with_topic": train_config.get("with_topic", False),
        "load_in_4bit": train_config["load_in_4bit"],
        "load_in_8bit": train_config["load_in_8bit"],
        "max_seq_len": train_config["max_seq_len"],
        "date_string": train_config.get("date_string", None),
        "replace_map": train_config.get("replace_map", None),
        "train_parameters": train_config,
    })
    return eval_config