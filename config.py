import argparse
import json

def load_config(config_path: str, config_name: str):
    with open(config_path, "r") as f:
        configs = json.loads(f.read())
    matching_configs = [config for config in configs if config["name"] == config_name]
    if len(matching_configs) == 0:
        raise Exception(f"Could not find config `{config_name}` in {config_path}.")
    else:
        return matching_configs[0]
    

def get_correct_setting(args: argparse.Namespace, config: dict, key: str):
    args_dict = vars(args)
    return args_dict[key] if key not in config else config[key]