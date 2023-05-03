import argparse
import json


def load_config(prompt_path: str, prompt_name: str):
    with open(prompt_path, "r") as f:
        prompts = json.loads(f.read())
    return prompts[prompt_name]
    

def get_correct_setting(args: argparse.Namespace, config: dict, key: str):
    args_dict = vars(args)
    return args_dict[key] if key not in config else config[key]