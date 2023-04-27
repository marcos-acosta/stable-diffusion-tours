from diffusionify import Diffusioner
from config import load_config, get_correct_setting

import argparse
import realtime


PROMPT = "The interior of a miniature building made of legos, the inside of a lego house"
SEED = 42
N_STEPS = 20
DEPTH_MIDAS_PREPROCESSOR_NAME = "depth_midas"
DEPTH_LERES_PREPROCESSOR_NAME = "depth_leres"
DEPTH_ZOE_PREPROCESSOR_NAME = "depth_zoe"
CANNY_PREPROCESSOR_NAME = "canny"
DEPTH_MODEL_NAME = "control_depth-fp16 [400750f6]"
CANNY_MODEL_NAME = "control_canny-fp16 [e3fe7712]"
EULER_A_SAMPLER_NAME = "Euler a"
WEIGHT = 1.0

MODEL_NAMES = {
    'depth': DEPTH_MODEL_NAME,
    'canny': CANNY_MODEL_NAME
}

PREPROCESSOR_NAMES = {
    'depth_midas': DEPTH_MIDAS_PREPROCESSOR_NAME,
    'depth_leres': DEPTH_LERES_PREPROCESSOR_NAME,
    'depth_zoe': DEPTH_ZOE_PREPROCESSOR_NAME,
    'canny': CANNY_PREPROCESSOR_NAME
}


parser = argparse.ArgumentParser()
parser.add_argument("txt2img_endpoint_id", type=str, help="Stable difussion UI API endpoint ID")
parser.add_argument('--config_path', type=str, default="configs.json", help="Path to a config JSON file")
parser.add_argument("--config", "-c", type=str, help="Name of the config to load from --config_path")
parser.add_argument("--prompt", type=str, help="Prompt to pass to the stable diffusion model")
parser.add_argument("--input_dir", type=str, help="Path to a directory of images to diffusionify")
parser.add_argument("--output_dir", type=str, help="Output directory where diffusionified images will be written to")
parser.add_argument("--preprocessor", type=str, default='canny', help=f"Which ControlNet preprocessor to use. Must be one of [{', '.join(PREPROCESSOR_NAMES.keys())}]")
parser.add_argument("--model", type=str, default='canny', help=f"Which ControlNet preprocessor to use. Must be one of [{', '.join(MODEL_NAMES.keys())}]")
parser.add_argument("--seed", "-s", type=int, default=-1, help="Seed to pass to the stable diffusion model")
parser.add_argument("--n_steps", "-n", type=int, default=20, help="Number of stable diffusion steps")
parser.add_argument("--sampler", type=str, default=EULER_A_SAMPLER_NAME, help="Name of the sampler to pass to the stable diffusion model")
parser.add_argument("--weight", "-w", type=float, default=1.0, help="ControlNet weight")
parser.add_argument("--realtime", "-r", action="store_true", help="If set, run stable diffusion real-time from camera")
parser.add_argument("--camera_id", type=int, default=0, help="The camera index (when --realtime is set)")
args = parser.parse_args()

if args.config:
    saved_config = load_config(args.config_path, args.config)['args']

diffusioner_settings = {
    "txt2img_endpoint": f"https://{args.txt2img_endpoint_id}.gradio.live/sdapi/v1/txt2img/",
    "prompt": get_correct_setting(args, saved_config, "prompt"),
    "preprocessor_name": PREPROCESSOR_NAMES[get_correct_setting(args, saved_config, "preprocessor")],
    "model_name": MODEL_NAMES[get_correct_setting(args, saved_config, "model")],
    "seed": get_correct_setting(args, saved_config, "seed"),
    "n_steps": get_correct_setting(args, saved_config, "n_steps"),
    "sampler_name": get_correct_setting(args, saved_config, "sampler"),
    "batch_size": 1,
    "weight": get_correct_setting(args, saved_config, "weight"),
}

d = Diffusioner(**diffusioner_settings)

if args.realtime:
    realtime.run_realtime(
        diffusioner=d,
        camera_id=args.camera_id
    )
else:
    if not args.input_dir or not args.output_dir:
        raise Exception("If --realtime is not set, --input_dir and --output_dir must be set")
    d.diffusionify_dir(args.input_dir, args.output_dir)
