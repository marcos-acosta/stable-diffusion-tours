# Stable Diffusion Tours

[Video demo!](https://www.youtube.com/watch?v=QsQevePhlbY)

This repository contains code for running stable diffusion on videos (saved and real-time).

## Installation

To install the python dependencies, run `install.sh`.

## Running the notebook

Import `controlnet.ipynb` into Google Colab and `Run All`. The last output will contain a public url like `https://2fa342ca61dffb9105.gradio.live/`. The alphanumeric id will be used later in the workflow.

## Video processing workflow

It may be convenient to maintain four directories:

```
|_ input_videos/
|_ input_frames/
|_ output_frames/
|_ output_videos/
```

Place an input video in `input_videos/` e.g. `house_tour.mov`. Then, create a JSON configuration for stable diffusion parameters. The format of the config is as follows:

```
# configs.json
[
  {
    "name": "minecraft",
    "args": {
      # place args to diffusionify.Diffusioner() here
      "prompt": "The interior of a Minecraft house",
      "preprocessor": "depth_midas",
      "model": "depth",
      "seed": 42
    }
  }
]
```

Then, the following scripts can be run:

```
# parameters: [video name (stem)] [target FPS]
bash sd_tours/scripts/split.sh house_tour 10

# parameters: [stable diffusion API endpoint id] [video name (stem)] [config name]
bash sd_tours/scripts/diffusionify.sh 5bbeded4c4c23e44c5 house_tour minecraft

# parameters: [video name (stem)] [target FPS]
bash sd_tours/scripts/stitch.sh house_tour 10
```

## Running in real-time

To quickly run stable diffusion in realtime, you may use the shortcut script:

```
# parameters: [stable diffusion API endpoint id] [config name] [camera id]
bash sd_tours/scripts/realtime.sh 5bbeded4c4c23e44c5 minecraft 0
```
