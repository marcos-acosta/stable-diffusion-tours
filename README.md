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
{
  "minecraft": {
    # Place args to Diffusioner() here
    "preprocessor": "canny",
    "model": "canny"
  },
  "lego": {
    "preprocessor": "depth_midas",
    "model": "depth"
  }
}
```

Prompts are set on a by-frame basis to account for changes in the input video. They are configured in a JSON file as follows:

```
# prompts.json
{
  "house_tour": [
    {
      "prompt": "The window of a house rendered in Minecraft",
      "until": 60
    }, {
      "prompt": "The door of a house rendered in Minecraft",
      "until": 68
    },
    ...
    {
      "prompt": "The staircase of a house rendered in Minecraft",
      "until": null
    }
  ]
}
```

The use of `null` indicates "until the end of the video". Thus, a single prompt can be used for the entire video by simply setting the only prompt's `until` property to `null`.

Then, the following scripts can be run:

```
# parameters: [input video filename] [input frame directory] [target FPS]
bash sd_tours/scripts/split.sh greg.mov greg_10fps 10

# parameters: [stable diffusion API endpoint id] [input frame directory] [diffusioned frame directory] [prompts name] [config name]
bash sd_tours/scripts/diffusionify.sh 741d7442ca2750cb1f greg_10fps greg_10fps_minecraft greg minecraft

# parameters: [diffusioned frame directory] [original frame directory] [target FPS]
bash sd_tours/scripts/stitch.sh greg_10fps_minecraft greg_10fps 10
```

## Running in real-time

To quickly run stable diffusion in realtime, you may use the shortcut script:

```
# parameters: [stable diffusion API endpoint id] [config name] [camera id]
bash sd_tours/scripts/realtime.sh 741d7442ca2750cb1f minecraft "A scene rendered in Minecraft" 0
```
