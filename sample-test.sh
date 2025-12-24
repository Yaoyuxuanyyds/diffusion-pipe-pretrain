#!/bin/bash

### -----------------------------
### Basic Settings
### -----------------------------
export CUDA_VISIBLE_DEVICES=0

MODEL="sd3"
NFE=50
CFG=0.0
IMGSIZE=256
BATCHSIZE=1


SAVEDIR="/inspire/hdd/project/chineseculture/public/yuxuan/SD3-Light-Eval/logs/generate/test"




python validate_sd3_light.py \
  --model_dir "/inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/outputs/sd3_light_pretrain/base-mmdit/20251223_17-52-21/step10000" \
  --prompt "a photo of a cute cat" \
  --height $IMGSIZE \
  --width $IMGSIZE \
  --steps $NFE \
  --seed 1234 \
  --dtype float16 \
  --output_dir "$SAVEDIR" \
  --ema_shadow "/inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/outputs/sd3_light_pretrain/base-mmdit/20251223_17-52-21/step10000/ema_shadow.pt"




# A woman holding a Hello Kitty phone on her hands.

# the word'START'written inchalk on asidewalk

# A mirror that tracks your daily health metrics.

# three black cats standing next to two orange cats.

# a spaceship that looks like the Sydney OperaHouse.

# Generate an image of an animal with (3 + 6) lives.

# A cat holding a sign that says hello world

# A picture of a bookshelf with some books on it. The bottom shelf is empty.

# A man holds a letter stamped “Accepted” from his dream university, with his emotional response clearly visible.
