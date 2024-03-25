# RIFE

<p align="center">
    <a href="https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2Fstyler00dollar%2FVapourSynth-RIFE-ncnn-Vulkan%2F"><img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fstyler00dollar%2FVapourSynth-RIFE-ncnn-Vulkan%2F&labelColor=%23697689&countColor=%23ff8a65&style=plastic&labelStyle=none" /></a> 
    <a href="https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan.svg?style=flat-square" /></a>
    <a href="https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan/releases"><img alt="GitHub All Releases" src="https://img.shields.io/github/downloads/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan/total.svg?style=flat-square&color=%2364ff82" /></a>
    <a href="https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan/commits"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan.svg?style=flat-square" /></a>
</p>

Real-Time Intermediate Flow Estimation for Video Frame Interpolation, based on [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan).


## Usage
    rife.RIFE(vnode clip[, int model=5, int factor_num=2, int factor_den=1, int fps_num=None, int fps_den=None, string model_path=None, int gpu_id=None, int gpu_thread=2, bint tta=False, bint uhd=False, bint sc=False, bint skip=False, float skip_threshold=60.0, bint list_gpu=False])

- clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

The `models` folder needs to be in the same folder as the compiled binary.

By default models are exported with ensemble=False and Fast=True

- model: Model to use.
  - 0 = rife
  - 1 = rife-HD
  - 2 = rife-UHD
  - 3 = rife-anime
  - 4 = rife-v2
  - 5 = rife-v2.3
  - 6 = rife-v2.4
  - 7 = rife-v3.0
  - 8 = rife-v3.1
  - 9 = rife-v3.9 (ensemble=False / fast=True)
  - 10 = rife-v3.9 (ensemble=True / fast=False)
  - 11 = rife-v4 (ensemble=False / fast=True)
  - 12 = rife-v4 (ensemble=True / fast=False)
  - 13 = [rife-v4.1](https://github.com/mirrorsysu/rife-ncnn-vulkan/tree/model_4_1) (ensemble=False / fast=True)
  - 14 = rife-v4.1 (ensemble=True / fast=False)
  - 15 = rife-v4.2 (ensemble=False / fast=True)
  - 16 = rife-v4.2 (ensemble=True / fast=False)
  - 17 = rife-v4.3 (ensemble=False / fast=True)
  - 18 = rife-v4.3 (ensemble=True / fast=False)
  - 19 = rife-v4.4 (ensemble=False / fast=True)
  - 20 = rife-v4.4 (ensemble=True / fast=False)
  - 21 = rife-v4.5 (ensemble=False)
  - 22 = rife-v4.5 (ensemble=True)
  - 23 = rife-v4.6 (ensemble=False)
  - 24 = rife-v4.6 (ensemble=True)
  - 25 = rife-v4.7 (ensemble=False)
  - 26 = rife-v4.7 (ensemble=True)
  - 27 = rife-v4.8 (ensemble=False)
  - 28 = rife-v4.8 (ensemble=True)
  - 29 = rife-v4.9 (ensemble=False)
  - 30 = rife-v4.9 (ensemble=True)
  - 31 = rife-v4.10 (ensemble=False)
  - 32 = rife-v4.10 (ensemble=True)
  - 33 = rife-v4.11 (ensemble=False)
  - 34 = rife-v4.11 (ensemble=True)
  - 35 = rife-v4.12 (ensemble=False)
  - 36 = rife-v4.12 (ensemble=True)
  - 37 = rife-v4.12-light (ensemble=False)
  - 38 = rife-v4.12-light (ensemble=True)
  - 39 = rife-v4.13 (ensemble=False)
  - 40 = rife-v4.13 (ensemble=True)
  - 41 = rife-v4.13-lite (ensemble=False)
  - 42 = rife-v4.13-lite (ensemble=True)
  - 43 = rife-v4.14 (ensemble=False)
  - 44 = rife-v4.14 (ensemble=True)
  - 45 = rife-v4.14-lite (ensemble=False)
  - 46 = rife-v4.14-lite (ensemble=True)
  - 47 = rife-v4.15 (ensemble=False)
  - 48 = rife-v4.15 (ensemble=True)
  - 49 = rife-v4.16-lite (ensemble=False)
  - 50 = rife-v4.16-lite (ensemble=True)

  ## My experimental custom models (only works with 2x)

  - 51 = sudo_rife4 (ensemble=False / fast=True)
  - 52 = sudo_rife4 (ensemble=True / fast=False)
  - 53 = sudo_rife4 (ensemble=True / fast=True)

- factor_num, factor_den: Factor of target frame rate. For example `factor_num=5, factor_den=2` will multiply input clip FPS by 2.5. Only rife-v4 model supports custom frame rate.

- fps_num, fps_den: Target frame rate. Only rife-v4 model supports custom frame rate. Supersedes `factor_num`/`factor_den` parameter if specified.

- model_path: RIFE model path. Supersedes `model` parameter if specified.

- gpu_id: GPU device to use.

- gpu_thread: Thread count for interpolation. Using larger values may increase GPU usage and consume more GPU memory. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.

- tta: Enable TTA(Test-Time Augmentation) mode.

- uhd: Enable UHD mode.

- sc: Avoid interpolating frames over scene changes. You must invoke `misc.SCDetect` on YUV or Gray format of the input beforehand so as to set frame properties.

- skip: Skip interpolating static frames. Requires [VMAF](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF) plugin.

- skip_threshold: PSNR threshold to determine whether the current frame and the next one are static.

- list_gpu: Simply print a list of available GPU devices on the frame and does no interpolation.

## Compilation

Requires `Vulkan SDK`.

```
git submodule update --init --recursive --depth 1
meson build
ninja -C build
ninja -C build install
```
