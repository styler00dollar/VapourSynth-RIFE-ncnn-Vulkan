# RIFE
[![CI](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan/actions/workflows/CI.yml/badge.svg)](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan/actions/workflows/CI.yml)

Real-Time Intermediate Flow Estimation for Video Frame Interpolation, based on [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan).


## Usage
    rife.RIFE(vnode clip[, int model=5, int factor_num=2, int factor_den=1, int fps_num=None, int fps_den=None, string model_path=None, int gpu_id=None, int gpu_thread=2, bint tta=False, bint uhd=False, bint sc=False, bint skip=False, float skip_threshold=60.0, bint list_gpu=False])

- clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

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
  - 9 = rife-v4

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
