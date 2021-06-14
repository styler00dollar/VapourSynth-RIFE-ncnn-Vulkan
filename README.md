Description
===========
[![CI](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan/actions/workflows/CI.yml/badge.svg)](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan/actions/workflows/CI.yml)

RIFE filter for VapourSynth, based on [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan).


Usage
=====

    rife.RIFE(clip clip[, int model=0, int gpu_id=auto, int gpu_thread=2, bint tta=False, bint uhd=False, bint sc=False, bint list_gpu=False])

* clip: Clip to process. Only planar RGB format with float sample type of 32 bit depth is supported.

* model: Model to use.
  * 0 = rife-v3.1
  * 1 = rife-v2.4
  * 2 = rife-anime

* gpu_id: GPU device to use.

* gpu_thread: Thread count for interpolation. Using larger values may increase GPU usage and consume more GPU memory. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.

* tta: TTA(Test-Time Augmentation) mode. It increases quality, but significantly slower.

* uhd: UHD mode. Recommended for 2K above resolution.

* sc: Repeat last frame at scene change to avoid artifacts. You must invoke `misc.SCDetect` on YUV or Gray format of the input beforehand so as to set frame properties.

* list_gpu: Print a list of available GPU device.


Compilation
===========

Requires `Vulkan SDK`.

```
git submodule update --init --recursive --depth 1
meson build
ninja -C build install
```
