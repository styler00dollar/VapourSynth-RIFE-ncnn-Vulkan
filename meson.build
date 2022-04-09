project('RIFE-ncnn-Vulkan', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20'],
  license: 'MIT',
  meson_version: '>=0.55.0',
  version: '3'
)

cxx = meson.get_compiler('cpp')

gcc_syntax = cxx.get_argument_syntax() == 'gcc'

if get_option('buildtype') == 'release'
  add_global_arguments(gcc_syntax ? ['-fno-math-errno', '-fno-trapping-math'] : '/GS-', language: 'cpp')
endif

deps = [dependency('openmp')]

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)
  deps += vapoursynth_dep
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'
else
  add_project_arguments('/DNOMINMAX', language: 'cpp')
  install_dir = get_option('libdir') / 'vapoursynth'
endif

cmake = import('cmake')

opt_var = cmake.subproject_options()

opt_var.add_cmake_defines({'NCNN_INSTALL_SDK': false})
opt_var.add_cmake_defines({'NCNN_PIXEL_ROTATE': false})
opt_var.add_cmake_defines({'NCNN_PIXEL_AFFINE': false})
opt_var.add_cmake_defines({'NCNN_PIXEL_DRAWING': false})
opt_var.add_cmake_defines({'NCNN_VULKAN': true})
opt_var.add_cmake_defines({'NCNN_VULKAN_ONLINE_SPIRV': true})
opt_var.add_cmake_defines({'NCNN_BUILD_BENCHMARK': false})
opt_var.add_cmake_defines({'NCNN_BUILD_TESTS': false})
opt_var.add_cmake_defines({'NCNN_BUILD_TOOLS': false})
opt_var.add_cmake_defines({'NCNN_BUILD_EXAMPLES': false})
opt_var.add_cmake_defines({'NCNN_INT8': false})

opt_var.add_cmake_defines({'WITH_LAYER_absval': false})
opt_var.add_cmake_defines({'WITH_LAYER_argmax': false})
opt_var.add_cmake_defines({'WITH_LAYER_batchnorm': false})
opt_var.add_cmake_defines({'WITH_LAYER_bias': false})
opt_var.add_cmake_defines({'WITH_LAYER_bnll': false})
opt_var.add_cmake_defines({'WITH_LAYER_concat': true})
opt_var.add_cmake_defines({'WITH_LAYER_convolution': true})
opt_var.add_cmake_defines({'WITH_LAYER_crop': true})
opt_var.add_cmake_defines({'WITH_LAYER_deconvolution': true})
opt_var.add_cmake_defines({'WITH_LAYER_dropout': false})
opt_var.add_cmake_defines({'WITH_LAYER_eltwise': true})
opt_var.add_cmake_defines({'WITH_LAYER_elu': false})
opt_var.add_cmake_defines({'WITH_LAYER_embed': false})
opt_var.add_cmake_defines({'WITH_LAYER_exp': false})
opt_var.add_cmake_defines({'WITH_LAYER_flatten': true})
opt_var.add_cmake_defines({'WITH_LAYER_innerproduct': true})
opt_var.add_cmake_defines({'WITH_LAYER_input': true})
opt_var.add_cmake_defines({'WITH_LAYER_log': false})
opt_var.add_cmake_defines({'WITH_LAYER_lrn': false})
opt_var.add_cmake_defines({'WITH_LAYER_memorydata': false})
opt_var.add_cmake_defines({'WITH_LAYER_mvn': false})
opt_var.add_cmake_defines({'WITH_LAYER_pooling': true})
opt_var.add_cmake_defines({'WITH_LAYER_power': false})
opt_var.add_cmake_defines({'WITH_LAYER_prelu': true})
opt_var.add_cmake_defines({'WITH_LAYER_proposal': false})
opt_var.add_cmake_defines({'WITH_LAYER_reduction': false})
opt_var.add_cmake_defines({'WITH_LAYER_relu': true})
opt_var.add_cmake_defines({'WITH_LAYER_reshape': false})
opt_var.add_cmake_defines({'WITH_LAYER_roipooling': false})
opt_var.add_cmake_defines({'WITH_LAYER_scale': false})
opt_var.add_cmake_defines({'WITH_LAYER_sigmoid': true})
opt_var.add_cmake_defines({'WITH_LAYER_slice': true})
opt_var.add_cmake_defines({'WITH_LAYER_softmax': false})
opt_var.add_cmake_defines({'WITH_LAYER_split': true})
opt_var.add_cmake_defines({'WITH_LAYER_spp': false})
opt_var.add_cmake_defines({'WITH_LAYER_tanh': false})
opt_var.add_cmake_defines({'WITH_LAYER_threshold': false})
opt_var.add_cmake_defines({'WITH_LAYER_tile': false})
opt_var.add_cmake_defines({'WITH_LAYER_rnn': false})
opt_var.add_cmake_defines({'WITH_LAYER_lstm': false})
opt_var.add_cmake_defines({'WITH_LAYER_binaryop': true})
opt_var.add_cmake_defines({'WITH_LAYER_unaryop': true})
opt_var.add_cmake_defines({'WITH_LAYER_convolutiondepthwise': false})
opt_var.add_cmake_defines({'WITH_LAYER_padding': true})
opt_var.add_cmake_defines({'WITH_LAYER_squeeze': false})
opt_var.add_cmake_defines({'WITH_LAYER_expanddims': false})
opt_var.add_cmake_defines({'WITH_LAYER_normalize': false})
opt_var.add_cmake_defines({'WITH_LAYER_permute': false})
opt_var.add_cmake_defines({'WITH_LAYER_priorbox': false})
opt_var.add_cmake_defines({'WITH_LAYER_detectionoutput': false})
opt_var.add_cmake_defines({'WITH_LAYER_interp': true})
opt_var.add_cmake_defines({'WITH_LAYER_deconvolutiondepthwise': false})
opt_var.add_cmake_defines({'WITH_LAYER_shufflechannel': false})
opt_var.add_cmake_defines({'WITH_LAYER_instancenorm': false})
opt_var.add_cmake_defines({'WITH_LAYER_clip': true})
opt_var.add_cmake_defines({'WITH_LAYER_reorg': false})
opt_var.add_cmake_defines({'WITH_LAYER_yolodetectionoutput': false})
opt_var.add_cmake_defines({'WITH_LAYER_quantize': false})
opt_var.add_cmake_defines({'WITH_LAYER_dequantize': false})
opt_var.add_cmake_defines({'WITH_LAYER_yolov3detectionoutput': false})
opt_var.add_cmake_defines({'WITH_LAYER_psroipooling': false})
opt_var.add_cmake_defines({'WITH_LAYER_roialign': false})
opt_var.add_cmake_defines({'WITH_LAYER_packing': true})
opt_var.add_cmake_defines({'WITH_LAYER_requantize': false})
opt_var.add_cmake_defines({'WITH_LAYER_cast': true})
opt_var.add_cmake_defines({'WITH_LAYER_hardsigmoid': false})
opt_var.add_cmake_defines({'WITH_LAYER_selu': false})
opt_var.add_cmake_defines({'WITH_LAYER_hardswish': false})
opt_var.add_cmake_defines({'WITH_LAYER_noop': false})
opt_var.add_cmake_defines({'WITH_LAYER_pixelshuffle': true})
opt_var.add_cmake_defines({'WITH_LAYER_deepcopy': false})
opt_var.add_cmake_defines({'WITH_LAYER_mish': false})
opt_var.add_cmake_defines({'WITH_LAYER_statisticspooling': false})
opt_var.add_cmake_defines({'WITH_LAYER_swish': false})
opt_var.add_cmake_defines({'WITH_LAYER_gemm': false})
opt_var.add_cmake_defines({'WITH_LAYER_groupnorm': false})
opt_var.add_cmake_defines({'WITH_LAYER_layernorm': false})
opt_var.add_cmake_defines({'WITH_LAYER_softplus': false})
opt_var.add_cmake_defines({'WITH_LAYER_gru': false})
opt_var.add_cmake_defines({'WITH_LAYER_multiheadattention': false})
opt_var.add_cmake_defines({'WITH_LAYER_gelu': false})
opt_var.add_cmake_defines({'WITH_LAYER_convolution1d': false})
opt_var.add_cmake_defines({'WITH_LAYER_pooling1d': false})
opt_var.add_cmake_defines({'WITH_LAYER_convolutiondepthwise1d': false})
opt_var.add_cmake_defines({'WITH_LAYER_convolution3d': false})
opt_var.add_cmake_defines({'WITH_LAYER_convolutiondepthwise3d': false})
opt_var.add_cmake_defines({'WITH_LAYER_pooling3d': false})
opt_var.add_cmake_defines({'WITH_LAYER_matmul': false})
opt_var.add_cmake_defines({'WITH_LAYER_deconvolution1d': false})
opt_var.add_cmake_defines({'WITH_LAYER_deconvolutiondepthwise1d': false})
opt_var.add_cmake_defines({'WITH_LAYER_deconvolution3d': false})
opt_var.add_cmake_defines({'WITH_LAYER_deconvolutiondepthwise3d': false})

if host_machine.system() == 'darwin'
  opt_var.add_cmake_defines({'CMAKE_OSX_ARCHITECTURES': 'x86_64'})
endif

if gcc_syntax
  opt_var.append_compile_args('cpp', '-fvisibility=hidden')
endif

opt_var.set_install(false)

sub_proj = cmake.subproject('ncnn', options: opt_var)

deps += sub_proj.dependency('GenericCodeGen')
deps += sub_proj.dependency('glslang')
deps += sub_proj.dependency('MachineIndependent')
deps += sub_proj.dependency('ncnn')
deps += sub_proj.dependency('OGLCompiler')
deps += sub_proj.dependency('OSDependent')
deps += sub_proj.dependency('SPIRV')

sources = [
  'RIFE/plugin.cpp',
  'RIFE/rife.cpp',
  'RIFE/rife.h',
  'RIFE/rife_ops.h',
  'RIFE/warp.cpp'
]

if host_machine.cpu_family().startswith('x86') and gcc_syntax
  add_project_arguments('-mfpmath=sse', '-msse2', language: 'cpp')
endif

shared_module('rife', sources,
  dependencies: deps,
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden'
)

install_subdir('models',
  install_dir: install_dir
)
