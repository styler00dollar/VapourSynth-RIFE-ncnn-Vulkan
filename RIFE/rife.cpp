// rife implemented with ncnn library

#include "rife.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "rife_preproc.comp.hex.h"
#include "rife_postproc.comp.hex.h"
#include "rife_preproc_tta.comp.hex.h"
#include "rife_postproc_tta.comp.hex.h"
#include "rife_flow_tta_avg.comp.hex.h"
#include "rife_v2_flow_tta_avg.comp.hex.h"
#include "rife_flow_tta_temporal_avg.comp.hex.h"
#include "rife_v2_flow_tta_temporal_avg.comp.hex.h"
#include "rife_out_tta_temporal_avg.comp.hex.h"
#include "rife_v4_timestep.comp.hex.h"

#include "rife_ops.h"

DEFINE_LAYER_CREATOR(Warp)

RIFE::RIFE(int gpuid, bool _tta_mode, bool _uhd_mode, int _num_threads, bool _rife_v2, bool _rife_v4, int _padding)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    rife_preproc = 0;
    rife_postproc = 0;
    rife_flow_tta_avg = 0;
    rife_flow_tta_temporal_avg = 0;
    rife_out_tta_temporal_avg = 0;
    rife_v4_timestep = 0;
    rife_uhd_downscale_image = 0;
    rife_uhd_upscale_flow = 0;
    rife_uhd_double_flow = 0;
    rife_v2_slice_flow = 0;
    tta_mode = _tta_mode;
    tta_temporal_mode = false;
    uhd_mode = _uhd_mode;
    num_threads = _num_threads;
    rife_v2 = _rife_v2;
    rife_v4 = _rife_v4;
    padding = _padding;
}

RIFE::~RIFE()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete rife_preproc;
        delete rife_postproc;
        delete rife_flow_tta_avg;
        delete rife_flow_tta_temporal_avg;
        delete rife_out_tta_temporal_avg;
        delete rife_v4_timestep;
    }

    if (uhd_mode)
    {
        rife_uhd_downscale_image->destroy_pipeline(flownet.opt);
        delete rife_uhd_downscale_image;

        rife_uhd_upscale_flow->destroy_pipeline(flownet.opt);
        delete rife_uhd_upscale_flow;

        rife_uhd_double_flow->destroy_pipeline(flownet.opt);
        delete rife_uhd_double_flow;
    }

    if (rife_v2)
    {
        rife_v2_slice_flow->destroy_pipeline(flownet.opt);
        delete rife_v2_slice_flow;
    }
}

#if _WIN32
static void load_param_model(ncnn::Net& net, const std::wstring& modeldir, const wchar_t* name)
{
    wchar_t parampath[256];
    wchar_t modelpath[256];
    swprintf(parampath, 256, L"%s/%s.param", modeldir.c_str(), name);
    swprintf(modelpath, 256, L"%s/%s.bin", modeldir.c_str(), name);

    {
        FILE* fp = _wfopen(parampath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath);
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE* fp = _wfopen(modelpath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath);
        }

        net.load_model(fp);

        fclose(fp);
    }
}
#else
static void load_param_model(ncnn::Net& net, const std::string& modeldir, const char* name)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/%s.param", modeldir.c_str(), name);
    sprintf(modelpath, "%s/%s.bin", modeldir.c_str(), name);

    net.load_param(parampath);
    net.load_model(modelpath);
}
#endif

#if _WIN32
int RIFE::load(const std::wstring& modeldir)
#else
int RIFE::load(const std::string& modeldir)
#endif
{
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = vkdev ? true : false;
    opt.use_fp16_packed = vkdev ? true : false;
    opt.use_fp16_storage = vkdev ? true : false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;

    flownet.opt = opt;
    contextnet.opt = opt;
    fusionnet.opt = opt;

    flownet.set_vulkan_device(vkdev);
    contextnet.set_vulkan_device(vkdev);
    fusionnet.set_vulkan_device(vkdev);

    flownet.register_custom_layer("rife.Warp", Warp_layer_creator);
    contextnet.register_custom_layer("rife.Warp", Warp_layer_creator);
    fusionnet.register_custom_layer("rife.Warp", Warp_layer_creator);

#if _WIN32
    load_param_model(flownet, modeldir, L"flownet");
    if (!rife_v4)
    {
        load_param_model(contextnet, modeldir, L"contextnet");
        load_param_model(fusionnet, modeldir, L"fusionnet");
    }
#else
    load_param_model(flownet, modeldir, "flownet");
    if (!rife_v4)
    {
        load_param_model(contextnet, modeldir, "contextnet");
        load_param_model(fusionnet, modeldir, "fusionnet");
    }
#endif

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_preproc_tta_comp_data, sizeof(rife_preproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_preproc_comp_data, sizeof(rife_preproc_comp_data), opt, spirv);
                }
            }

            rife_preproc = new ncnn::Pipeline(vkdev);
            rife_preproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_postproc_tta_comp_data, sizeof(rife_postproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_postproc_comp_data, sizeof(rife_postproc_comp_data), opt, spirv);
                }
            }

            rife_postproc = new ncnn::Pipeline(vkdev);
            rife_postproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    if (vkdev && tta_mode)
    {
        std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                if (rife_v2)
                {
                    compile_spirv_module(rife_v2_flow_tta_avg_comp_data, sizeof(rife_v2_flow_tta_avg_comp_data), opt, spirv);
                }
                else
                {
                    compile_spirv_module(rife_flow_tta_avg_comp_data, sizeof(rife_flow_tta_avg_comp_data), opt, spirv);
                }
            }
        }

        std::vector<ncnn::vk_specialization_type> specializations(0);

        rife_flow_tta_avg = new ncnn::Pipeline(vkdev);
        rife_flow_tta_avg->set_optimal_local_size_xyz(8, 8, 1);
        rife_flow_tta_avg->create(spirv.data(), spirv.size() * 4, specializations);
    }

    if (vkdev && tta_temporal_mode)
    {
        std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                if (rife_v2)
                {
                    compile_spirv_module(rife_v2_flow_tta_temporal_avg_comp_data, sizeof(rife_v2_flow_tta_temporal_avg_comp_data), opt, spirv);
                }
                else
                {
                    compile_spirv_module(rife_flow_tta_temporal_avg_comp_data, sizeof(rife_flow_tta_temporal_avg_comp_data), opt, spirv);
                }
            }
        }

        std::vector<ncnn::vk_specialization_type> specializations(0);

        rife_flow_tta_temporal_avg = new ncnn::Pipeline(vkdev);
        rife_flow_tta_temporal_avg->set_optimal_local_size_xyz(8, 8, 1);
        rife_flow_tta_temporal_avg->create(spirv.data(), spirv.size() * 4, specializations);
    }

    if (vkdev && tta_temporal_mode)
    {
        std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(rife_out_tta_temporal_avg_comp_data, sizeof(rife_out_tta_temporal_avg_comp_data), opt, spirv);
            }
        }

        std::vector<ncnn::vk_specialization_type> specializations(0);

        rife_out_tta_temporal_avg = new ncnn::Pipeline(vkdev);
        rife_out_tta_temporal_avg->set_optimal_local_size_xyz(8, 8, 1);
        rife_out_tta_temporal_avg->create(spirv.data(), spirv.size() * 4, specializations);
    }

    if (uhd_mode)
    {
        {
            rife_uhd_downscale_image = ncnn::create_layer("Interp");
            rife_uhd_downscale_image->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, 0.5f);
            pd.set(2, 0.5f);
            rife_uhd_downscale_image->load_param(pd);

            rife_uhd_downscale_image->create_pipeline(opt);
        }
        {
            rife_uhd_upscale_flow = ncnn::create_layer("Interp");
            rife_uhd_upscale_flow->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// bilinear
            pd.set(1, 2.f);
            pd.set(2, 2.f);
            rife_uhd_upscale_flow->load_param(pd);

            rife_uhd_upscale_flow->create_pipeline(opt);
        }
        {
            rife_uhd_double_flow = ncnn::create_layer("BinaryOp");
            rife_uhd_double_flow->vkdev = vkdev;

            ncnn::ParamDict pd;
            pd.set(0, 2);// mul
            pd.set(1, 1);// with_scalar
            pd.set(2, 2.f);// b
            rife_uhd_double_flow->load_param(pd);

            rife_uhd_double_flow->create_pipeline(opt);
        }
    }

    if (rife_v2)
    {
        {
            rife_v2_slice_flow = ncnn::create_layer("Slice");
            rife_v2_slice_flow->vkdev = vkdev;

            ncnn::Mat slice_points(2);
            slice_points.fill<int>(-233);

            ncnn::ParamDict pd;
            pd.set(0, slice_points);
            pd.set(1, 0);// axis

            rife_v2_slice_flow->load_param(pd);

            rife_v2_slice_flow->create_pipeline(opt);
        }
    }

    if (rife_v4)
    {
        if (vkdev)
        {
            std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    compile_spirv_module(rife_v4_timestep_comp_data, sizeof(rife_v4_timestep_comp_data), opt, spirv);
                }
            }

            std::vector<ncnn::vk_specialization_type> specializations;

            rife_v4_timestep = new ncnn::Pipeline(vkdev);
            rife_v4_timestep->set_optimal_local_size_xyz(8, 8, 1);
            rife_v4_timestep->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

int RIFE::process(const float* src0R, const float* src0G, const float* src0B,
                  const float* src1R, const float* src1G, const float* src1B,
                  float* dstR, float* dstG, float* dstB,
                  const int w, const int h, const ptrdiff_t stride, const float timestep) const
{
    if (rife_v4)
        return process_v4(src0R, src0G, src0B, src1R, src1G, src1B, dstR, dstG, dstB, w, h, stride, timestep);

    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    in0.create(w, h, channels, sizeof(float), 1);
    in1.create(w, h, channels, sizeof(float), 1);
    float* in0R{ in0.channel(0) };
    float* in0G{ in0.channel(1) };
    float* in0B{ in0.channel(2) };
    float* in1R{ in1.channel(0) };
    float* in1G{ in1.channel(1) };
    float* in1B{ in1.channel(2) };
    for (auto y{ 0 }; y < h; y++) {
        for (auto x{ 0 }; x < w; x++) {
            in0R[w * y + x] = src0R[stride * y + x] * 255.0f;
            in0G[w * y + x] = src0G[stride * y + x] * 255.0f;
            in0B[w * y + x] = src0B[stride * y + x] * 255.0f;
            in1R[w * y + x] = src1R[stride * y + x] * 255.0f;
            in1G[w * y + x] = src1G[stride * y + x] * 255.0f;
            in1B[w * y + x] = src1B[stride * y + x] * 255.0f;
        }
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
    }

    ncnn::VkMat out_gpu;

    if (tta_mode)
    {
        // preproc
        ncnn::VkMat in0_gpu_padded[8];
        ncnn::VkMat in1_gpu_padded[8];
        {
            in0_gpu_padded[0].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[1].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[2].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[3].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[4].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[5].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[6].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[7].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded[0];
            bindings[2] = in0_gpu_padded[1];
            bindings[3] = in0_gpu_padded[2];
            bindings[4] = in0_gpu_padded[3];
            bindings[5] = in0_gpu_padded[4];
            bindings[6] = in0_gpu_padded[5];
            bindings[7] = in0_gpu_padded[6];
            bindings[8] = in0_gpu_padded[7];

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded[0].w;
            constants[4].i = in0_gpu_padded[0].h;
            constants[5].i = in0_gpu_padded[0].cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded[0]);
        }
        {
            in1_gpu_padded[0].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[1].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[2].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[3].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[4].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[5].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[6].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[7].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded[0];
            bindings[2] = in1_gpu_padded[1];
            bindings[3] = in1_gpu_padded[2];
            bindings[4] = in1_gpu_padded[3];
            bindings[5] = in1_gpu_padded[4];
            bindings[6] = in1_gpu_padded[5];
            bindings[7] = in1_gpu_padded[6];
            bindings[8] = in1_gpu_padded[7];

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded[0].w;
            constants[4].i = in1_gpu_padded[0].h;
            constants[5].i = in1_gpu_padded[0].cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded[0]);
        }

        ncnn::VkMat flow[8];
        for (int ti = 0; ti < 8; ti++)
        {
            // flownet
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            if (uhd_mode)
            {
                ncnn::VkMat in0_gpu_padded_downscaled;
                ncnn::VkMat in1_gpu_padded_downscaled;
                rife_uhd_downscale_image->forward(in0_gpu_padded[ti], in0_gpu_padded_downscaled, cmd, opt);
                rife_uhd_downscale_image->forward(in1_gpu_padded[ti], in1_gpu_padded_downscaled, cmd, opt);

                ex.input("input0", in0_gpu_padded_downscaled);
                ex.input("input1", in1_gpu_padded_downscaled);

                ncnn::VkMat flow_downscaled;
                ex.extract("flow", flow_downscaled, cmd);

                ncnn::VkMat flow_half;
                rife_uhd_upscale_flow->forward(flow_downscaled, flow_half, cmd, opt);

                rife_uhd_double_flow->forward(flow_half, flow[ti], cmd, opt);
            }
            else
            {
                ex.input("input0", in0_gpu_padded[ti]);
                ex.input("input1", in1_gpu_padded[ti]);
                ex.extract("flow", flow[ti], cmd);
            }
        }

        ncnn::VkMat flow_reversed[8];
        if (tta_temporal_mode)
        {
            for (int ti = 0; ti < 8; ti++)
            {
                // flownet
                ncnn::Extractor ex = flownet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                if (uhd_mode)
                {
                    ncnn::VkMat in0_gpu_padded_downscaled;
                    ncnn::VkMat in1_gpu_padded_downscaled;
                    rife_uhd_downscale_image->forward(in0_gpu_padded[ti], in0_gpu_padded_downscaled, cmd, opt);
                    rife_uhd_downscale_image->forward(in1_gpu_padded[ti], in1_gpu_padded_downscaled, cmd, opt);

                    ex.input("input0", in1_gpu_padded_downscaled);
                    ex.input("input1", in0_gpu_padded_downscaled);

                    ncnn::VkMat flow_downscaled;
                    ex.extract("flow", flow_downscaled, cmd);

                    ncnn::VkMat flow_half;
                    rife_uhd_upscale_flow->forward(flow_downscaled, flow_half, cmd, opt);

                    rife_uhd_double_flow->forward(flow_half, flow_reversed[ti], cmd, opt);
                }
                else
                {
                    ex.input("input0", in1_gpu_padded[ti]);
                    ex.input("input1", in0_gpu_padded[ti]);
                    ex.extract("flow", flow_reversed[ti], cmd);
                }
            }
        }

        // avg flow
        ncnn::VkMat flow0[8];
        ncnn::VkMat flow1[8];
        {
            std::vector<ncnn::VkMat> bindings(8);
            bindings[0] = flow[0];
            bindings[1] = flow[1];
            bindings[2] = flow[2];
            bindings[3] = flow[3];
            bindings[4] = flow[4];
            bindings[5] = flow[5];
            bindings[6] = flow[6];
            bindings[7] = flow[7];

            std::vector<ncnn::vk_constant_type> constants(3);
            constants[0].i = flow[0].w;
            constants[1].i = flow[0].h;
            constants[2].i = flow[0].cstep;

            ncnn::VkMat dispatcher;
            dispatcher.w = flow[0].w;
            dispatcher.h = flow[0].h;
            dispatcher.c = 1;
            cmd.record_pipeline(rife_flow_tta_avg, bindings, constants, dispatcher);
        }

        if (tta_temporal_mode)
        {
            std::vector<ncnn::VkMat> bindings(8);
            bindings[0] = flow_reversed[0];
            bindings[1] = flow_reversed[1];
            bindings[2] = flow_reversed[2];
            bindings[3] = flow_reversed[3];
            bindings[4] = flow_reversed[4];
            bindings[5] = flow_reversed[5];
            bindings[6] = flow_reversed[6];
            bindings[7] = flow_reversed[7];

            std::vector<ncnn::vk_constant_type> constants(3);
            constants[0].i = flow_reversed[0].w;
            constants[1].i = flow_reversed[0].h;
            constants[2].i = flow_reversed[0].cstep;

            ncnn::VkMat dispatcher;
            dispatcher.w = flow_reversed[0].w;
            dispatcher.h = flow_reversed[0].h;
            dispatcher.c = 1;
            cmd.record_pipeline(rife_flow_tta_avg, bindings, constants, dispatcher);

            // merge flow and flow_reversed
            for (int ti = 0; ti < 8; ti++)
            {
                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = flow[ti];
                bindings[1] = flow_reversed[ti];

                std::vector<ncnn::vk_constant_type> constants(3);
                constants[0].i = flow[ti].w;
                constants[1].i = flow[ti].h;
                constants[2].i = flow[ti].cstep;

                ncnn::VkMat dispatcher;
                dispatcher.w = flow[ti].w;
                dispatcher.h = flow[ti].h;
                dispatcher.c = 1;

                cmd.record_pipeline(rife_flow_tta_temporal_avg, bindings, constants, dispatcher);
            }
        }

        if (rife_v2)
        {
            for (int ti = 0; ti < 8; ti++)
            {
                std::vector<ncnn::VkMat> inputs(1);
                inputs[0] = flow[ti];
                std::vector<ncnn::VkMat> outputs(2);
                rife_v2_slice_flow->forward(inputs, outputs, cmd, opt);
                flow0[ti] = outputs[0];
                flow1[ti] = outputs[1];
            }
        }

        ncnn::VkMat out_gpu_padded[8];
        for (int ti = 0; ti < 8; ti++)
        {
            // contextnet
            ncnn::VkMat ctx0[4];
            ncnn::VkMat ctx1[4];
            {
                ncnn::Extractor ex = contextnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input.1", in0_gpu_padded[ti]);
                if (rife_v2)
                {
                    ex.input("flow.0", flow0[ti]);
                }
                else
                {
                    ex.input("flow.0", flow[ti]);
                }
                ex.extract("f1", ctx0[0], cmd);
                ex.extract("f2", ctx0[1], cmd);
                ex.extract("f3", ctx0[2], cmd);
                ex.extract("f4", ctx0[3], cmd);
            }
            {
                ncnn::Extractor ex = contextnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("input.1", in1_gpu_padded[ti]);
                if (rife_v2)
                {
                    ex.input("flow.0", flow1[ti]);
                }
                else
                {
                    ex.input("flow.1", flow[ti]);
                }
                ex.extract("f1", ctx1[0], cmd);
                ex.extract("f2", ctx1[1], cmd);
                ex.extract("f3", ctx1[2], cmd);
                ex.extract("f4", ctx1[3], cmd);
            }

            // fusionnet
            {
                ncnn::Extractor ex = fusionnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("img0", in0_gpu_padded[ti]);
                ex.input("img1", in1_gpu_padded[ti]);
                ex.input("flow", flow[ti]);
                ex.input("3", ctx0[0]);
                ex.input("4", ctx0[1]);
                ex.input("5", ctx0[2]);
                ex.input("6", ctx0[3]);
                ex.input("7", ctx1[0]);
                ex.input("8", ctx1[1]);
                ex.input("9", ctx1[2]);
                ex.input("10", ctx1[3]);

                // save some memory
                if (!tta_temporal_mode)
                {
                    if (ti == 0)
                    {
                        in0_gpu.release();
                        in1_gpu.release();
                    }
                    else
                    {
                        in0_gpu_padded[ti - 1].release();
                        in1_gpu_padded[ti - 1].release();
                    }
                    ctx0[0].release();
                    ctx0[1].release();
                    ctx0[2].release();
                    ctx0[3].release();
                    ctx1[0].release();
                    ctx1[1].release();
                    ctx1[2].release();
                    ctx1[3].release();
                }
                if (ti != 0)
                {
                    flow[ti - 1].release();
                }

                ex.extract("output", out_gpu_padded[ti], cmd);
            }

            if (tta_temporal_mode)
            {
                // fusionnet
                ncnn::VkMat out_gpu_padded_reversed;
                {
                    ncnn::Extractor ex = fusionnet.create_extractor();
                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("img0", in1_gpu_padded[ti]);
                    ex.input("img1", in0_gpu_padded[ti]);
                    ex.input("flow", flow_reversed[ti]);
                    ex.input("3", ctx1[0]);
                    ex.input("4", ctx1[1]);
                    ex.input("5", ctx1[2]);
                    ex.input("6", ctx1[3]);
                    ex.input("7", ctx0[0]);
                    ex.input("8", ctx0[1]);
                    ex.input("9", ctx0[2]);
                    ex.input("10", ctx0[3]);

                    // save some memory
                    if (ti == 0)
                    {
                        in0_gpu.release();
                        in1_gpu.release();
                    }
                    else
                    {
                        in0_gpu_padded[ti - 1].release();
                        in1_gpu_padded[ti - 1].release();
                        flow_reversed[ti - 1].release();
                    }
                    ctx0[0].release();
                    ctx0[1].release();
                    ctx0[2].release();
                    ctx0[3].release();
                    ctx1[0].release();
                    ctx1[1].release();
                    ctx1[2].release();
                    ctx1[3].release();

                    ex.extract("output", out_gpu_padded_reversed, cmd);
                }

                // merge output
                {
                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = out_gpu_padded[ti];
                    bindings[1] = out_gpu_padded_reversed;

                    std::vector<ncnn::vk_constant_type> constants(3);
                    constants[0].i = out_gpu_padded[ti].w;
                    constants[1].i = out_gpu_padded[ti].h;
                    constants[2].i = out_gpu_padded[ti].cstep;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = out_gpu_padded[ti].w;
                    dispatcher.h = out_gpu_padded[ti].h;
                    dispatcher.c = 3;
                    cmd.record_pipeline(rife_out_tta_temporal_avg, bindings, constants, dispatcher);
                }
            }
        }

        out_gpu.create(w, h, channels, sizeof(float), 1, blob_vkallocator);

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = out_gpu_padded[0];
            bindings[1] = out_gpu_padded[1];
            bindings[2] = out_gpu_padded[2];
            bindings[3] = out_gpu_padded[3];
            bindings[4] = out_gpu_padded[4];
            bindings[5] = out_gpu_padded[5];
            bindings[6] = out_gpu_padded[6];
            bindings[7] = out_gpu_padded[7];
            bindings[8] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded[0].w;
            constants[1].i = out_gpu_padded[0].h;
            constants[2].i = out_gpu_padded[0].cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }
    else
    {
        // preproc
        ncnn::VkMat in0_gpu_padded;
        ncnn::VkMat in1_gpu_padded;
        {
            in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded.w;
            constants[4].i = in0_gpu_padded.h;
            constants[5].i = in0_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
        }
        {
            in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded.w;
            constants[4].i = in1_gpu_padded.h;
            constants[5].i = in1_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
        }

        // flownet
        ncnn::VkMat flow;
        ncnn::VkMat flow0;
        ncnn::VkMat flow1;
        {
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            if (uhd_mode)
            {
                ncnn::VkMat in0_gpu_padded_downscaled;
                ncnn::VkMat in1_gpu_padded_downscaled;
                rife_uhd_downscale_image->forward(in0_gpu_padded, in0_gpu_padded_downscaled, cmd, opt);
                rife_uhd_downscale_image->forward(in1_gpu_padded, in1_gpu_padded_downscaled, cmd, opt);

                ex.input("input0", in0_gpu_padded_downscaled);
                ex.input("input1", in1_gpu_padded_downscaled);

                ncnn::VkMat flow_downscaled;
                ex.extract("flow", flow_downscaled, cmd);

                ncnn::VkMat flow_half;
                rife_uhd_upscale_flow->forward(flow_downscaled, flow_half, cmd, opt);

                rife_uhd_double_flow->forward(flow_half, flow, cmd, opt);
            }
            else
            {
                ex.input("input0", in0_gpu_padded);
                ex.input("input1", in1_gpu_padded);
                ex.extract("flow", flow, cmd);
            }
        }

        ncnn::VkMat flow_reversed;
        if (tta_temporal_mode)
        {
            // flownet
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            if (uhd_mode)
            {
                ncnn::VkMat in0_gpu_padded_downscaled;
                ncnn::VkMat in1_gpu_padded_downscaled;
                rife_uhd_downscale_image->forward(in0_gpu_padded, in0_gpu_padded_downscaled, cmd, opt);
                rife_uhd_downscale_image->forward(in1_gpu_padded, in1_gpu_padded_downscaled, cmd, opt);

                ex.input("input0", in1_gpu_padded_downscaled);
                ex.input("input1", in0_gpu_padded_downscaled);

                ncnn::VkMat flow_downscaled;
                ex.extract("flow", flow_downscaled, cmd);

                ncnn::VkMat flow_half;
                rife_uhd_upscale_flow->forward(flow_downscaled, flow_half, cmd, opt);

                rife_uhd_double_flow->forward(flow_half, flow_reversed, cmd, opt);
            }
            else
            {
                ex.input("input0", in1_gpu_padded);
                ex.input("input1", in0_gpu_padded);
                ex.extract("flow", flow_reversed, cmd);
            }

            // merge flow and flow_reversed
            {
                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = flow;
                bindings[1] = flow_reversed;

                std::vector<ncnn::vk_constant_type> constants(3);
                constants[0].i = flow.w;
                constants[1].i = flow.h;
                constants[2].i = flow.cstep;

                ncnn::VkMat dispatcher;
                dispatcher.w = flow.w;
                dispatcher.h = flow.h;
                dispatcher.c = 1;

                cmd.record_pipeline(rife_flow_tta_temporal_avg, bindings, constants, dispatcher);
            }
        }

        if (rife_v2)
        {
            std::vector<ncnn::VkMat> inputs(1);
            inputs[0] = flow;
            std::vector<ncnn::VkMat> outputs(2);
            rife_v2_slice_flow->forward(inputs, outputs, cmd, opt);
            flow0 = outputs[0];
            flow1 = outputs[1];
        }

        // contextnet
        ncnn::VkMat ctx0[4];
        ncnn::VkMat ctx1[4];
        {
            ncnn::Extractor ex = contextnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input.1", in0_gpu_padded);
            if (rife_v2)
            {
                ex.input("flow.0", flow0);
            }
            else
            {
                ex.input("flow.0", flow);
            }
            ex.extract("f1", ctx0[0], cmd);
            ex.extract("f2", ctx0[1], cmd);
            ex.extract("f3", ctx0[2], cmd);
            ex.extract("f4", ctx0[3], cmd);
        }
        {
            ncnn::Extractor ex = contextnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("input.1", in1_gpu_padded);
            if (rife_v2)
            {
                ex.input("flow.0", flow1);
            }
            else
            {
                ex.input("flow.1", flow);
            }
            ex.extract("f1", ctx1[0], cmd);
            ex.extract("f2", ctx1[1], cmd);
            ex.extract("f3", ctx1[2], cmd);
            ex.extract("f4", ctx1[3], cmd);
        }

        // fusionnet
        ncnn::VkMat out_gpu_padded;
        {
            ncnn::Extractor ex = fusionnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("img0", in0_gpu_padded);
            ex.input("img1", in1_gpu_padded);
            ex.input("flow", flow);
            ex.input("3", ctx0[0]);
            ex.input("4", ctx0[1]);
            ex.input("5", ctx0[2]);
            ex.input("6", ctx0[3]);
            ex.input("7", ctx1[0]);
            ex.input("8", ctx1[1]);
            ex.input("9", ctx1[2]);
            ex.input("10", ctx1[3]);

            if (!tta_temporal_mode)
            {
                // save some memory
                in0_gpu.release();
                in1_gpu.release();
                ctx0[0].release();
                ctx0[1].release();
                ctx0[2].release();
                ctx0[3].release();
                ctx1[0].release();
                ctx1[1].release();
                ctx1[2].release();
                ctx1[3].release();
            }
            flow.release();

            ex.extract("output", out_gpu_padded, cmd);
        }

        if (tta_temporal_mode)
        {
            // fusionnet
            ncnn::VkMat out_gpu_padded_reversed;
            {
                ncnn::Extractor ex = fusionnet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("img0", in1_gpu_padded);
                ex.input("img1", in0_gpu_padded);
                ex.input("flow", flow_reversed);
                ex.input("3", ctx1[0]);
                ex.input("4", ctx1[1]);
                ex.input("5", ctx1[2]);
                ex.input("6", ctx1[3]);
                ex.input("7", ctx0[0]);
                ex.input("8", ctx0[1]);
                ex.input("9", ctx0[2]);
                ex.input("10", ctx0[3]);

                // save some memory
                in0_gpu.release();
                in1_gpu.release();
                ctx0[0].release();
                ctx0[1].release();
                ctx0[2].release();
                ctx0[3].release();
                ctx1[0].release();
                ctx1[1].release();
                ctx1[2].release();
                ctx1[3].release();
                flow_reversed.release();

                ex.extract("output", out_gpu_padded_reversed, cmd);
            }

            // merge output
            {
                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = out_gpu_padded;
                bindings[1] = out_gpu_padded_reversed;

                std::vector<ncnn::vk_constant_type> constants(3);
                constants[0].i = out_gpu_padded.w;
                constants[1].i = out_gpu_padded.h;
                constants[2].i = out_gpu_padded.cstep;

                ncnn::VkMat dispatcher;
                dispatcher.w = out_gpu_padded.w;
                dispatcher.h = out_gpu_padded.h;
                dispatcher.c = 3;
                cmd.record_pipeline(rife_out_tta_temporal_avg, bindings, constants, dispatcher);
            }
        }

        out_gpu.create(w, h, channels, sizeof(float), 1, blob_vkallocator);

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = out_gpu_padded;
            bindings[1] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded.w;
            constants[1].i = out_gpu_padded.h;
            constants[2].i = out_gpu_padded.cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }

    // download
    {
        ncnn::Mat out;

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        const float* outR{ out.channel(0) };
        const float* outG{ out.channel(1) };
        const float* outB{ out.channel(2) };
        for (auto y{ 0 }; y < h; y++) {
            for (auto x{ 0 }; x < w; x++) {
                dstR[stride * y + x] = outR[w * y + x] * (1 / 255.0f);
                dstG[stride * y + x] = outG[w * y + x] * (1 / 255.0f);
                dstB[stride * y + x] = outB[w * y + x] * (1 / 255.0f);
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RIFE::process_v4(const float* src0R, const float* src0G, const float* src0B,
                     const float* src1R, const float* src1G, const float* src1B,
                     float* dstR, float* dstG, float* dstB,
                     const int w, const int h, const ptrdiff_t stride, const float timestep) const
{
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;
    // padding, the default is 32, but newer rife models require 64
    int w_padded, h_padded;
    w_padded = (w + padding - 1) / padding * padding;
    h_padded = (h + padding - 1) / padding * padding;
    

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    in0.create(w, h, channels, sizeof(float), 1);
    in1.create(w, h, channels, sizeof(float), 1);
    float* in0R{ in0.channel(0) };
    float* in0G{ in0.channel(1) };
    float* in0B{ in0.channel(2) };
    float* in1R{ in1.channel(0) };
    float* in1G{ in1.channel(1) };
    float* in1B{ in1.channel(2) };
    for (auto y{ 0 }; y < h; y++) {
        for (auto x{ 0 }; x < w; x++) {
            in0R[w * y + x] = src0R[stride * y + x] * 255.0f;
            in0G[w * y + x] = src0G[stride * y + x] * 255.0f;
            in0B[w * y + x] = src0B[stride * y + x] * 255.0f;
            in1R[w * y + x] = src1R[stride * y + x] * 255.0f;
            in1G[w * y + x] = src1G[stride * y + x] * 255.0f;
            in1B[w * y + x] = src1B[stride * y + x] * 255.0f;
        }
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
    }

    ncnn::VkMat out_gpu;

    {
        // preproc
        ncnn::VkMat in0_gpu_padded;
        ncnn::VkMat in1_gpu_padded;
        ncnn::VkMat timestep_gpu_padded;
        {
            in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded.w;
            constants[4].i = in0_gpu_padded.h;
            constants[5].i = in0_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
        }
        {
            in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded.w;
            constants[4].i = in1_gpu_padded.h;
            constants[5].i = in1_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
        }
        {
            timestep_gpu_padded.create(w_padded, h_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(1);
            bindings[0] = timestep_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = timestep_gpu_padded.w;
            constants[1].i = timestep_gpu_padded.h;
            constants[2].i = timestep_gpu_padded.cstep;
            constants[3].f = timestep;

            cmd.record_pipeline(rife_v4_timestep, bindings, constants, timestep_gpu_padded);
        }

        // flownet
        ncnn::VkMat out_gpu_padded;
        {
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("in0", in0_gpu_padded);
            ex.input("in1", in1_gpu_padded);
            ex.input("in2", timestep_gpu_padded);
            ex.extract("out0", out_gpu_padded, cmd);
        }

        out_gpu.create(w, h, channels, sizeof(float), 1, blob_vkallocator);

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = out_gpu_padded;
            bindings[1] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded.w;
            constants[1].i = out_gpu_padded.h;
            constants[2].i = out_gpu_padded.cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }

    // download
    {
        ncnn::Mat out;

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        const float* outR{ out.channel(0) };
        const float* outG{ out.channel(1) };
        const float* outB{ out.channel(2) };
        for (auto y{ 0 }; y < h; y++) {
            for (auto x{ 0 }; x < w; x++) {
                dstR[stride * y + x] = outR[w * y + x] * (1 / 255.0f);
                dstG[stride * y + x] = outG[w * y + x] * (1 / 255.0f);
                dstB[stride * y + x] = outB[w * y + x] * (1 / 255.0f);
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
