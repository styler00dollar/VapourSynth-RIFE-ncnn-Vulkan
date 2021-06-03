// rife implemented with ncnn library

#ifndef RIFE_H
#define RIFE_H

#include <string>

// ncnn
#include "net.h"

class RIFE
{
public:
    RIFE(int gpuid, bool tta_mode = false, bool uhd_mode = false, int num_threads = 1, bool rife_v2 = false);
    ~RIFE();

#if _WIN32
    int load(const std::wstring& modeldir, const bool fp32);
#else
    int load(const std::string& modeldir, const bool fp32);
#endif

    int process(const float* src0R, const float* src0G, const float* src0B,
                const float* src1R, const float* src1G, const float* src1B,
                float* dstR, float* dstG, float* dstB,
                const int w, const int h, const int stride) const;

    int process_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net flownet;
    ncnn::Net contextnet;
    ncnn::Net fusionnet;
    ncnn::Pipeline* rife_preproc;
    ncnn::Pipeline* rife_postproc;
    ncnn::Pipeline* rife_flow_tta_avg;
    ncnn::Pipeline* rife_flow_tta_temporal_avg;
    ncnn::Pipeline* rife_out_tta_temporal_avg;
    ncnn::Layer* rife_uhd_downscale_image;
    ncnn::Layer* rife_uhd_upscale_flow;
    ncnn::Layer* rife_uhd_double_flow;
    ncnn::Layer* rife_v2_slice_flow;
    bool tta_mode;
    bool tta_temporal_mode;
    bool uhd_mode;
    int num_threads;
    bool rife_v2;
};

#endif // RIFE_H