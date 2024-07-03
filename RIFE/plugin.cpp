/*
    MIT License

    Copyright (c) 2021-2022 HolyWu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <atomic>
#include <fstream>
#include <memory>
#include <semaphore>
#include <string>
#include <vector>
#include <iostream>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#include "rife.h"

using namespace std::literals;

static std::atomic<int> numGPUInstances{ 0 };

struct RIFEData final {
    VSNode* node;
    VSNode* psnr;
    VSVideoInfo vi;
    bool sceneChange;
    bool skip;
    double skipThreshold;
    int64_t factor;
    int64_t factorNum;
    int64_t factorDen;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
};

static void filter(const VSFrame* src0, const VSFrame* src1, VSFrame* dst,
                   const float timestep, const RIFEData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    const auto width{ vsapi->getFrameWidth(src0, 0) };
    const auto height{ vsapi->getFrameHeight(src0, 0) };
    const auto stride{ vsapi->getStride(src0, 0) / d->vi.format.bytesPerSample };
    auto src0R{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 0)) };
    auto src0G{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 1)) };
    auto src0B{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 2)) };
    auto src1R{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 0)) };
    auto src1G{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 1)) };
    auto src1B{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 2)) };
    auto dstR{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0)) };
    auto dstG{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 1)) };
    auto dstB{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 2)) };

    d->semaphore->acquire();
    d->rife->process(src0R, src0G, src0B, src1R, src1G, src1B, dstR, dstG, dstB, width, height, stride, timestep);
    d->semaphore->release();
}

static const VSFrame* VS_CC rifeGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                         VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEData*>(instanceData) };

    auto frameNum{ static_cast<int>(n * d->factorDen / d->factorNum) };
    auto remainder{ n * d->factorDen % d->factorNum };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(frameNum, d->node, frameCtx);
        if (remainder != 0 && n < d->vi.numFrames - d->factor)
            vsapi->requestFrameFilter(frameNum + 1, d->node, frameCtx);

        if (d->skip)
            vsapi->requestFrameFilter(frameNum, d->psnr, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src0{ vsapi->getFrameFilter(frameNum, d->node, frameCtx) };
        decltype(src0) src1{};
        decltype(src0) psnr{};
        VSFrame* dst{};

        if (remainder != 0 && n < d->vi.numFrames - d->factor) {
            bool sceneChange{};
            double psnrY{ -1.0 };
            int err;

            if (d->sceneChange)
                sceneChange = !!vsapi->mapGetInt(vsapi->getFramePropertiesRO(src0), "_SceneChangeNext", 0, &err);

            if (d->skip) {
                psnr = vsapi->getFrameFilter(frameNum, d->psnr, frameCtx);
                psnrY = vsapi->mapGetFloat(vsapi->getFramePropertiesRO(psnr), "psnr_y", 0, nullptr);
            }

            if (sceneChange || psnrY >= d->skipThreshold) {
                dst = vsapi->copyFrame(src0, core);
            } else {
                src1 = vsapi->getFrameFilter(frameNum + 1, d->node, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src0, core);
                filter(src0, src1, dst, static_cast<float>(remainder) / d->factorNum, d, vsapi);
            }
        } else {
            dst = vsapi->copyFrame(src0, core);
        }

        auto props{ vsapi->getFramePropertiesRW(dst) };
        int errNum, errDen;
        auto durationNum{ vsapi->mapGetInt(props, "_DurationNum", 0, &errNum) };
        auto durationDen{ vsapi->mapGetInt(props, "_DurationDen", 0, &errDen) };
        if (!errNum && !errDen) {
            vsh::muldivRational(&durationNum, &durationDen, d->factorDen, d->factorNum);
            vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
            vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
        }

        vsapi->freeFrame(src0);
        vsapi->freeFrame(src1);
        vsapi->freeFrame(psnr);
        return dst;
    }

    return nullptr;
}

static void VS_CC rifeFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEData*>(instanceData) };
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->psnr);
    delete d;

    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static void VS_CC rifeCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{ std::make_unique<RIFEData>() };

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);
        int err;

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            d->vi.format.colorFamily != cfRGB ||
            d->vi.format.sampleType != stFloat ||
            d->vi.format.bitsPerSample != 32)
            throw "only constant RGB format 32 bit float input supported";

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;

        auto model{ vsapi->mapGetIntSaturated(in, "model", 0, &err) };
        if (err)
            model = 5;

        auto factorNum{ vsapi->mapGetInt(in, "factor_num", 0, &err) };
        if (err)
            factorNum = 2;

        auto factorDen{ vsapi->mapGetInt(in, "factor_den", 0, &err) };
        if (err)
            factorDen = 1;

        auto fpsNum{ vsapi->mapGetInt(in, "fps_num", 0, &err) };
        if (!err && fpsNum < 1)
            throw "fps_num must be at least 1";

        auto fpsDen{ vsapi->mapGetInt(in, "fps_den", 0, &err) };
        if (!err && fpsDen < 1)
            throw "fps_den must be at least 1";

        auto model_path{ vsapi->mapGetData(in, "model_path", 0, &err) };
        std::string modelPath{ err ? "" : model_path };

        auto gpuId{ vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err) };
        if (err)
            gpuThread = 2;

        auto tta{ !!vsapi->mapGetInt(in, "tta", 0, &err) };
        auto uhd{ !!vsapi->mapGetInt(in, "uhd", 0, &err) };
        d->sceneChange = !!vsapi->mapGetInt(in, "sc", 0, &err);
        d->skip = !!vsapi->mapGetInt(in, "skip", 0, &err);

        d->skipThreshold = vsapi->mapGetFloat(in, "skip_threshold", 0, &err);
        if (err)
            d->skipThreshold = 60.0;

        if (model < 0 || model > 61)
            throw "model must be between 0 and 59 (inclusive)";

        if (factorNum < 1)
            throw "factor_num must be at least 1";

        if (factorDen < 1)
            throw "factor_den must be at least 1";

        if (fpsNum && fpsDen && !(d->vi.fpsNum && d->vi.fpsDen))
            throw "clip does not have a valid frame rate and hence fps_num and fps_den cannot be used";

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";

        if (auto queueCount{ ncnn::get_gpu_info(gpuId).compute_queue_count() }; static_cast<uint32_t>(gpuThread) > queueCount)
            std::cerr << "Warning: gpu_thread is recommended to be between 1 and " << queueCount << " (inclusive)" << std::endl;
        
        if (auto queueCount{ ncnn::get_gpu_info(gpuId).compute_queue_count() }; gpuThread < 1)
            throw "gpu_thread must be greater than 0";

        
        if (d->skipThreshold < 0 || d->skipThreshold > 60)
            throw "skip_threshold must be between 0.0 and 60.0 (inclusive)";

        if (fpsNum && fpsDen) {
            vsh::muldivRational(&fpsNum, &fpsDen, d->vi.fpsDen, d->vi.fpsNum);
            d->factorNum = fpsNum;
            d->factorDen = fpsDen;
        } else {
            d->factorNum = factorNum;
            d->factorDen = factorDen;
        }
        vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, d->factorNum, d->factorDen);

        if (d->vi.numFrames < 2)
            throw "clip's number of frames must be at least 2";

        if (d->vi.numFrames / d->factorDen > INT_MAX / d->factorNum)
            throw "resulting clip is too long";

        auto oldNumFrames{ d->vi.numFrames };
        d->vi.numFrames = static_cast<int>(d->vi.numFrames * d->factorNum / d->factorDen);

        d->factor = d->factorNum / d->factorDen;

        if (!!vsapi->mapGetInt(in, "list_gpu", 0, &err)) {
            std::string text;

            for (auto i{ 0 }; i < ncnn::get_gpu_count(); i++)
                text += std::to_string(i) + ": " + ncnn::get_gpu_info(i).device_name() + "\n";

            auto args{ vsapi->createMap() };
            vsapi->mapConsumeNode(args, "clip", d->node, maReplace);
            vsapi->mapSetData(args, "text", text.c_str(), -1, dtUtf8, maReplace);

            auto ret{ vsapi->invoke(vsapi->getPluginByID(VSH_TEXT_PLUGIN_ID, core), "Text", args) };
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            vsapi->mapConsumeNode(out, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);

            if (--numGPUInstances == 0)
                ncnn::destroy_gpu_instance();
            return;
        }

        if (modelPath.empty()) {
            std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginByID("com.holywu.rife", core)) };
            modelPath = pluginPath.substr(0, pluginPath.rfind('/')) + "/models";

            switch (model) {
            case 0:
                modelPath += "/rife";
                break;
            case 1:
                modelPath += "/rife-HD";
                break;
            case 2:
                modelPath += "/rife-UHD";
                break;
            case 3:
                modelPath += "/rife-anime";
                break;
            case 4:
                modelPath += "/rife-v2";
                break;
            case 5:
                modelPath += "/rife-v2.3";
                break;
            case 6:
                modelPath += "/rife-v2.4";
                break;
            case 7:
                modelPath += "/rife-v3.0";
                break;
            case 8:
                modelPath += "/rife-v3.1";
                break;
            
            case 9:
                modelPath += "/rife-v3.9_ensembleFalse_fastTrue";
                break;
            case 10:
                modelPath += "/rife-v3.9_ensembleTrue_fastFalse";
                break;
            case 11:
                modelPath += "/rife-v4_ensembleFalse_fastTrue";
                break;
            case 12:
                modelPath += "/rife-v4_ensembleTrue_fastFalse";
                break;
            case 13:
                modelPath += "/rife-v4.1_ensembleFalse_fastTrue";
                break;
            case 14:
                modelPath += "/rife-v4.1_ensembleTrue_fastFalse";
                break;
            case 15:
                modelPath += "/rife-v4.2_ensembleFalse_fastTrue";
                break;
            case 16:
                modelPath += "/rife-v4.2_ensembleTrue_fastFalse";
                break;
            case 17:
                modelPath += "/rife-v4.3_ensembleFalse_fastTrue";
                break;
            case 18:
                modelPath += "/rife-v4.3_ensembleTrue_fastFalse";
                break;
            case 19:
                modelPath += "/rife-v4.4_ensembleFalse_fastTrue";
                break;
            case 20:
                modelPath += "/rife-v4.4_ensembleTrue_fastFalse";
                break;
            case 21:
                modelPath += "/rife-v4.5_ensembleFalse";
                break;
            case 22:
                modelPath += "/rife-v4.5_ensembleTrue";
                break;
            case 23:
                modelPath += "/rife-v4.6_ensembleFalse";
                break;
            case 24:
                modelPath += "/rife-v4.6_ensembleTrue";
                break;
            case 25:
                modelPath += "/rife-v4.7_ensembleFalse";
                break;
            case 26:
                modelPath += "/rife-v4.7_ensembleTrue";
                break;
            case 27:
                modelPath += "/rife-v4.8_ensembleFalse";
                break;
            case 28:
                modelPath += "/rife-v4.8_ensembleTrue";
                break;
            case 29:
                modelPath += "/rife-v4.9_ensembleFalse";
                break;
            case 30:
                modelPath += "/rife-v4.9_ensembleTrue";
                break;
            case 31:
                modelPath += "/rife-v4.10_ensembleFalse";
                break;
            case 32:
                modelPath += "/rife-v4.10_ensembleTrue";
                break;
            case 33:
                modelPath += "/rife-v4.11_ensembleFalse";
                break;
            case 34:
                modelPath += "/rife-v4.11_ensembleTrue";
                break;
            case 35:
                modelPath += "/rife-v4.12_ensembleFalse";
                break;
            case 36:
                modelPath += "/rife-v4.12_ensembleTrue";
                break;
            case 37:
                modelPath += "/rife-v4.12_lite_ensembleFalse";
                break;
            case 38:
                modelPath += "/rife-v4.12_lite_ensembleTrue";
                break;
            case 39:
                modelPath += "/rife-v4.13_ensembleFalse";
                break;
            case 40:
                modelPath += "/rife-v4.13_ensembleTrue";
                break;
            case 41:
                modelPath += "/rife-v4.13_lite_ensembleFalse";
                break;
            case 42:
                modelPath += "/rife-v4.13_lite_ensembleTrue";
                break;
            case 43:
                modelPath += "/rife-v4.14_ensembleFalse";
                break;
            case 44:
                modelPath += "/rife-v4.14_ensembleTrue";
                break;
            case 45:
                modelPath += "/rife-v4.14_lite_ensembleFalse";
                break;
            case 46:
                modelPath += "/rife-v4.14_lite_ensembleTrue";
                break;
            case 47:
                modelPath += "/rife-v4.15_ensembleFalse";
                break;
            case 48:
                modelPath += "/rife-v4.15_ensembleTrue";
                break;
            case 49:
                modelPath += "/rife-v4.15_lite_ensembleFalse";
                break;
            case 50:
                modelPath += "/rife-v4.15_lite_ensembleTrue";
                break;
            case 51:
                modelPath += "/rife-v4.16_lite_ensembleFalse";
                break;
            case 52:
                modelPath += "/rife-v4.16_lite_ensembleTrue";
                break;
            case 53:
                modelPath += "/rife-v4.17_ensembleFalse";
                break;
            case 54:
                modelPath += "/rife-v4.17_ensembleTrue";
                break;
            case 55:
                modelPath += "/rife-v4.17_lite_ensembleFalse";
                break;
            case 56:
                modelPath += "/rife-v4.17_lite_ensembleTrue";
                break;
            case 57:
                modelPath += "/rife-v4.18_ensembleFalse";
                break;
            case 58:
                modelPath += "/rife-v4.18_ensembleTrue";
                break;
            case 59:
                modelPath += "/sudo_rife4_ensembleFalse_fastTrue";
                break;
            case 60:
                modelPath += "/sudo_rife4_ensembleTrue_fastFalse";
                break;
            case 61:
                modelPath += "/sudo_rife4_ensembleTrue_fastTrue";
                break;
            
            }
        }

        std::ifstream ifs{ modelPath + "/flownet.param" };
        if (!ifs.is_open())
            throw "failed to load model";
        ifs.close();

        bool rife_v2{};
        bool rife_v4{};

        if (modelPath.find("rife-v2") != std::string::npos)
            rife_v2 = true;
        else if (modelPath.find("rife-v3.9") != std::string::npos)
            rife_v4 = true;
        
        else if (modelPath.find("rife-v3") != std::string::npos)
            rife_v2 = true;
        else if (modelPath.find("rife-v4") != std::string::npos)
            rife_v4 = true;
        else if (modelPath.find("rife4") != std::string::npos)
            rife_v4 = true;
        else if (modelPath.find("rife") == std::string::npos)
            throw "unknown model dir type";

        if (!rife_v4 && (d->factorNum != 2 || d->factorDen != 1))
            throw "only rife-v4 model supports custom frame rate";

        if (rife_v4 && tta)
            throw "rife-v4 model does not support TTA mode";

        d->semaphore = std::make_unique<std::counting_semaphore<>>(gpuThread);

        if (d->skip) {
            auto vmaf{ vsapi->getPluginByID("com.holywu.vmaf", core) };

            if (!vmaf)
                throw "VMAF plugin is required when skip=True";

            auto args{ vsapi->createMap() };
            vsapi->mapConsumeNode(args, "clip", d->node, maReplace);
            vsapi->mapSetInt(args, "width", std::min(d->vi.width, 512), maReplace);
            vsapi->mapSetInt(args, "height", std::min(d->vi.height, 512), maReplace);
            vsapi->mapSetInt(args, "format", pfYUV420P8, maReplace);
            vsapi->mapSetData(args, "matrix_s", "709", -1, dtUtf8, maReplace);

            auto ret{ vsapi->invoke(vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Bicubic", args) };
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            vsapi->clearMap(args);
            auto reference{ vsapi->mapGetNode(ret, "clip", 0, nullptr) };
            vsapi->mapSetNode(args, "clip", reference, maReplace);
            vsapi->mapSetInt(args, "frames", oldNumFrames - 1, maReplace);

            vsapi->freeMap(ret);
            ret = vsapi->invoke(vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "DuplicateFrames", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            vsapi->clearMap(args);
            vsapi->mapConsumeNode(args, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->mapSetInt(args, "first", 1, maReplace);

            vsapi->freeMap(ret);
            ret = vsapi->invoke(vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "Trim", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            vsapi->clearMap(args);
            vsapi->mapConsumeNode(args, "reference", reference, maReplace);
            vsapi->mapConsumeNode(args, "distorted", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->mapSetInt(args, "feature", 0, maReplace);

            vsapi->freeMap(ret);
            ret = vsapi->invoke(vmaf, "Metric", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            d->psnr = vsapi->mapGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
        }

        d->rife = std::make_unique<RIFE>(gpuId, tta, uhd, 1, rife_v2, rife_v4);

#ifdef _WIN32
        auto bufferSize{ MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0) };
        std::vector<wchar_t> wbuffer(bufferSize);
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wbuffer.data(), bufferSize);
        d->rife->load(wbuffer.data());
#else
        d->rife->load(modelPath);
#endif
    } catch (const char* error) {
        vsapi->mapSetError(out, ("RIFE: "s + error).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->psnr);

        if (--numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    std::vector<VSFilterDependency> deps{ {d->node, rpGeneral} };
    if (d->skip)
        deps.push_back({ d->psnr, rpGeneral });
    vsapi->createVideoFilter(out, "RIFE", &d->vi, rifeGetFrame, rifeFree, fmParallel, deps.data(), deps.size(), d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.rife", "rife", "Real-Time Intermediate Flow Estimation for Video Frame Interpolation",
                         VS_MAKE_VERSION(9, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("RIFE",
                             "clip:vnode;"
                             "model:int:opt;"
                             "factor_num:int:opt;"
                             "factor_den:int:opt;"
                             "fps_num:int:opt;"
                             "fps_den:int:opt;"
                             "model_path:data:opt;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "tta:int:opt;"
                             "uhd:int:opt;"
                             "sc:int:opt;"
                             "skip:int:opt;"
                             "skip_threshold:float:opt;"
                             "list_gpu:int:opt;",
                             "clip:vnode;",
                             rifeCreate, nullptr, plugin);
}
