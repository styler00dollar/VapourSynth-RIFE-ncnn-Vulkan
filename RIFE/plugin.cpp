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

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "rife.h"

using namespace std::literals;

static std::atomic<int> numGPUInstances{ 0 };

struct RIFEData final {
    VSNode* node;
    VSVideoInfo vi;
    bool sceneChange;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
};

static void filter(const VSFrame* src0, const VSFrame* src1, VSFrame* dst, const RIFEData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
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
    d->rife->process(src0R, src0G, src0B, src1R, src1G, src1B, dstR, dstG, dstB, width, height, stride);
    d->semaphore->release();
}

static const VSFrame* VS_CC rifeGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEData*>(instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n / 2, d->node, frameCtx);
        if ((n & 1) && n < d->vi.numFrames - 2)
            vsapi->requestFrameFilter(n / 2 + 1, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src0{ vsapi->getFrameFilter(n / 2, d->node, frameCtx) };
        decltype(src0) src1{ nullptr };
        VSFrame* dst{ nullptr };

        if ((n & 1) && n < d->vi.numFrames - 2) {
            auto sceneChange{ false };

            if (d->sceneChange) {
                auto props{ vsapi->getFramePropertiesRO(src0) };
                int err;
                sceneChange = !!vsapi->mapGetInt(props, "_SceneChangeNext", 0, &err);
            }

            if (sceneChange) {
                dst = vsapi->copyFrame(src0, core);
            } else {
                src1 = vsapi->getFrameFilter(n / 2 + 1, d->node, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src0, core);
                filter(src0, src1, dst, d, vsapi);
            }
        } else {
            dst = vsapi->copyFrame(src0, core);
        }

        auto props{ vsapi->getFramePropertiesRW(dst) };
        int errNum, errDen;
        auto durationNum{ vsapi->mapGetInt(props, "_DurationNum", 0, &errNum) };
        auto durationDen{ vsapi->mapGetInt(props, "_DurationDen", 0, &errDen) };
        if (!errNum && !errDen) {
            vsh::muldivRational(&durationNum, &durationDen, 1, 2);
            vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
            vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
        }

        vsapi->freeFrame(src0);
        vsapi->freeFrame(src1);
        return dst;
    }

    return nullptr;
}

static void VS_CC rifeFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEData*>(instanceData) };
    vsapi->freeNode(d->node);
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

        auto gpuId{ vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err) };
        if (err)
            gpuThread = 2;

        auto tta{ !!vsapi->mapGetInt(in, "tta", 0, &err) };
        auto uhd{ !!vsapi->mapGetInt(in, "uhd", 0, &err) };
        d->sceneChange = !!vsapi->mapGetInt(in, "sc", 0, &err);

        if (model < 0 || model > 2)
            throw "model must be 0, 1, or 2";

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";

        if (gpuThread < 1 || static_cast<uint32_t>(gpuThread) > ncnn::get_gpu_info(gpuId).compute_queue_count())
            throw ("gpu_thread must be between 1 and " + std::to_string(ncnn::get_gpu_info(gpuId).compute_queue_count()) + " (inclusive)").c_str();

        if (d->vi.numFrames < 2)
            throw "number of frames must be at least 2";

        if (d->vi.numFrames > INT_MAX / 2)
            throw "resulting clip is too long";

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

        d->vi.numFrames *= 2;
        vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, 2, 1);

        std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginByID("com.holywu.rife", core)) };
        auto modelPath{ pluginPath.substr(0, pluginPath.rfind('/')) + "/models" };
        switch (model) {
        case 0:
            modelPath += "/rife-v3.1";
            break;
        case 1:
            modelPath += "/rife-v2.4";
            break;
        case 2:
            modelPath += "/rife-anime";
            break;
        }

        std::ifstream ifs{ modelPath + "/contextnet.param" };
        if (!ifs.is_open())
            throw "failed to load model";
        ifs.close();

        d->rife = std::make_unique<RIFE>(gpuId, tta, uhd, 1, model != 2);

#ifdef _WIN32
        auto bufferSize{ MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0) };
        std::vector<wchar_t> wbuffer(bufferSize);
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wbuffer.data(), bufferSize);
        d->rife->load(wbuffer.data());
#else
        d->rife->load(modelPath);
#endif

        d->semaphore = std::make_unique<std::counting_semaphore<>>(gpuThread);
    } catch (const char* error) {
        vsapi->mapSetError(out, ("RIFE: "s + error).c_str());
        vsapi->freeNode(d->node);

        if (--numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    VSFilterDependency deps[]{ {d->node, rpGeneral} };
    vsapi->createVideoFilter(out, "RIFE", &d->vi, rifeGetFrame, rifeFree, fmParallel, deps, 1, d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.rife", "rife", "Real-Time Intermediate Flow Estimation for Video Frame Interpolation", VS_MAKE_VERSION(2, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("RIFE",
                             "clip:vnode;"
                             "model:int:opt;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "tta:int:opt;"
                             "uhd:int:opt;"
                             "sc:int:opt;"
                             "list_gpu:int:opt;",
                             "clip:vnode;",
                             rifeCreate, nullptr, plugin);
}
