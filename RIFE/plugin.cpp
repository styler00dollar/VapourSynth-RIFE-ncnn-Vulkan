/*
  MIT License

  Copyright (c) 2021 HolyWu

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

#include <cmath>

#include <atomic>
#include <fstream>
#include <memory>
#include <semaphore>
#include <string>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#include "rife.h"

using namespace std::literals;

static std::atomic<int> numGPUInstance{ 0 };

struct RIFEData final {
    VSNodeRef* node;
    VSVideoInfo vi;
    bool sceneChange;
    std::vector<int> sx;
    std::vector<float> timesteps;
    std::unique_ptr<RIFE> rife;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
};

static void filter(const VSFrameRef* src0, const VSFrameRef* src1, VSFrameRef* dst, const RIFEData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    const auto width{ vsapi->getFrameWidth(src0, 0) };
    const auto height{ vsapi->getFrameHeight(src0, 0) };
    const auto stride{ vsapi->getStride(src0, 0) / d->vi.format->bytesPerSample };
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

static void VS_CC rifeInit([[maybe_unused]] VSMap* in, [[maybe_unused]] VSMap* out, void** instanceData, VSNode* node, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEData*>(*instanceData) };
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef* VS_CC rifeGetFrame(int n, int activationReason, void** instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEData*>(*instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(d->sx[n], d->node, frameCtx);
        vsapi->requestFrameFilter(d->sx[n] + 1, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src0{ vsapi->getFrameFilter(d->sx[n], d->node, frameCtx) };
        auto src1{ vsapi->getFrameFilter(d->sx[n] + 1, d->node, frameCtx) };
        auto dst{
            d->timesteps[n] == 0.0f ?
            vsapi->copyFrame(src0, core) :
            d->timesteps[n] == 1.0f ?
            vsapi->copyFrame(src1, core) :
            nullptr };

        if (d->timesteps[n] == 0.5f) {
            auto sceneChange{ false };

            if (d->sceneChange) {
                auto props{ vsapi->getFramePropsRO(src0) };
                int err;
                sceneChange = !!vsapi->propGetInt(props, "_SceneChangeNext", 0, &err);
            }

            if (sceneChange) {
                dst = vsapi->copyFrame(src0, core);
            } else {
                dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src0, core);
                filter(src0, src1, dst, d, vsapi);
            }
        }

        auto props{ vsapi->getFramePropsRW(dst) };
        int errNum, errDen;
        auto durationNum{ vsapi->propGetInt(props, "_DurationNum", 0, &errNum) };
        auto durationDen{ vsapi->propGetInt(props, "_DurationDen", 0, &errDen) };
        if (!errNum && !errDen) {
            muldivRational(&durationNum, &durationDen, 1, 2);
            vsapi->propSetInt(props, "_DurationNum", durationNum, paReplace);
            vsapi->propSetInt(props, "_DurationDen", durationDen, paReplace);
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

    if (--numGPUInstance == 0)
        ncnn::destroy_gpu_instance();
}

static void VS_CC rifeCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{ std::make_unique<RIFEData>() };

    try {
        d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);
        int err;

        if (!isConstantFormat(&d->vi) ||
            d->vi.format->colorFamily != cmRGB ||
            d->vi.format->sampleType != stFloat ||
            d->vi.format->bitsPerSample != 32)
            throw "only constant RGB format 32 bit float input supported";

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstance;

        auto model{ int64ToIntS(vsapi->propGetInt(in, "model", 0, &err)) };

        auto gpuId{ int64ToIntS(vsapi->propGetInt(in, "gpu_id", 0, &err)) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ int64ToIntS(vsapi->propGetInt(in, "gpu_thread", 0, &err)) };
        if (err)
            gpuThread = 2;

        auto tta{ !!vsapi->propGetInt(in, "tta", 0, &err) };
        auto uhd{ !!vsapi->propGetInt(in, "uhd", 0, &err) };
        d->sceneChange = !!vsapi->propGetInt(in, "sc", 0, &err);
        auto fp32{ !!vsapi->propGetInt(in, "fp32", 0, &err) };

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

        if (!!vsapi->propGetInt(in, "list_gpu", 0, &err)) {
            std::string text;

            for (auto i{ 0 }; i < ncnn::get_gpu_count(); i++)
                text += std::to_string(i) + ": " + ncnn::get_gpu_info(i).device_name() + "\n";

            auto args{ vsapi->createMap() };
            vsapi->propSetNode(args, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);
            vsapi->propSetData(args, "text", text.c_str(), -1, paReplace);

            auto ret{ vsapi->invoke(vsapi->getPluginById("com.vapoursynth.text", core), "Text", args) };
            if (vsapi->getError(ret)) {
                vsapi->setError(out, vsapi->getError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstance == 0)
                    ncnn::destroy_gpu_instance();
                return;
            }

            d->node = vsapi->propGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
            vsapi->propSetNode(out, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);

            if (--numGPUInstance == 0)
                ncnn::destroy_gpu_instance();
            return;
        }

        auto count{ d->vi.numFrames };

        d->vi.numFrames *= 2;
        muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, 2, 1);

        d->sx.resize(d->vi.numFrames);
        d->timesteps.resize(d->vi.numFrames);

        for (auto i{ 0 }; i < d->vi.numFrames; i++) {
            auto fx{ i * 0.5f };
            auto sx{ static_cast<int>(std::floor(fx)) };
            fx -= sx;

            if (sx >= count - 1) {
                sx = count - 2;
                fx = 1.0f;
            }

            d->sx[i] = sx;
            d->timesteps[i] = fx;
        }

        std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginById("com.holywu.rife", core)) };
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
        auto wbuffer{ std::make_unique<wchar_t[]>(bufferSize) };
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wbuffer.get(), bufferSize);
        d->rife->load(wbuffer.get(), fp32);
#else
        d->rife->load(modelPath, fp32);
#endif

        d->semaphore = std::make_unique<std::counting_semaphore<>>(gpuThread);
    } catch (const char* error) {
        vsapi->setError(out, ("RIFE: "s + error).c_str());
        vsapi->freeNode(d->node);

        if (--numGPUInstance == 0)
            ncnn::destroy_gpu_instance();
        return;
    }

    vsapi->createFilter(in, out, "RIFE", rifeInit, rifeGetFrame, rifeFree, fmParallel, 0, d.release(), core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin) {
    configFunc("com.holywu.rife", "rife", "Real-Time Intermediate Flow Estimation for Video Frame Interpolation", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("RIFE",
                 "clip:clip;"
                 "model:int:opt;"
                 "gpu_id:int:opt;"
                 "gpu_thread:int:opt;"
                 "tta:int:opt;"
                 "uhd:int:opt;"
                 "sc:int:opt;"
                 "fp32:int:opt;"
                 "list_gpu:int:opt;",
                 rifeCreate, nullptr, plugin);
}
