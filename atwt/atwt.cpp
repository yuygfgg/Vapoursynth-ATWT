#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <vector>

#include "VSHelper4.h"
#include "VapourSynth4.h"

namespace {
constexpr std::array<int, 5> KERNEL = {1, 4, 6, 4, 1};

template <typename T>
concept PixelType = std::integral<T> || std::floating_point<T>;

template <typename T>
constexpr float get_neutral(const VSVideoFormat* fi) noexcept {
    if constexpr (std::floating_point<T>) {
        return 0.0F;
    } else {
        return static_cast<float>(1 << (fi->bitsPerSample - 1));
    }
}

template <typename T>
constexpr float get_max(const VSVideoFormat* fi) noexcept {
    if constexpr (std::floating_point<T>) {
        return 1.0F;
    } else {
        return static_cast<float>((1LL << fi->bitsPerSample) - 1);
    }
}

// 101 reflection
constexpr int mirror_boundary(int pos, int max_pos) noexcept {
    if (pos < 0) {
        return -pos;
    }
    if (pos >= max_pos) {
        return (2 * max_pos) - 2 - pos;
    }
    return pos;
}

template <typename T>
void conv_h(const T* src, float* dst, int width, int height,
            ptrdiff_t src_stride, int step) {
    for (int y = 0; y < height; ++y) {
        const T* src_row = src + (y * src_stride);
        float* dst_row = dst + (y * width);

        for (int x = 0; x < width; ++x) {
            float sum = 0.0F;
            for (int k = -2; k <= 2; ++k) {
                int offset_idx = mirror_boundary(x + (k * step), width);
                sum +=
                    static_cast<float>(src_row[offset_idx]) * KERNEL.at(k + 2);
            }
            dst_row[x] = sum;
        }
    }
}

template <typename T>
void conv_v_and_extract(const float* temp_src, const T* orig_src, T* dst,
                        int width, int height, ptrdiff_t src_stride,
                        ptrdiff_t dst_stride, int step,
                        const VSVideoFormat* fi) {

    const float neutral = get_neutral<T>(fi);
    const float max_val = get_max<T>(fi);

    for (int y = 0; y < height; ++y) {
        const T* src_row = orig_src + (y * src_stride);
        T* dst_row = dst + (y * dst_stride);

        for (int x = 0; x < width; ++x) {
            float sum = 0.0F;
            for (int k = -2; k <= 2; ++k) {
                int y_tap = mirror_boundary(y + (k * step), height);
                sum += temp_src[(y_tap * width) + x] * KERNEL.at(k + 2);
            }

            // Kernel sum is 16*16 = 256
            float blurred_pixel = sum / 256.0F;
            auto original_pixel = static_cast<float>(src_row[x]);

            float detail = original_pixel - blurred_pixel + neutral;

            if constexpr (std::integral<T>) {
                dst_row[x] = static_cast<T>(
                    std::clamp(std::round(detail), 0.0F, max_val));
            } else {
                dst_row[x] = detail;
            }
        }
    }
}

struct ATWTData {
    VSNode* node;
    VSVideoInfo vi;
    int radius;
};

struct ReplaceData {
    VSNode* base;
    VSNode* detail;
    VSVideoInfo vi;
};

template <typename T>
void process_extract_plane(const VSFrame* src, VSFrame* dst, int plane,
                           int radius, const VSVideoFormat* fi,
                           const VSAPI* vsapi) {
    const int width = vsapi->getFrameWidth(src, plane);
    const int height = vsapi->getFrameHeight(src, plane);
    const ptrdiff_t src_stride = vsapi->getStride(src, plane) / sizeof(T);
    const ptrdiff_t dst_stride = vsapi->getStride(dst, plane) / sizeof(T);

    const T* srcp = reinterpret_cast<const T*>(vsapi->getReadPtr(src, plane));
    T* dstp = reinterpret_cast<T*>(vsapi->getWritePtr(dst, plane));

    int step = 1 << (radius - 1);

    std::vector<float> temp_buffer(static_cast<size_t>(width) * height);

    conv_h<T>(srcp, temp_buffer.data(), width, height, src_stride, step);
    conv_v_and_extract<T>(temp_buffer.data(), srcp, dstp, width, height,
                          src_stride, dst_stride, step, fi);
}

const VSFrame* VS_CC ExtractGetFrame(int n, int activationReason,
                                     void* instanceData,
                                     [[maybe_unused]] void** frameData,
                                     VSFrameContext* frameCtx, VSCore* core,
                                     const VSAPI* vsapi) {
    auto* d = static_cast<ATWTData*>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat* fi = vsapi->getVideoFrameFormat(src);

        VSFrame* dst =
            vsapi->newVideoFrame(fi, vsapi->getFrameWidth(src, 0),
                                 vsapi->getFrameHeight(src, 0), src, core);

        for (int plane = 0; plane < fi->numPlanes; plane++) {
            if (fi->sampleType == stInteger) {
                switch (fi->bytesPerSample) {
                case 1:
                    process_extract_plane<uint8_t>(src, dst, plane, d->radius,
                                                   fi, vsapi);
                    break;
                case 2:
                    process_extract_plane<uint16_t>(src, dst, plane, d->radius,
                                                    fi, vsapi);
                    break;
                case 4:
                    process_extract_plane<uint32_t>(src, dst, plane, d->radius,
                                                    fi, vsapi);
                    break;
                }
            } else if (fi->sampleType == stFloat) {
                switch (fi->bytesPerSample) {
                case 4:
                    process_extract_plane<float>(src, dst, plane, d->radius, fi,
                                                 vsapi);
                    break;
                }
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }
    return nullptr;
}

void VS_CC ExtractFree(void* instanceData, [[maybe_unused]] VSCore* core,
                       const VSAPI* vsapi) {
    auto d = std::unique_ptr<ATWTData>(static_cast<ATWTData*>(instanceData));
    vsapi->freeNode(d->node);
}

void VS_CC ExtractCreate(const VSMap* in, VSMap* out,
                         [[maybe_unused]] void* userData, VSCore* core,
                         const VSAPI* vsapi) {
    auto d = std::make_unique<ATWTData>();
    int err = 0;

    d->node = vsapi->mapGetNode(in, "clip", 0, 0);
    d->vi = *vsapi->getVideoInfo(d->node);

    d->radius = vsh::int64ToIntS(vsapi->mapGetInt(in, "radius", 0, &err));
    if (err != 0) {
        d->radius = 1;
    }

    if (d->radius < 1) {
        vsapi->mapSetError(out, "ExtractFrequency: radius must be >= 1");
        vsapi->freeNode(d->node);
        return;
    }

    if (!vsh::isConstantVideoFormat(&d->vi)) {
        vsapi->mapSetError(
            out,
            "ExtractFrequency: only clips with constant format are accepted");
        vsapi->freeNode(d->node);
        return;
    }

    if (((d->vi.format.bitsPerSample < 8 || d->vi.format.bitsPerSample > 16 ||
          d->vi.format.sampleType != stInteger) &&
         (d->vi.format.bitsPerSample != 32 ||
          d->vi.format.sampleType != stFloat))) {
        vsapi->mapSetError(out, "ExtractFrequency: only 8-16 bit "
                                "integer or 32 bit float input "
                                "are accepted");
        vsapi->freeNode(d->node);
        return;
    }

    VSFilterDependency deps[] = {{d->node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "ExtractFrequency", &d->vi, ExtractGetFrame,
                             ExtractFree, fmParallel, std::data(deps), 1,
                             d.release(), core);
}

template <typename T>
void ProcessReplacePlane(const VSFrame* base, const VSFrame* detail,
                         VSFrame* dst, int plane, const VSVideoFormat* fi,
                         const VSAPI* vsapi) {
    const int width = vsapi->getFrameWidth(dst, plane);
    const int height = vsapi->getFrameHeight(dst, plane);
    const ptrdiff_t stride = vsapi->getStride(dst, plane) / sizeof(T);

    const T* basep = reinterpret_cast<const T*>(vsapi->getReadPtr(base, plane));
    const T* detailp =
        reinterpret_cast<const T*>(vsapi->getReadPtr(detail, plane));
    T* dstp = reinterpret_cast<T*>(vsapi->getWritePtr(dst, plane));

    const float neutral = get_neutral<T>(fi);
    const float max_val = get_max<T>(fi);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto b = static_cast<float>(basep[x]);
            auto d = static_cast<float>(detailp[x]);

            float val = b + d - neutral;

            if constexpr (std::integral<T>) {
                dstp[x] =
                    static_cast<T>(std::clamp(std::round(val), 0.0F, max_val));
            } else {
                dstp[x] = static_cast<T>(val);
            }
        }
        basep += stride;
        detailp += stride;
        dstp += stride;
    }
}

const VSFrame* VS_CC ReplaceGetFrame(int n, int activationReason,
                                     void* instanceData,
                                     [[maybe_unused]] void** frameData,
                                     VSFrameContext* frameCtx, VSCore* core,
                                     const VSAPI* vsapi) {
    auto* d = static_cast<ReplaceData*>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->base, frameCtx);
        vsapi->requestFrameFilter(n, d->detail, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame* base = vsapi->getFrameFilter(n, d->base, frameCtx);
        const VSFrame* detail = vsapi->getFrameFilter(n, d->detail, frameCtx);
        const VSVideoFormat* fi = vsapi->getVideoFrameFormat(base);

        VSFrame* dst =
            vsapi->newVideoFrame(fi, vsapi->getFrameWidth(base, 0),
                                 vsapi->getFrameHeight(base, 0), base, core);

        for (int plane = 0; plane < fi->numPlanes; plane++) {
            if (fi->sampleType == stInteger) {
                switch (fi->bytesPerSample) {
                case 1:
                    ProcessReplacePlane<uint8_t>(base, detail, dst, plane, fi,
                                                 vsapi);
                    break;
                case 2:
                    ProcessReplacePlane<uint16_t>(base, detail, dst, plane, fi,
                                                  vsapi);
                    break;
                case 4:
                    ProcessReplacePlane<uint32_t>(base, detail, dst, plane, fi,
                                                  vsapi);
                    break;
                }
            } else if (fi->sampleType == stFloat) {
                switch (fi->bytesPerSample) {
                case 4:
                    ProcessReplacePlane<float>(base, detail, dst, plane, fi,
                                               vsapi);
                    break;
                }
            }
        }

        vsapi->freeFrame(base);
        vsapi->freeFrame(detail);
        return dst;
    }
    return nullptr;
}

void VS_CC ReplaceFree(void* instanceData, [[maybe_unused]] VSCore* core,
                       const VSAPI* vsapi) {
    auto d =
        std::unique_ptr<ReplaceData>(static_cast<ReplaceData*>(instanceData));
    vsapi->freeNode(d->base);
    vsapi->freeNode(d->detail);
}

void VS_CC ReplaceCreate(const VSMap* in, VSMap* out,
                         [[maybe_unused]] void* userData, VSCore* core,
                         const VSAPI* vsapi) {
    auto d = std::make_unique<ReplaceData>();

    d->base = vsapi->mapGetNode(in, "base", 0, 0);
    d->detail = vsapi->mapGetNode(in, "detail", 0, 0);
    d->vi = *vsapi->getVideoInfo(d->base);
    const VSVideoInfo* vi_detail = vsapi->getVideoInfo(d->detail);

    if (!vsh::isSameVideoFormat(&d->vi.format, &vi_detail->format)) {
        vsapi->mapSetError(out,
                           "ReplaceFrequency: base and detail must have the "
                           "same format and dimensions");
        vsapi->freeNode(d->base);
        vsapi->freeNode(d->detail);
        return;
    }

    if (((d->vi.format.bitsPerSample < 8 || d->vi.format.bitsPerSample > 16 ||
          d->vi.format.sampleType != stInteger) &&
         (d->vi.format.bitsPerSample != 32 ||
          d->vi.format.sampleType != stFloat)) ||
        !vsh::isConstantVideoFormat(&d->vi)) {
        vsapi->mapSetError(out, "ReplaceFrequency: only constant 8-16 bit "
                                "integer or 32 bit float input "
                                "are accepted");
        vsapi->freeNode(d->base);
        vsapi->freeNode(d->detail);
        return;
    }

    VSFilterDependency deps[] = {{d->base, rpStrictSpatial},
                                 {d->detail, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "ReplaceFrequency", &d->vi, ReplaceGetFrame,
                             ReplaceFree, fmParallel, std::data(deps), 2,
                             d.release(), core);
}

} // namespace

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.yuygfgg.atwt", "atwt",
                         "À Trous Wavelet Transform", VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("ExtractFrequency", "clip:vnode;radius:int:opt;",
                             "clip:vnode;", ExtractCreate, nullptr, plugin);
    vspapi->registerFunction("ReplaceFrequency", "base:vnode;detail:vnode;",
                             "clip:vnode;", ReplaceCreate, nullptr, plugin);
}