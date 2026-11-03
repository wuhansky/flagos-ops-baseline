// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <string>
static thread_local std::string error;
extern "C" const char* opencv_version() { return CV_VERSION; }
extern "C" const char* opencv_build_info() { return cv::getBuildInformation().c_str(); }
extern "C" const char* opencv_error() { return error.c_str(); }
extern "C" int opencv_add(void* a, void* b, void* out, int h, int w,
                          int type, int device, void* stream_ptr) {
    try {
        cv::cuda::setDevice(device);
        cv::cuda::GpuMat x(h, w, type, a), y(h, w, type, b), z(h, w, type, out);
        auto stream = cv::cuda::StreamAccessor::wrapStream(
            reinterpret_cast<cudaStream_t>(stream_ptr));
        cv::cuda::add(x, y, z, cv::noArray(), -1, stream);
        return 0;
    } catch (const std::exception& exc) {
        error = exc.what();
        return 1;
    }
}
