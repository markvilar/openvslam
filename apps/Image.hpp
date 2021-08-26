#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include <sl/Camera.hpp>

#include <assert.h>

// ---- Stereolabs / CV -------------------------------------------------------

auto StereolabsToCVMatType(const sl::MAT_TYPE type);

cv::Mat StereolabsToCVMat(const sl::Mat& mat);

cv::cuda::GpuMat StereolabsToCVGpuMat(const sl::Mat& mat);
