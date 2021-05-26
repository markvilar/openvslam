#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <torch/script.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include <sl/Camera.hpp>

#include <assert.h>

// TODO: Remove!
std::string ToString(sl::Mat& mat);
std::string ToString(const cv::Mat& mat);
std::string ToString(const torch::Tensor& tensor);

// ---- Stereolabs / CV -------------------------------------------------------

auto StereolabsToCVMatType(const sl::MAT_TYPE type);

cv::Mat StereolabsToCVMat(const sl::Mat& mat);

cv::cuda::GpuMat StereolabsToCVGpuMat(const sl::Mat& mat);

// ---- Stereolabs / Torch ----------------------------------------------------

torch::Tensor StereolabsToTensor(const sl::Mat& mat, const int device,
    const bool requiresGrad);

sl::Mat TensorToStereolabs(const torch::Tensor& tensor);

// ---- CV / Torch ------------------------------------------------------------

cv::Mat TensorToCVMat(torch::Tensor& tensor);

torch::Tensor CVMatToTensor(const cv::Mat& mat);
