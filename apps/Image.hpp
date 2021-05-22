#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <torch/script.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include <sl/Camera.hpp>

#include <assert.h>

auto StereolabsToCVMatType(const sl::MAT_TYPE type);

cv::Mat StereolabsToCVMat(const sl::Mat& input);

cv::cuda::GpuMat StereolabsToCVGpuMat(const sl::Mat& input);

torch::Tensor CVMatToTensor(const cv::cuda::GpuMat& mat);

cv::cuda::GpuMat TensorToCVMat(const torch::Tensor& tensor);
