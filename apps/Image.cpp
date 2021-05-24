#include "Image.hpp"

#include <string>
#include <sstream>

std::string CVMatTypeToStr(int type) 
{
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

std::string ToString(sl::Mat& mat)
{
    std::stringstream ss;
    ss << mat.getInfos() << ", " << mat.getChannels() << ", " 
        << mat.getHeight() << ", " << mat.getWidth();
    return ss.str();
}

std::string ToString(const cv::Mat& mat)
{
    std::stringstream ss;
    ss << mat.rows << ", " << mat.cols << ", " << mat.channels() << ", " 
        << mat.total() * mat.elemSize() << ", " << CVMatTypeToStr(mat.type());
    return ss.str();
}

std::string ToString(const torch::Tensor& tensor)
{
    std::stringstream ss;
    ss << tensor.data_ptr() << ", " << tensor.sizes() << ", " 
        << tensor.strides() << ", " << tensor.nbytes();
    return ss.str();
}

// ---- Stereolabs / CV -------------------------------------------------------

auto StereolabsToCVMatType(const sl::MAT_TYPE type)
{
	switch (type) 
	{
		case sl::MAT_TYPE::F32_C1: 
            return CV_32FC1;
		case sl::MAT_TYPE::F32_C2: 
            return CV_32FC2;
		case sl::MAT_TYPE::F32_C3: 
            return CV_32FC3;
		case sl::MAT_TYPE::F32_C4: 
            return CV_32FC4;
		case sl::MAT_TYPE::U8_C1:  
            return CV_8UC1;
		case sl::MAT_TYPE::U8_C2:  
            return CV_8UC2;
		case sl::MAT_TYPE::U8_C3:  
            return CV_8UC3;
		case sl::MAT_TYPE::U8_C4:  
            return CV_8UC4;
		default: 
            throw std::logic_error("default MAT_TYPE");
	}
}

cv::Mat StereolabsToCVMat(const sl::Mat& mat) 
{
	return cv::Mat(static_cast<int>(mat.getHeight()), 
        static_cast<int>(mat.getWidth()), 
        StereolabsToCVMatType(mat.getDataType()), 
		mat.getPtr<sl::uchar1>(sl::MEM::CPU),
        mat.getStepBytes(sl::MEM::CPU));
}

cv::cuda::GpuMat StereolabsToCVGpuMat(const sl::Mat& mat)
{
    return cv::cuda::GpuMat(static_cast<int>(mat.getHeight()), 
        static_cast<int>(mat.getWidth()), 
        StereolabsToCVMatType(mat.getDataType()),
        mat.getPtr<sl::uchar1>(sl::MEM::GPU), 
        mat.getStepBytes(sl::MEM::GPU));
}

// ---- Stereolabs / Torch ----------------------------------------------------

torch::Tensor StereolabsToTensor(const sl::Mat& mat, const int device=0, 
    const bool requiresGrad=false)
{
    // std::vector<int64_t> dims = {1, mat.channels(), mat.rows, mat.cols};
    std::vector<int64_t> dims = { 1, 
        static_cast<int64_t>(mat.getChannels()), 
        static_cast<int64_t>(mat.getHeight()), 
        static_cast<int64_t>(mat.getWidth()) };
    std::vector<int64_t> strides = { 1, 1, 
        static_cast<int64_t>(mat.getStepBytes()), 
        static_cast<int64_t>(mat.getChannels()) };
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(
        torch::kCUDA, device);
    return torch::from_blob(mat.getPtr<sl::uchar1>(sl::MEM::GPU), dims, 
        strides,  options);
}

sl::Mat TensorToStereolabs(const torch::Tensor& tensor)
{
    return sl::Mat();
}

// ---- CV / Torch ------------------------------------------------------------



cv::Mat TensorToCVMat(torch::Tensor& tensor) 
{
    tensor = tensor.squeeze().detach();
    tensor = tensor.permute({1, 2, 0}).contiguous();
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    tensor = tensor.reshape({width * height * 3});
    cv::Mat mat = cv::Mat(cv::Size(width, height), CV_8UC3, 
        tensor.data_ptr<uchar>());
    return mat.clone();
}

torch::Tensor CVMatToTensor(const cv::Mat& mat)
{
    std::vector<int64_t> dims = { 1, mat.channels(), mat.rows, mat.cols };
    int64 step = mat.step; // / sizeof(float);
    std::vector<int64_t> strides = { 1, 1, step, mat.channels() };
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    return torch::from_blob(mat.data, dims, strides,  options);
}
