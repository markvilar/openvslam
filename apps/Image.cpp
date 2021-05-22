#include "Image.hpp"

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

torch::Tensor StereolabsToTensor(const sl::Mat& mat)
{
    // std::vector<int64_t> dims = {1, mat.channels(), mat.rows, mat.cols};
    std::vector<int64_t> dims = { 1, mat.getChannels(), mat.getHeight(), 
        mat.getWidth() };
    // int64 step = im.step; // / sizeof(float);
    int64 step = mat.getStep();
    std::vector<int64_t> strides = { 1, 1, step, mat.getChannels() };
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(
        torch::kCUDA);
    return torch::from_blob(mat.getPtr<sl::uchar1>(sl::MEM::GPU), dims, 
        strides,  options);
}

sl::Mat TensorToStereolabs(const torch::Tensor& tensor)
{
}

// ---- CV / Torch ------------------------------------------------------------



cv::Mat TensorToCVMat(const torch::Tensor& tensor) 
{
    auto output = tensor.permute({ 0, 2, 3, 1 });
    output = output.squeeze(0).detach();
    output = output.to(torch::kU8).to(torch::kCPU);

    auto sizes = output.sizes();
    cv::Mat mat{cv::Size{static_cast<int>(sizes[1]), 
        static_cast<int>(sizes[0]) }, CV_8UC(static_cast<int>(sizes[2])), 
        output.data_ptr()};

    return mat;
}

torch::Tensor CVMatToTensor(const cv::Mat& mat)
{
    std::vector<int64_t> dims = { 1, mat.channels(), mat.rows, mat.cols };
    int64 step = mat.step; // / sizeof(float);
    std::vector<int64_t> strides = { 1, 1, step, mat.channels() };
    auto options = torch::TensorOptions().dtype(torch::kUInt8)
        .device(torch::kCUDA);
    return torch::from_blob(mat.data, dims, strides,  options);
}
