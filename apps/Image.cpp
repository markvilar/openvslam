#include "Image.hpp"

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

cv::Mat StereolabsToCVMat(const sl::Mat& input) 
{
	return cv::Mat(static_cast<int>(input.getHeight()), 
        static_cast<int>(input.getWidth()), 
        StereolabsToCVMatType(input.getDataType()), 
		input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

cv::cuda::GpuMat StereolabsToCVGpuMat(const sl::Mat& input)
{
    return cv::cuda::GpuMat(static_cast<int>(input.getHeight()), 
        static_cast<int>(input.getWidth()), 
        StereolabsToCVMatType(input.getDataType()),
        input.getPtr<sl::uchar1>(sl::MEM::GPU), 
        input.getStepBytes(sl::MEM::GPU));
}
