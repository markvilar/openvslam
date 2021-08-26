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
