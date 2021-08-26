#include "ImageProcessing.hpp"

ImagePreprocessingType IntToImagePreprocessingType(const int i)
{
    auto o = ImagePreprocessingType::None;
    switch (i)
    {
    case 0:
        o = ImagePreprocessingType::None;
        break;
    case 1:
        o = ImagePreprocessingType::HistogramEqualization;
        break;
    case 2:
        o = ImagePreprocessingType::CLAHE;
        break;
    default:
        o = ImagePreprocessingType::None;
        break;
    }
    return o;
}

int ImagePreprocessingTypeToInt(const ImagePreprocessingType x)
{
    int y = 0;
    switch (x)
    {
    case ImagePreprocessingType::None:
        y = 0;
        break;
    case ImagePreprocessingType::HistogramEqualization:
        y = 1;
        break;
    case ImagePreprocessingType::CLAHE:
        y = 2;
        break;
    default:
        y = 0;
        break;
    }
    return y;
}

std::string ToString(const ImagePreprocessingType x)
{
    std::string y;
    switch (x)
    {
    case ImagePreprocessingType::None:
        y = "None";
        break;
    case ImagePreprocessingType::HistogramEqualization:
        y = "HE";
        break;
    case ImagePreprocessingType::CLAHE:
        y = "CLAHE";
        break;
    default:
        y = "None";
        break;
    }
    return y;
}

std::ostream& operator<<(std::ostream& os, const ImagePreprocessingType x)
{
    return os << ToString(x);
}


std::pair<cv::Mat, cv::Mat> StereoImageProcessor::ProcessImagePairRaw(
    const std::shared_ptr<sl::Camera>& camera)
{
	sl::Mat leftImage, rightImage;

    camera->retrieveImage(leftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);
    camera->retrieveImage(rightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);

    auto leftImageGray = StereolabsToCVMat(leftImage);
    auto rightImageGray = StereolabsToCVMat(rightImage);

    return { leftImageGray.clone(), rightImageGray.clone() };
}

std::pair<cv::Mat, cv::Mat> StereoImageProcessor::ProcessImagePairHE(
    const std::shared_ptr<sl::Camera>& camera)
{
   	sl::Mat leftImage, rightImage;
	cv::Mat leftImageEqualized, rightImageEqualized;

    camera->retrieveImage(leftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);
    camera->retrieveImage(rightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);

    auto leftImageGray = StereolabsToCVMat(leftImage);
    auto rightImageGray = StereolabsToCVMat(rightImage);

    openvslam::util::histogram_equalization(leftImageGray, 
        leftImageEqualized);
    openvslam::util::histogram_equalization(rightImageGray, 
        rightImageEqualized);

    return { leftImageEqualized, rightImageEqualized };
}

std::pair<cv::Mat, cv::Mat> StereoImageProcessor::ProcessImagePairCLAHE(
    const std::shared_ptr<sl::Camera>& camera, const double clipLimit, 
    const int size)
{
   	sl::Mat leftImage, rightImage;
	cv::Mat leftImageEqualized, rightImageEqualized;

    camera->retrieveImage(leftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);
    camera->retrieveImage(rightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);

    auto leftImageGray = StereolabsToCVMat(leftImage);
    auto rightImageGray = StereolabsToCVMat(rightImage);

    openvslam::util::clahe(leftImageGray, leftImageEqualized, clipLimit, 
        { size, size });

    openvslam::util::clahe(rightImageGray, rightImageEqualized, clipLimit, 
        { size, size });

    return { leftImageEqualized, rightImageEqualized };
}

void StereoImageProcessor::FilterImageBilateral(const cv::Mat& src, 
    cv::Mat& dst, const uint32_t diameter, const uint32_t sigmaColor, 
    const uint32_t sigmaSpace)
{
    cv::bilateralFilter(src, dst, diameter, sigmaColor, sigmaSpace);
}
