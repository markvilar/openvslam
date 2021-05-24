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
    case 3:
        o = ImagePreprocessingType::UIENet;
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
    case ImagePreprocessingType::UIENet:
        y = 3;
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
    case ImagePreprocessingType::UIENet:
        y = "UIENet";
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

std::pair<cv::Mat, cv::Mat> StereoImageProcessor::ProcessImagePairUIENet(
    const std::shared_ptr<sl::Camera>& camera,
    const std::shared_ptr<torch::jit::script::Module>& model)
{
   	sl::Mat leftImage, rightImage;
    sl::Mat leftImageTest, rightImageTest;

    camera->retrieveImage(leftImage, sl::VIEW::LEFT_UNRECTIFIED, 
        sl::MEM::GPU);
    camera->retrieveImage(rightImage, sl::VIEW::RIGHT_UNRECTIFIED, 
        sl::MEM::GPU);

    auto leftTensorBGRA = StereolabsToTensor(leftImage, 0, false);
    auto rightTensorBGRA = StereolabsToTensor(rightImage, 0, false);

    // Slice to BGR.
    auto leftTensorBGR = leftTensorBGRA.slice(1, 0, 3);
    auto rightTensorBGR = rightTensorBGRA.slice(1, 0, 3);

    // Convert to RGB.
    auto leftTensorRGB = leftTensorBGR.flip(1);
    auto rightTensorRGB = rightTensorBGR.flip(1);

    // Convert to float.
    auto leftInputTensor = leftTensorRGB / 255.0f;
    auto rightInputTensor = rightTensorRGB / 255.0f;

    // Input tensor to model.
    std::vector<torch::jit::IValue> leftInput;
    leftInput.push_back(leftInputTensor);
    auto leftOutputGPU = model->forward(leftInput).toTensor();
    std::vector<torch::jit::IValue> rightInput;
    rightInput.push_back(rightInputTensor);
    auto rightOutputGPU = model->forward(rightInput).toTensor();

    // Transfer to CPU, scale, convert to uint8 and OpenCV.
    auto leftOutput = TensorToCVMat(leftOutputGPU);
    auto rightOutput = TensorToCVMat(rightOutputGPU);

    return { leftOutput.clone(), rightOutput.clone() };
}

void StereoImageProcessor::FilterImageBilateral(const cv::Mat& src, 
    cv::Mat& dst, const uint32_t diameter, const uint32_t sigmaColor, 
    const uint32_t sigmaSpace)
{
    cv::bilateralFilter(src, dst, diameter, sigmaColor, sigmaSpace);
}
