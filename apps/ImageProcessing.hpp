#pragma once

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/highgui.hpp>

#include "openvslam/util/image_converter.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include "sl/Camera.hpp"

#include "Image.hpp"

enum class ImagePreprocessingType
{
    None                  = 0,
    HistogramEqualization = 1,
    CLAHE                 = 2,
};

ImagePreprocessingType IntToImagePreprocessingType(const int x);
int ImagePreprocessingTypeToInt(const ImagePreprocessingType x);

std::string ToString(const ImagePreprocessingType x);
std::ostream& operator<<(std::ostream& os, const ImagePreprocessingType x);

class StereoImageProcessor
{
public:
    static std::pair<cv::Mat, cv::Mat> ProcessImagePairRaw(
        const std::shared_ptr<sl::Camera>& camera);
    static std::pair<cv::Mat, cv::Mat> ProcessImagePairHE(
        const std::shared_ptr<sl::Camera>& camera);
    static std::pair<cv::Mat, cv::Mat> ProcessImagePairCLAHE(
        const std::shared_ptr<sl::Camera>& camera, const double clipLimit, 
        const int size);

    static void FilterImageBilateral(const cv::Mat& src, cv::Mat& dst, 
        const uint32_t diameter, const uint32_t sigmaColor, 
        const uint32_t sigmaSpace);
};
