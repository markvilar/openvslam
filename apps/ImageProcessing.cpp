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
