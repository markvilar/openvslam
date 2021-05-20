#pragma once

#include <iostream>
#include <string>

enum class ImagePreprocessingType
{
    None                  = 0,
    HistogramEqualization = 1,
    CLAHE                 = 2,
    UIENet                = 3,
};

ImagePreprocessingType IntToImagePreprocessingType(const int x);
int ImagePreprocessingTypeToInt(const ImagePreprocessingType x);

std::string ToString(const ImagePreprocessingType x);
std::ostream& operator<<(std::ostream& os, const ImagePreprocessingType x);
