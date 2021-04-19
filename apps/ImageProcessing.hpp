#pragma once

enum class ImagePreprocessingType : uint8_t
{
    None                  = 0,
    HistogramEqualization = 1,
    Clahe                 = 2,
};
