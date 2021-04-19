#pragma once

#include "ImageProcessing.hpp"

struct ZEDSlamConfiguration
{
public:
    std::string VocabularyPath;
    std::string SVOPath;
    std::string ConfigurationPath;
    std::string MapPath;

    uint32_t FrameSkip = 0;
    uint32_t FrameStart = 0;

    bool Realtime = false;
    bool Terminate = false;
    bool Debug = false;
    bool Evaluate = false;

    ImagePreprocessingType Preprocessing = ImagePreprocessingType::None;
};
