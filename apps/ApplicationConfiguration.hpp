#pragma once

#include <memory>

#include "openvslam/config.h"

#include "ImageProcessing.hpp"

struct ApplicationConfiguration
{
public:
    std::string VocabularyPath;
    std::string SVOPath;
    std::string ConfigurationPath;

    std::string MapPath;
    std::string FrameTrajectoryPath;
    std::string KeyframeTrajectoryPath;
    std::string TrackTimesPath;

    uint32_t FrameSkip = 1;
    uint32_t FrameStart = 0;

    bool Realtime = false;
    bool Terminate = false;
    bool Debug = false;
    bool Evaluate = false;

    ImagePreprocessingType Preprocessing = ImagePreprocessingType::None;

    std::shared_ptr<openvslam::config> OpenVSLAMConfiguration;
};

std::ostream& operator<<(std::ostream& os, 
    const ApplicationConfiguration& configuration);
