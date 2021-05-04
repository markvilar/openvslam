#pragma once

#include <chrono>
#include <fstream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"
#include "openvslam/util/image_converter.h"

#include "pangolin_viewer/viewer.h"

#include "ApplicationConfiguration.hpp"
#include "Utilities.hpp"

class Application
{
public:
    Application(const ApplicationConfiguration& configuration);
    ~Application();

    Application* Get() { return s_Instance; }

private:
    void Run();
    void TrackFrames();

private:
    ApplicationConfiguration m_Configuration;
    std::shared_ptr<openvslam::system> m_System;
    std::shared_ptr<pangolin_viewer::viewer> m_Viewer;

    // Tracking.
    std::unique_ptr<std::thread> m_TrackingThread;
    std::shared_ptr<std::vector<float>> m_TrackTimes;
    openvslam::util::stereo_rectifier m_Rectifier;

    static Application* s_Instance;
    friend int main(int argc, char** argv);
};
