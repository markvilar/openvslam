#pragma once

#include <chrono>
#include <fstream>
#include <memory>

#include <opencv2/core/core.hpp>

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/yaml.h"
#include "openvslam/util/stereo_rectifier.h"


#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif

#include <torch/script.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include "sl/Camera.hpp"

#include "ApplicationConfiguration.hpp"
#include "Image.hpp"
#include "ImageProcessing.hpp"

class Application
{
public:
    Application(const ApplicationConfiguration& configuration);
    ~Application();

    Application* Get() { return s_Instance; }

private:
    std::tuple<cv::Mat, cv::Mat, double> ProcessImages(
        const std::shared_ptr<sl::Camera>& camera);

    void Run();
    void TrackFrames();

    void SaveTrackTimes();
    void SaveProcessTimes();

private:
    ApplicationConfiguration m_Configuration;
    std::shared_ptr<openvslam::system> m_System;

#ifdef USE_PANGOLIN_VIEWER
    std::shared_ptr<pangolin_viewer::viewer> m_Viewer;
#endif
    // Tracking.
    std::unique_ptr<std::thread> m_TrackingThread;
    std::shared_ptr<std::vector<float>> m_TrackTimes;
    std::shared_ptr<std::vector<float>> m_ProcessTimes;

    // Rectifier.
    openvslam::util::stereo_rectifier m_Rectifier;

    // UIENet.
    std::shared_ptr<torch::jit::script::Module> m_UIENet;

    static Application* s_Instance;
    friend int main(int argc, char** argv);
};
