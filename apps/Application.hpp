#pragma once

#include <chrono>
#include <fstream>
#include <memory>

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"
#include "openvslam/util/image_converter.h"

#include "pangolin_viewer/viewer.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include "sl/Camera.hpp"

#include "ApplicationConfiguration.hpp"
#include "Utilities.hpp"

class Application
{
public:
    Application(const ApplicationConfiguration& configuration);
    ~Application();

    Application* Get() { return s_Instance; }

private:
    std::tuple<cv::Mat, cv::Mat, double> ProcessImages(
        const std::shared_ptr<sl::Camera>& camera);
    void ProcessImageRaw(const cv::Mat& src, cv::Mat& dst);
    void ProcessImageHE(const cv::Mat& src, cv::Mat& dst);
    void ProcessImageCLAHE(const cv::Mat& src, cv::Mat& dst);
    void ProcessImageUIENet(const cv::Mat& src, cv::Mat& dst);
    void FilterImageBilateral(const cv::Mat& src, cv::Mat& dst);

    void Run();
    void TrackFrames();

    void SaveTrackTimes();
    void SaveProcessTimes();

private:
    ApplicationConfiguration m_Configuration;
    std::shared_ptr<openvslam::system> m_System;
    std::shared_ptr<pangolin_viewer::viewer> m_Viewer;

    // Tracking.
    std::unique_ptr<std::thread> m_TrackingThread;
    std::shared_ptr<std::vector<float>> m_TrackTimes;
    std::shared_ptr<std::vector<float>> m_ProcessTimes;
    openvslam::util::stereo_rectifier m_Rectifier;

    static Application* s_Instance;
    friend int main(int argc, char** argv);
};
