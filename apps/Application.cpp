#include "Application.hpp"

Application* Application::s_Instance = nullptr;

Application::Application(const ApplicationConfiguration& configuration)
    : m_Configuration(configuration),
    m_Rectifier(m_Configuration.OpenVSLAMConfiguration->camera_, 
        openvslam::util::yaml_optional_ref(
            m_Configuration.OpenVSLAMConfiguration->yaml_node_, 
            "StereoRectifier"))
{
    s_Instance = this;
	m_System = std::make_shared<openvslam::system>(
        m_Configuration.OpenVSLAMConfiguration, m_Configuration.VocabularyPath);

    // Load the Torch model if we want to use it.
    if (configuration.Preprocessing == ImagePreprocessingType::UIENet)
    {
        m_UIENet = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(configuration.UIENetPath));
    }
}

Application::~Application()
{
}

std::tuple<cv::Mat, cv::Mat, double> Application::ProcessImages(
    const std::shared_ptr<sl::Camera>& camera)
{
    // Allocate images.
	cv::Mat leftImage, rightImage;
 	cv::Mat leftImageFiltered, rightImageFiltered;
	cv::Mat leftImageRectified, rightImageRectified;

    const auto processStart = std::chrono::steady_clock::now();

    switch(m_Configuration.Preprocessing)
    {
    case ImagePreprocessingType::None:
        std::tie(leftImage, rightImage) = 
            StereoImageProcessor::ProcessImagePairRaw(camera);
        break;
    case ImagePreprocessingType::HistogramEqualization:
        std::tie(leftImage, rightImage) = 
            StereoImageProcessor::ProcessImagePairHE(camera);
        break;
    case ImagePreprocessingType::CLAHE:
        std::tie(leftImage, rightImage) = 
            StereoImageProcessor::ProcessImagePairCLAHE(camera, 
            m_Configuration.CLAHEClip, m_Configuration.CLAHESize);
        break;
    case ImagePreprocessingType::UIENet:
        std::tie(leftImage, rightImage) = 
            StereoImageProcessor::ProcessImagePairUIENet(camera, m_UIENet);
        break;
    default:
        std::tie(leftImage, rightImage) = 
            StereoImageProcessor::ProcessImagePairRaw(camera);
        break;
    }

    if (m_Configuration.Filter)
    {
        // Noise reduction filter.
        StereoImageProcessor::FilterImageBilateral(leftImage, 
            leftImageFiltered, m_Configuration.FilterDiameter, 
            m_Configuration.FilterColor, m_Configuration.FilterSpace);
        StereoImageProcessor::FilterImageBilateral(rightImage, 
            rightImageFiltered, m_Configuration.FilterDiameter, 
            m_Configuration.FilterColor, m_Configuration.FilterSpace);
    }
    else
    {
        leftImageFiltered = leftImage;
        rightImageFiltered = rightImage;
    }
    
    m_Rectifier.rectify(leftImageFiltered, rightImageFiltered, 
        leftImageRectified, rightImageRectified);

    const auto processEnd = std::chrono::steady_clock::now();
    const auto processTime = std::chrono::duration_cast
        <std::chrono::duration<double>>(processEnd - processStart).count();

    return { leftImageRectified, rightImageRectified, processTime };
}

void Application::Run()
{
	// Build and start SLAM process.
	m_System->startup();

	// Create viewer.
#ifdef USE_PANGOLIN_VIEWER
	m_Viewer = std::make_shared<pangolin_viewer::viewer>(
        openvslam::util::yaml_optional_ref(
            m_Configuration.OpenVSLAMConfiguration->yaml_node_, 
            "PangolinViewer"), 
        &*m_System, m_System->get_frame_publisher(), 
        m_System->get_map_publisher());
#endif

    // Start tracking thread.
    m_TrackingThread = std::make_unique<std::thread>(&Application::TrackFrames,
        this);

	// Run the viewer in the current thread.
#ifdef USE_PANGOLIN_VIEWER
	m_Viewer->run();
#endif

    // Join tracking thread.
	m_TrackingThread->join();

	// Shutdown SLAM process.
	m_System->shutdown();

	// Output information for evaluation.
	if (m_Configuration.Evaluate) 
	{
		m_System->save_frame_trajectory(m_Configuration.FrameTrajectoryPath, 
            openvslam::io::trajectory_format::csv);
		m_System->save_keyframe_trajectory(
            m_Configuration.KeyframeTrajectoryPath, 
            openvslam::io::trajectory_format::csv);

        SaveProfilingTimes();
        SaveKeyPointData();
	}

	// Output map information.
	if (!m_Configuration.MapPath.empty()) 
	{
		m_System->save_map_database(m_Configuration.MapPath);
	}
}

void Application::TrackFrames()
{
	// InitParameters - Open SVO file.
	auto camera = std::make_shared<sl::Camera>(); 
	sl::InitParameters initParameters;
	initParameters.input.setFromSVOFile(m_Configuration.SVOPath.c_str());
	initParameters.svo_real_time_mode = m_Configuration.Realtime;

	auto cameraState= camera->open(initParameters);
	if (cameraState != sl::ERROR_CODE::SUCCESS)
	{
		std::cerr << toString(cameraState) << ": " << toVerbose(cameraState) 
            << std::endl;
		return;
	}

    // Frame information.
	unsigned int framesTotal = camera->getSVONumberOfFrames();
	unsigned int frame = m_Configuration.FrameStart;
    double timestamp = 0.0;
    double fps = camera->getInitParameters().camera_fps;

    // Profiling information.
    m_TrackTimes = std::make_shared<std::vector<float>>();
    m_ProcessTimes = std::make_shared<std::vector<float>>();

    // Keypoint information. TODO: Needs to be vector of vector.
    m_KeyPoints = std::make_shared<std::vector<cv::KeyPoint>>();
    m_KeyPoints->reserve(framesTotal);
    m_Matches = std::make_shared<std::vector<int>>(framesTotal, -1);

	if (m_Configuration.FrameStart < framesTotal)
	{
		camera->setSVOPosition(m_Configuration.FrameStart);
	}
	else
	{
		std::cerr << "Invalid start frame." << std::endl;
		return;
	}

    if (m_Configuration.FrameSkip == 0)
	{
		std::cerr << "Invalid frame skip." << std::endl;
		return;
	}

	while (frame < framesTotal - 1 and frame < m_Configuration.FrameStop) 
	{
        cameraState = camera->grab();
        if (cameraState != sl::ERROR_CODE::SUCCESS)
        {
            continue;
        }
        
        frame++;

        auto[leftImageProcessed, rightImageProcessed, processTime] = 
            ProcessImages(camera);

        if (leftImageProcessed.empty() || rightImageProcessed.empty()) 
        {
            continue;
        }

        const auto trackStart = std::chrono::steady_clock::now();

        if (frame % m_Configuration.FrameSkip == 0) 
        {
            timestamp = camera->getTimestamp(sl::TIME_REFERENCE::IMAGE)
                .getMilliseconds() / double(1000) + m_Configuration.Bias;

            m_System->feed_stereo_frame(leftImageProcessed, 
                rightImageProcessed, timestamp);
        }

        const auto trackEnd = std::chrono::steady_clock::now();

        const auto trackTime = std::chrono::duration_cast
            <std::chrono::duration<double>>(trackEnd - trackStart).count();
        if (frame % m_Configuration.FrameSkip == 0) 
        {
            m_ProcessTimes->push_back(processTime);
            m_TrackTimes->push_back(trackTime);
        }

        // Wait until the timestamp of the next frame.
        if (!m_Configuration.Realtime && frame < framesTotal - 1) 
        {
            const auto wait = 1.0f / double(fps) - trackTime;
            if (wait > 0.0) 
            {
                std::this_thread::sleep_for(std::chrono::microseconds(
                    static_cast<unsigned int>(wait * 1e6)));
            }
        }

        // Check if the SLAM process termination is requested.
        if (m_System->terminate_is_requested()) 
        {
            break;
        }
    }
        
    // Wait until the loop BA is finished.
    while (m_System->loop_BA_is_running()) 
    {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // Automatically close the viewer.
    if (m_Configuration.Terminate) 
    {
#ifdef USE_PANGOLIN_VIEWER
        m_Viewer->request_terminate();
#endif
    }
}

void Application::SaveProfilingTimes()
{
    std::ofstream ofs(m_Configuration.ProfilingPath, std::ios::out);
    if (ofs.is_open()) 
    {
        auto trackSize = m_TrackTimes->size();
        auto processSize = m_ProcessTimes->size();
        assert(trackSize == processSize);

        ofs << "Index, Track-Time, Process-Time" << std::endl;
        for (unsigned int i = 0; i < trackSize; i++) 
        {
            ofs << i << ", " << (*m_TrackTimes)[i] << ", " 
                << (*m_ProcessTimes)[i] << std::endl;
        }
        ofs.close();
    }
}

void Application::SaveKeyPointData()
{
    std::ofstream ofs(m_Configuration.KeyPointDataPath, std::ios::out);
    if (ofs.is_open()) 
    {
        auto keyPointSize = m_KeyPoints->size();
        auto matchSize = m_Matches->size();
        assert(keyPointSize == matchSize);

        /*
        ofs << "Process-Time" << std::endl;
        for (const auto processTime : *m_ProcessTimes) 
        {
            ofs << processTime << std::endl;
        }
        */

        ofs.close();
    }
}
