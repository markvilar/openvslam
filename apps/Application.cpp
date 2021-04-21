#include "Application.hpp"

Application* Application::s_Instance = nullptr;

Application::Application(const ApplicationConfiguration& configuration)
    : m_Configuration(configuration), 
    m_Rectifier(m_Configuration.OpenVSLAMConfiguration)
{
    s_Instance = this;
	m_System = std::make_shared<openvslam::system>(
        m_Configuration.OpenVSLAMConfiguration, m_Configuration.VocabularyPath);
}

Application::~Application()
{
}

void Application::Run()
{
	// Build and start SLAM process.
	m_System->startup();

	// Create viewer.
	m_Viewer = std::make_shared<pangolin_viewer::viewer>(
        m_Configuration.OpenVSLAMConfiguration, &*m_System, 
        m_System->get_frame_publisher(), m_System->get_map_publisher());

    // Start tracking thread.
    m_TrackingThread = std::make_unique<std::thread>(&Application::TrackFrames,
        this);

	// Run the viewer in the current thread.
	m_Viewer->run();

    // Join tracking thread.
	m_TrackingThread->join();

	// Shutdown SLAM process.
	m_System->shutdown();

	// Output information for evaluation.
	if (m_Configuration.Evaluate) 
	{
		m_System->save_frame_trajectory(m_Configuration.FrameTrajectoryPath, 
            "TUM");
		m_System->save_keyframe_trajectory(
            m_Configuration.KeyframeTrajectoryPath, "TUM");

		std::ofstream ofs(m_Configuration.TrackTimesPath, std::ios::out);
		if (ofs.is_open()) 
		{
			for (const auto trackTime : *m_TrackTimes) 
			{
				ofs << trackTime << std::endl;
			}
			ofs.close();
		}
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
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.input.setFromSVOFile(m_Configuration.SVOPath.c_str());
	initParameters.svo_real_time_mode = m_Configuration.Realtime;

	auto zedState= zed.open(initParameters);
	if (zedState != sl::ERROR_CODE::SUCCESS)
	{
		std::cerr << toString(zedState) << ": " << toVerbose(zedState) 
            << std::endl;
		return;
	}

    // Frame information.
	unsigned int framesTotal = zed.getSVONumberOfFrames();
	unsigned int frame = m_Configuration.FrameStart;
    m_TrackTimes = std::make_shared<std::vector<float>>(framesTotal, 0.0);

    // TODO: Invalid FPS.
	int fps = zed.getInitParameters().camera_fps;
	double timestep = 1.0f / double(fps);

	if (m_Configuration.FrameStart < framesTotal)
	{
		zed.setSVOPosition(m_Configuration.FrameStart);
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

	sl::Timestamp previousTimestamp, currentTimestamp;
	sl::Mat rawLeftImage, rawRightImage;
	cv::Mat leftImage, rightImage;
    cv::Mat processedLeftImage, processedRightImage;
	cv::Mat rectifiedLeftImage, rectifiedRightImage;

	while (frame < framesTotal - 1) 
	{
        zedState = zed.grab();
        if (zedState != sl::ERROR_CODE::SUCCESS)
        {
            continue;
        }
        
        frame += m_Configuration.FrameSkip;

        // Retrieve grayscale images.
        zed.retrieveImage(rawLeftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
            sl::MEM::CPU);
        zed.retrieveImage(rawRightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
            sl::MEM::CPU);

        // Convert to OpenCV.
        leftImage = StereolabsToOpenCV(rawLeftImage);
        rightImage = StereolabsToOpenCV(rawRightImage);

        // Extract images and process.
        if (m_Configuration.Preprocessing 
            == ImagePreprocessingType::HistogramEqualization) 
        {
            // Equalize image histograms.
            openvslam::util::histogram_equalization(leftImage, 
                processedLeftImage);
            openvslam::util::histogram_equalization(rightImage,
                processedRightImage);
        }
        else if (m_Configuration.Preprocessing == ImagePreprocessingType::CLAHE)
        {
            openvslam::util::clahe(leftImage, processedLeftImage, 1, { 8, 8 });
            openvslam::util::clahe(rightImage, processedRightImage, 1, 
                { 8, 8 });
        }
        else
        {
            processedLeftImage = leftImage;
            processedRightImage = rightImage;
        }

        if (processedLeftImage.empty() || processedRightImage.empty()) 
        {
            continue;
        }

        m_Rectifier.rectify(processedLeftImage, processedRightImage, 
            rectifiedLeftImage, rectifiedRightImage);

        const auto tp_1 = std::chrono::steady_clock::now();

        // Get timestamps.
        previousTimestamp = currentTimestamp.getNanoseconds();
        currentTimestamp = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);

        if (frame % m_Configuration.FrameSkip == 0) 
        {
            if (previousTimestamp.getMilliseconds() != 0)
            {
                timestep = double(currentTimestamp.getMilliseconds() 
                    - previousTimestamp.getMilliseconds()) 
                    / double(1000);
            }

            m_System->feed_stereo_frame(rectifiedLeftImage, 
                rectifiedRightImage, timestep);
        }

        const auto tp_2 = std::chrono::steady_clock::now();

        const auto trackTime = 
            std::chrono::duration_cast<std::chrono::duration<double>>(
            tp_2 - tp_1).count();
        if (frame % m_Configuration.FrameSkip == 0) 
        {
            m_TrackTimes->insert(m_TrackTimes->begin() + frame, trackTime);
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
        m_Viewer->request_terminate();
    }
}
