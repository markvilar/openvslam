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
            "Default");
		m_System->save_keyframe_trajectory(
            m_Configuration.KeyframeTrajectoryPath, "Default");

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
	sl::Camera camera;
	sl::InitParameters initParameters;
	initParameters.input.setFromSVOFile(m_Configuration.SVOPath.c_str());
	initParameters.svo_real_time_mode = m_Configuration.Realtime;

	auto cameraState= camera.open(initParameters);
	if (cameraState != sl::ERROR_CODE::SUCCESS)
	{
		std::cerr << toString(cameraState) << ": " << toVerbose(cameraState) 
            << std::endl;
		return;
	}

    // Frame information.
	unsigned int framesTotal = camera.getSVONumberOfFrames();
	unsigned int frame = m_Configuration.FrameStart;
    double timestamp = 0.0;
    double fps = camera.getInitParameters().camera_fps;
    m_TrackTimes = std::make_shared<std::vector<float>>(framesTotal, 0.0);

	if (m_Configuration.FrameStart < framesTotal)
	{
		camera.setSVOPosition(m_Configuration.FrameStart);
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

    // Allocate images.
	sl::Mat nativeLeftImage, nativeRightImage;
	cv::Mat unprocessedLeftImage, unprocessedRightImage;
    cv::Mat processedLeftImage, processedRightImage;
	cv::Mat rectifiedLeftImage, rectifiedRightImage;

	while (frame < framesTotal - 1) 
	{
        cameraState = camera.grab();
        if (cameraState != sl::ERROR_CODE::SUCCESS)
        {
            continue;
        }
        
        frame++;

        // Retrieve grayscale images.
        camera.retrieveImage(nativeLeftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
            sl::MEM::CPU);
        camera.retrieveImage(nativeRightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
            sl::MEM::CPU);

        // Convert to OpenCV.
        unprocessedLeftImage = StereolabsToOpenCV(nativeLeftImage);
        unprocessedRightImage = StereolabsToOpenCV(nativeRightImage);

        // Extract images and process.
        if (m_Configuration.Preprocessing 
            == ImagePreprocessingType::HistogramEqualization) 
        {
            // Equalize image histograms.
            openvslam::util::histogram_equalization(unprocessedLeftImage, 
                processedLeftImage);
            openvslam::util::histogram_equalization(unprocessedRightImage,
                processedRightImage);
        }
        else if (m_Configuration.Preprocessing == ImagePreprocessingType::CLAHE)
        {
            openvslam::util::clahe(unprocessedLeftImage, processedLeftImage, 
                2.0, { 8, 8 });
            openvslam::util::clahe(unprocessedRightImage, processedRightImage, 
                2.0, { 8, 8 });
        }
        else
        {
            processedLeftImage = unprocessedLeftImage;
            processedRightImage = unprocessedRightImage;
        }

        if (processedLeftImage.empty() || processedRightImage.empty()) 
        {
            continue;
        }

        m_Rectifier.rectify(processedLeftImage, processedRightImage, 
            rectifiedLeftImage, rectifiedRightImage);

        const auto trackStart = std::chrono::steady_clock::now();

        if (frame % m_Configuration.FrameSkip == 0) 
        {
            timestamp = camera.getTimestamp(sl::TIME_REFERENCE::IMAGE)
                .getMilliseconds() / double(1000);

            m_System->feed_stereo_frame(rectifiedLeftImage, 
                rectifiedRightImage, timestamp);
        }

        const auto trackEnd= std::chrono::steady_clock::now();

        const auto trackTime = std::chrono::duration_cast
            <std::chrono::duration<double>>(trackEnd - trackStart).count();
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
