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
        // TODO: Bug here!
        std::tie(leftImage, rightImage) = ProcessImagePairRaw(camera);
        break;
    case ImagePreprocessingType::HistogramEqualization:
        std::tie(leftImage, rightImage) = ProcessImagePairHE(camera);
        break;
    case ImagePreprocessingType::CLAHE:
        std::cout << "CLAHE" << std::endl;
        std::tie(leftImage, rightImage) = ProcessImagePairCLAHE(camera);
        break;
    case ImagePreprocessingType::UIENet:
        std::tie(leftImage, rightImage) = ProcessImagePairUIENet(camera);
        break;
    default:
        std::tie(leftImage, rightImage) = ProcessImagePairRaw(camera);
        break;
    }

    if (m_Configuration.Filter)
    {
        // Noise reduction filter.
        FilterImageBilateral(leftImage, leftImageFiltered);
        FilterImageBilateral(rightImage, rightImageFiltered);
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

std::pair<cv::Mat, cv::Mat> Application::ProcessImagePairRaw(
    const std::shared_ptr<sl::Camera>& camera)
{
	sl::Mat leftImage, rightImage;
	cv::Mat leftImageGray, rightImageGray;

    camera->retrieveImage(leftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);
    camera->retrieveImage(rightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);

    leftImageGray = StereolabsToCVMat(leftImage);
    rightImageGray = StereolabsToCVMat(rightImage);

    return { leftImageGray.clone(), rightImageGray.clone() };
}

std::pair<cv::Mat, cv::Mat> Application::ProcessImagePairHE(
    const std::shared_ptr<sl::Camera>& camera)
{
   	sl::Mat leftImage, rightImage;
	cv::Mat leftImageGray, rightImageGray;
	cv::Mat leftImageEqualized, rightImageEqualized;

    camera->retrieveImage(leftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);
    camera->retrieveImage(rightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);

    leftImageGray = StereolabsToCVMat(leftImage);
    rightImageGray = StereolabsToCVMat(rightImage);

    openvslam::util::histogram_equalization(leftImageGray, 
        leftImageEqualized);
    openvslam::util::histogram_equalization(rightImageGray, 
        rightImageEqualized);

    return { leftImageEqualized, rightImageEqualized };
}

std::pair<cv::Mat, cv::Mat> Application::ProcessImagePairCLAHE(
    const std::shared_ptr<sl::Camera>& camera)
{
   	sl::Mat leftImage, rightImage;
	cv::Mat leftImageGray, rightImageGray;
	cv::Mat leftImageEqualized, rightImageEqualized;

    camera->retrieveImage(leftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);
    camera->retrieveImage(rightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
        sl::MEM::CPU);

    leftImageGray = StereolabsToCVMat(leftImage);
    rightImageGray = StereolabsToCVMat(rightImage);

    openvslam::util::clahe(leftImageGray, leftImageEqualized, 
        m_Configuration.CLAHEClip, 
        { m_Configuration.CLAHESize, m_Configuration.CLAHESize });

    openvslam::util::clahe(rightImageGray, rightImageEqualized, 
        m_Configuration.CLAHEClip, 
        { m_Configuration.CLAHESize, m_Configuration.CLAHESize });

    return { leftImageEqualized, rightImageEqualized };
}

std::pair<cv::Mat, cv::Mat> Application::ProcessImagePairUIENet(
    const std::shared_ptr<sl::Camera>& camera)
{
   	sl::Mat leftImage, rightImage;
    cv::cuda::GpuMat leftImageGPU, rightImageGPU;

    camera->retrieveImage(leftImage, sl::VIEW::LEFT_UNRECTIFIED, 
        sl::MEM::GPU);
    camera->retrieveImage(rightImage, sl::VIEW::RIGHT_UNRECTIFIED, 
        sl::MEM::GPU);

    // Convert from uint8 to float32.
    /*
    cv::Mat imageFloat;
    src.convertTo(imageFloat, CV_32FC3, 1.0f / 255.0f);

    // Upload image tensor to GPU.
    cv::cuda::GpuMat imageGPU;
    imageGPU.upload(imageFloat);
    */

    // TODO: Input tensor to model.

    // TODO: Download output tensor fram model.

    // TODO: Convert to grayscale.

    return ProcessImagePairRaw(camera);
}

void Application::FilterImageBilateral(const cv::Mat& src, cv::Mat& dst)
{
    cv::bilateralFilter(src, dst, m_Configuration.FilterDiameter, 
        m_Configuration.FilterColor, m_Configuration.FilterSpace);
}

void Application::Run()
{
	// Build and start SLAM process.
	m_System->startup();

	// Create viewer.
	m_Viewer = std::make_shared<pangolin_viewer::viewer>(
        openvslam::util::yaml_optional_ref(
            m_Configuration.OpenVSLAMConfiguration->yaml_node_, 
            "PangolinViewer"), 
        &*m_System, m_System->get_frame_publisher(), 
        m_System->get_map_publisher());

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
            openvslam::io::trajectory_format::csv);
		m_System->save_keyframe_trajectory(
            m_Configuration.KeyframeTrajectoryPath, 
            openvslam::io::trajectory_format::csv);

        SaveTrackTimes();
        SaveProcessTimes();
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
    m_TrackTimes = std::make_shared<std::vector<float>>(framesTotal, 0.0);
    m_ProcessTimes = std::make_shared<std::vector<float>>(framesTotal, 0.0);

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
            m_ProcessTimes->insert(m_ProcessTimes->begin() + frame, processTime);
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

void Application::SaveTrackTimes()
{
    std::ofstream ofs(m_Configuration.TrackTimesPath, std::ios::out);
    if (ofs.is_open()) 
    {
        ofs << "Track-Time" << std::endl;
        for (const auto trackTime : *m_TrackTimes) 
        {
            ofs << trackTime << std::endl;
        }
        ofs.close();
    }
}

void Application::SaveProcessTimes()
{
    std::ofstream ofs(m_Configuration.ProcessTimesPath, std::ios::out);
    if (ofs.is_open()) 
    {
        ofs << "Process-Time" << std::endl;
        for (const auto processTime : *m_ProcessTimes) 
        {
            ofs << processTime << std::endl;
        }
        ofs.close();
    }
}
