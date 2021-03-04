#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>
#include <popl.hpp>

#include <sl/Camera.hpp>

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"
#include "openvslam/util/image_converter.h"

cv::Mat slMat2cvMat(const sl::Mat& input) 
{
	int cv_type = -1;
	switch (input.getDataType()) 
	{
		case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
		case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
		case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
		case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
		case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
		case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
		case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
		case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
		default: break;
	}

	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, 
		input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

void rectifyImages(
	const openvslam::util::stereo_rectifier rectifier(cfg);
	const std::shared_ptr<openvslam::config>& cfg,
	const std::string& vocabFilePath, 
	const std::string& svoFile,
	const unsigned int frameSkip, 
	const bool noSleep, 
	const bool autoTerm,
	const bool evalLog, 
	const std::string& mapDatabasePath, 
	const bool histEqualization)
{
	// Set up rectifier.
	const openvslam::util::stereo_rectifier rectifier(cfg);

	rectifier.rectify(leftImage, rightImage, leftImageRect, 
		rightImageRect);
}

int main(int argc, char* argv[]) 
{
#ifdef USE_STACK_TRACE_LOGGER
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
#endif

	// Create options.
	popl::OptionParser op("Allowed options");
	auto help = op.add<popl::Switch>("h", "help", "produce help message");
	auto input = op.add<popl::Value<std::string>>("i", "input", 
		"Image input directory containing the data.");
	auto output = op.add<popl::Value<std::string>>("o", "output", 
		"Image input directory containing the data.");
	auto config_file_path = op.add<popl::Value<std::string>>("c", "config",
		"config file path");
	auto start = op.add<pop::Value<uint32_t>>("", "start",
		"SVO start index.");
	auto stop = op.add<pop::Value<uint32_t>>("", "stop",
		"SVO stop index.");
	auto histEqualization = op.add<popl::Switch>("", "equal-hist", 
		"apply histogram equalization");

	try 
	{
		op.parse(argc, argv);
	}
	catch (const std::exception& e) 
	{
		std::cerr << e.what() << std::endl;
		std::cerr << std::endl;
		std::cerr << op << std::endl;
		return EXIT_FAILURE;
	}

	// Check validness of options.
	if (help->is_set()) 
	{
		std::cerr << op << std::endl;
		return EXIT_FAILURE;
	}

	if (!svo_file_path->is_set() || !config_file_path->is_set())
	{
		std::cerr << "invalid arguments" << std::endl;
		std::cerr << std::endl;
		std::cerr << op << std::endl;
		return EXIT_FAILURE;
	}

	// Setup logger.
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
	if (debug_mode->is_set()) 
	{
		spdlog::set_level(spdlog::level::debug);
	}
	else 
	{
		spdlog::set_level(spdlog::level::info);
	}

	// Load configuration.
	std::shared_ptr<openvslam::config> cfg;
	try 
	{
        	cfg = std::make_shared<openvslam::config>(
			config_file_path->value());
	}
	catch (const std::exception& e) 
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

#ifdef USE_GOOGLE_PERFTOOLS
	ProfilerStart("slam.prof");
#endif

	// Rectify images.
	if (cfg->camera_->setup_type_ 
		== openvslam::camera::setup_type_t::Stereo)
	{
		rectifyImages(cfg, input->value(), output->value(),
			start->value(), stop->value(), 
			histEqualization->is_set());
	}
	else 
	{
		throw std::runtime_error("Invalid setup type: " 
			+ cfg->camera_->get_setup_type_string());
	}

#ifdef USE_GOOGLE_PERFTOOLS
	ProfilerStop();
#endif

	return EXIT_SUCCESS;
}
