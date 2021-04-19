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

#include "openvslam/config.h"

#include "Application.hpp"
#include "Configuration.hpp"

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

int main(int argc, char* argv[]) 
{
	// Set up argument parser.
	popl::OptionParser options("Allowed options.");
	auto optionHelp = options.add<popl::Switch>(
        "h", "help", "Help message.");
	auto optionVocabularyPath = options.add<popl::Value<std::string>>(
        "v", "vocabulary", "ORB vocabulary file path.");
	auto optionSVOPath = options.add<popl::Value<std::string>>(
        "s", "svo", "SVO file containing the data.");
	auto optionConfigurationPath = options.add<popl::Value<std::string>>(
        "c", "configuration", "Configuration file path.");
	auto optionFrameSkip = options.add<popl::Value<unsigned int>>(
        "", "frame-skip", "Frame skip interval.", 1);
	auto optionFrameStart = options.add<popl::Value<unsigned int>>(
        "", "frame-start", "Start frame.", 0);
	auto optionRealtime = options.add<popl::Switch>(
        "", "realtime", "Not waiting for next frame.");
	auto optionAutoTerminate = options.add<popl::Switch>(
        "", "auto-term", "Automatically terminate the viewer.");
	auto optionDebug = options.add<popl::Switch>(
        "", "debug", "debug mode");
	auto optionEvaluate = options.add<popl::Switch>(
        "", "eval-log", "Store trajectory and tracking times.");
	auto optionMapPath = options.add<popl::Value<std::string>>(
        "p", "map-db", "Store a map database at this path after SLAM.", "");
	auto optionPreprocessing = options.add<popl::Value<int>>(
        "", "preprocessing", "Image preprocessing method.");
	
    try 
	{
		options.parse(argc, argv);
	}
	catch (const std::exception& e) 
	{
		std::cerr << e.what() << std::endl;
		std::cerr << std::endl;
		std::cerr << options << std::endl;
		return EXIT_FAILURE;
	}

	// Check if options.ions are valid.
	if (optionHelp->is_set()) 
	{
		std::cerr << options << std::endl;
		return EXIT_FAILURE;
	}

	if (!optionVocabularyPath->is_set() || !optionSVOPath->is_set() 
        || !optionConfigurationPath->is_set())
	{
		std::cerr << "invalid arguments" << std::endl;
		std::cerr << std::endl;
		std::cerr << options << std::endl;
		return EXIT_FAILURE;
	}

	// Setup logger.
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
	if (optionDebug->is_set()) 
	{
		spdlog::set_level(spdlog::level::debug);
	}
	else 
	{
		spdlog::set_level(spdlog::level::info);
	}

	// TODO: Fix loading. Load configuration.
	std::shared_ptr<openvslam::config> openvslamConfiguration;
	try 
	{
        openvslamConfiguration = std::make_shared<openvslam::config>(
            optionConfigurationPath->value());
	}
	catch (const std::exception& e) 
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

    // Parse preprocessing options.ion.

    /*
	// Run tracking.
	if (cfg->camera_->setup_type_ == options.nvslam::camera::setup_type_t::Stereo)
	{
		RunSLAM(openvslamConfiguration, configuration);
            svo_file_path->value(), 
			configuration.FrameSkip->value(), 
			configuration.FrameStart->value(),
			no_sleep->is_set(), 
			auto_terminate->is_set(), 
			eval_log->is_set(), 
			map_database_path->value(), 
            // TODO: Add preprocessing.
            );
	}
	else 
	{
		throw std::runtime_error("Invalid setup type: " 
            + cfg->camera_->get_setup_type_string());
	}
    */

	return EXIT_SUCCESS;
}
