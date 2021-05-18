#include <iostream>
//#include <algorithm>
//#include <iomanip>
//#include <numeric>

#include <glog/logging.h>
#include <popl.hpp>
#include <spdlog/spdlog.h>

#include "Application.hpp"
#include "ApplicationConfiguration.hpp"
#include "ImageProcessing.hpp"

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

	auto optionMapPath = options.add<popl::Value<std::string>>(
        "", "map", "Store the map database at this path.", "");
	auto optionFrameTrajectoryPath =  options.add<popl::Value<std::string>>(
        "", "frame", "Store the frame trajectory at this path.", "");
	auto optionKeyframeTrajectoryPath =  options.add<popl::Value<std::string>>(
        "", "keyframe", "Store the keyframe trajectory at this path.", "");
    auto optionTrackTimesPath = options.add<popl::Value<std::string>>(
        "", "track-times", "Store the track times at this path.", "");

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

    auto optionBias = options.add<popl::Value<double>>(
        "", "bias", "Constant time bias.", 0.0);

	auto optionPreprocessing = options.add<popl::Value<int>>(
        "", "preprocessing", "Image preprocessing method.", 0);

    auto optionCLAHEClip = options.add<popl::Value<double>>(
        "", "clahe-clip", "CLAHE clip value.", 8.0);
    auto optionCLAHESize= options.add<popl::Value<unsigned int>>(
        "", "clahe-size", "CLAHE clip value.", 40);
   
    // Parse arguments.
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

	// Check if options are valid.
	if (optionHelp->is_set()) 
	{
		std::cerr << options << std::endl;
		return EXIT_FAILURE;
	}

	if (!optionVocabularyPath->is_set() || !optionSVOPath->is_set() 
        || !optionConfigurationPath->is_set())
	{
		std::cerr << "Invalid arguments" << std::endl;
		std::cerr << std::endl;
		std::cerr << options << std::endl;
		return EXIT_FAILURE;
	}

	// Set up logger.
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
	if (optionDebug->is_set()) 
	{
		spdlog::set_level(spdlog::level::debug);
	}
	else 
	{
		spdlog::set_level(spdlog::level::info);
	}

    // Set up application configuration.
    ApplicationConfiguration configuration;

    // Input paths.
    configuration.VocabularyPath = optionVocabularyPath->value();
    configuration.SVOPath = optionSVOPath->value();
    configuration.ConfigurationPath = optionConfigurationPath->value();

    // Output paths.
    configuration.MapPath=optionMapPath->value();
    configuration.FrameTrajectoryPath = optionFrameTrajectoryPath->value();
    configuration.KeyframeTrajectoryPath =
        optionKeyframeTrajectoryPath->value();
    configuration.TrackTimesPath = optionTrackTimesPath->value();

    // Settings.
    configuration.FrameSkip = optionFrameSkip->value();
    configuration.FrameStart = optionFrameStart->value();
    configuration.Realtime = optionRealtime->value();
    configuration.Terminate = optionAutoTerminate->value();
    configuration.Debug = optionDebug->value();
    configuration.Evaluate = optionEvaluate->value();

    // Generic values.
    configuration.Bias = optionBias->value();

    // Preprocessing values.
    configuration.Preprocessing = IntToImagePreprocessingType(
        optionPreprocessing->value());
    configuration.CLAHEClip = optionCLAHEClip->value();
    configuration.CLAHESize = optionCLAHESize->value();

    spdlog::set_level(spdlog::level::debug);

    // OpenVSLAM configuration.
    configuration.OpenVSLAMConfiguration = std::make_shared<openvslam::config>(
        optionConfigurationPath->value());

    std::cout << configuration << std::endl;

	// Create application and run.
    Application app(configuration);
    app.Run();

	return EXIT_SUCCESS;
}
