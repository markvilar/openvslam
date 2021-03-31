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

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include <sl/Camera.hpp>

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"
#include "openvslam/util/image_converter.h"

#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

cv::Mat slMat2cvMat(const sl::Mat& input) 
{
	int cv_type = -1;
	switch (input.getDataType()) 
	{
		case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
		case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
		case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
		case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
		case sl::MAT_TYPE::U8_C1:  cv_type = CV_8UC1;  break;
		case sl::MAT_TYPE::U8_C2:  cv_type = CV_8UC2;  break;
		case sl::MAT_TYPE::U8_C3:  cv_type = CV_8UC3;  break;
		case sl::MAT_TYPE::U8_C4:  cv_type = CV_8UC4;  break;
		default: break;
	}

	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, 
		input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

void stereo_tracking(
	const std::shared_ptr<openvslam::config>& cfg,
	const std::string& vocabulary_file_path, 
	const std::string& svo_file_path,
	const unsigned int frame_skip, 
	const unsigned int frame_start, 
	const bool no_sleep, 
	const bool auto_terminate,
	const bool eval_log, 
	const std::string& map_database_path, 
	const bool histogram_equalization)
{
	// InitParameters - Open SVO file.
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.input.setFromSVOFile(svo_file_path.c_str());
	initParameters.svo_real_time_mode = false;

	// InitParameters - Image processing / calibration.
	initParameters.camera_disable_self_calib = true;
	initParameters.enable_image_enhancement = true;

	auto zed_state = zed.open(initParameters);
	if (zed_state != sl::ERROR_CODE::SUCCESS)
	{
		std::cerr << toString(zed_state) << ": " << toVerbose(zed_state) 
			<< std::endl;
		return;
	}

	// Get SVO information.
	unsigned int frames_total = zed.getSVONumberOfFrames();
	int fps = zed.getInitParameters().camera_fps;
	double dt = 1.0f / double(fps);
	if (frame_start < frames_total)
	{
		zed.setSVOPosition(frame_start);
	}
	else
	{
		std::cerr << "Invalid start frame." << std::endl;
		return;
	}
	
	// Set up rectifier.
	const openvslam::util::stereo_rectifier rectifier(cfg);

	// Build and start SLAM process.
	openvslam::system SLAM(cfg, vocabulary_file_path);
	SLAM.startup();

	// Create viewer and pass frame- and map publisher.
#ifdef USE_PANGOLIN_VIEWER
	pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), 
		SLAM.get_map_publisher());
#endif

	std::vector<double> track_times;
	track_times.reserve(frames_total);

	sl::Timestamp timestamp_prev;
	sl::Timestamp timestamp_curr;
	sl::Mat leftImageSl, rightImageSl;
	cv::Mat leftImage, rightImage;
	cv::Mat leftImageRect, rightImageRect;

    	// Run the SLAM in another thread
	std::thread thread([&]() 
	{
		unsigned int frame = frame_start;
		while (frame < frames_total - 1) 
		{
			zed_state = zed.grab();
			if (zed_state != sl::ERROR_CODE::SUCCESS)
			{
				continue;
			}
			
			frame++;
			// Get ZED data.
			zed.retrieveImage(leftImageSl, 
				sl::VIEW::LEFT_UNRECTIFIED_GRAY, sl::MEM::CPU);
			zed.retrieveImage(rightImageSl, 
				sl::VIEW::RIGHT_UNRECTIFIED_GRAY, sl::MEM::CPU);
			timestamp_prev = timestamp_curr.getNanoseconds();
			timestamp_curr = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
				
			// Convert to OpenCV.
			leftImage = slMat2cvMat(leftImageSl);
			rightImage = slMat2cvMat(rightImageSl);
			if (histogram_equalization) 
			{
				openvslam::util::equalize_histogram(leftImage);
				openvslam::util::equalize_histogram(rightImage);
			}

			if (leftImage.empty() || rightImage.empty()) 
			{
				continue;
			}

			rectifier.rectify(leftImage, rightImage, leftImageRect, 
				rightImageRect);
			
			const auto tp_1 = std::chrono::steady_clock::now();

			if (frame % frame_skip == 0) 
			{
				if (timestamp_prev.getMicroseconds() != 0)
				{
					dt = double(timestamp_curr.getMicroseconds() 
						- timestamp_prev.getMicroseconds()) 
						/ double(1e6);
				}

				SLAM.feed_stereo_frame(leftImageRect, rightImageRect, dt);
			}

			const auto tp_2 = std::chrono::steady_clock::now();

			const auto track_time = std::chrono::duration_cast
				<std::chrono::duration<double>>(tp_2 - tp_1)
				.count();
			if (frame % frame_skip == 0) 
			{
				track_times.push_back(track_time);
			}

			// Wait until the timestamp of the next frame.
			if (!no_sleep && frame < frames_total - 1) 
			{
				const auto wait_time = 1.0f / double(fps) - track_time;
				if (wait_time > 0.0) 
				{
					std::this_thread::sleep_for(std::chrono::microseconds(
						static_cast<unsigned int>(wait_time * 1e6)));
				}
			}

			// Check if the SLAM process termination is requested.
			if (SLAM.terminate_is_requested()) 
			{
				break;
			}
		}
        
	// Wait until the loop BA is finished.
	while (SLAM.loop_BA_is_running()) 
	{
		std::this_thread::sleep_for(std::chrono::microseconds(5000));
	}

	// Automatically close the viewer.
#ifdef USE_PANGOLIN_VIEWER
	if (auto_terminate) 
	{
		viewer.request_terminate();
	}
#endif
	});

	// Run the viewer in the current thread.
#ifdef USE_PANGOLIN_VIEWER
	viewer.run();
#endif

	thread.join();

	// Shutdown SLAM process.
	SLAM.shutdown();

	// Output information for evaluation.
	if (eval_log) 
	{
		SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
		SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
		std::ofstream ofs("track_times.txt", std::ios::out);
		if (ofs.is_open()) 
		{
			for (const auto track_time : track_times) 
			{
				ofs << track_time << std::endl;
			}
			ofs.close();
		}
	}

	// Output map information.
	if (!map_database_path.empty()) 
	{
		SLAM.save_map_database(map_database_path);
	}

	std::sort(track_times.begin(), track_times.end());
	const auto total_track_time = 
		std::accumulate(track_times.begin(), track_times.end(), 0.0);
	std::cout << "median tracking time: " 
		<< track_times.at(track_times.size() / 2) << "[s]" << std::endl;
	std::cout << "mean tracking time: " 
		<< total_track_time / track_times.size() << "[s]" << std::endl;
}

int main(int argc, char* argv[]) 
{
#ifdef USE_STACK_TRACE_LOGGER
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
#endif

	// Create options.
	popl::OptionParser op("Allowed options.");
	auto help = op.add<popl::Switch>("h", "help", "Help message.");
	auto vocabulary_file_path = op.add<popl::Value<std::string>>("v",
		"vocabulary", "ORB vocabulary file path.");
	auto svo_file_path = op.add<popl::Value<std::string>>("s", "svo", 
		"SVO file containing the data.");
	auto config_file_path = op.add<popl::Value<std::string>>("c", 
		"configuration", "Configuration file path.");
	auto frame_skip= op.add<popl::Value<unsigned int>>("", "frame-skip",
		"Frame skip interval.", 1);
	auto frame_start = op.add<popl::Value<unsigned int>>("", "frame-start",
		"Start frame.", 0);
	auto no_sleep = op.add<popl::Switch>("", "no-sleep",
		"Not waiting for next frame.");
	auto auto_terminate = op.add<popl::Switch>("", "auto-term", 
		"Automatically terminate the viewer.");
	auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
	auto eval_log = op.add<popl::Switch>("", "eval-log", 
		"Store trajectory and tracking times for evaluation.");
	auto map_database_path = op.add<popl::Value<std::string>>("p", "map-db", 
		"Store a map database at this path after SLAM.", "");
	auto histogram_equalization = op.add<popl::Switch>("", "equal-hist", 
		"Apply histogram equalization.");

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

	if (!vocabulary_file_path->is_set() || !svo_file_path->is_set() || 
		!config_file_path->is_set())
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

	// Run tracking.
	if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Stereo)
	{
		stereo_tracking(cfg, vocabulary_file_path->value(), 
			svo_file_path->value(), 
			frame_skip->value(), 
			frame_start->value(),
			no_sleep->is_set(), 
			auto_terminate->is_set(), 
			eval_log->is_set(), 
			map_database_path->value(), 
			histogram_equalization->is_set());
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
