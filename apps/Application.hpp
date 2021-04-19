#pragma once

#include <opencv2/core/core.hpp>

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"
#include "openvslam/util/image_converter.h"

#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif

#include "Configuration.hpp"
#include "Utilities.hpp"

void RunSLAM(const std::shared_ptr<openvslam::config>& openvslamConfiguration, 
    const ZEDSlamConfiguration& configuration)
{
	// InitParameters - Open SVO file.
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.input.setFromSVOFile(configuration.SVOPath.c_str());
	initParameters.svo_real_time_mode = configuration.Realtime;

	auto zedState= zed.open(initParameters);
	if (zedState != sl::ERROR_CODE::SUCCESS)
	{
		std::cerr << toString(zedState) << ": " << toVerbose(zedState) 
            << std::endl;
		return;
	}

	// Get SVO information.
	unsigned int framesTotal = zed.getSVONumberOfFrames();
	int fps = zed.getInitParameters().camera_fps;
	double timestep = 1.0f / double(fps);
	if (configuration.FrameStart < framesTotal)
	{
		zed.setSVOPosition(configuration.FrameStart);
	}
	else
	{
		std::cerr << "Invalid start frame." << std::endl;
		return;
	}
	
	// Set up rectifier.
	const openvslam::util::stereo_rectifier rectifier(openvslamConfiguration);

	// Build and start SLAM process.
	openvslam::system SLAM(openvslamConfiguration, configuration.VocabularyPath);
	SLAM.startup();

	// Create viewer and pass frame- and map publisher.
#ifdef USE_PANGOLIN_VIEWER
	pangolin_viewer::viewer viewer(openvslamConfiguration, &SLAM, 
        SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

	std::vector<double> track_times;
	track_times.reserve(framesTotal);

	sl::Timestamp previousTimestamp;
	sl::Timestamp currentTimestamp;
	sl::Mat rawLeftImage, rawRightImage;
	cv::Mat leftImage, rightImage;
    cv::Mat processedLeftImage, processedRightImage;
	cv::Mat rectifiedLeftImage, rectifiedRightImage;

    // Run the SLAM in another thread
	std::thread thread([&]() 
	{
		unsigned int frame = configuration.FrameStart;
		while (frame < framesTotal - 1) 
		{
			zedState = zed.grab();
			if (zedState != sl::ERROR_CODE::SUCCESS)
			{
				continue;
			}
			
			frame++;

            // Retrieve grayscale images.
		    zed.retrieveImage(rawLeftImage, sl::VIEW::LEFT_UNRECTIFIED_GRAY, 
                sl::MEM::CPU);
			zed.retrieveImage(rawRightImage, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, 
                sl::MEM::CPU);

            // Convert to OpenCV.
		    leftImage = StereolabsToOpenCV(rawLeftImage);
		    rightImage = StereolabsToOpenCV(rawRightImage);
    
            // Extract images and process.
			if (configuration.Preprocessing 
                == ImagePreprocessingType::HistogramEqualization) 
			{
                // Equalize image histograms.
				openvslam::util::histogram_equalization(leftImage, 
                    processedLeftImage);
				openvslam::util::histogram_equalization(rightImage,
                    processedRightImage);
			}
            else if (configuration.Preprocessing 
                == ImagePreprocessingType::Clahe)
            {
                // TODO: Add CLAHE.
                openvslam::util::clahe(leftImage, processedLeftImage, 2, 
                    { 8, 8 });
                openvslam::util::clahe(rightImage, processedRightImage, 2,
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

			rectifier.rectify(processedLeftImage, processedRightImage, 
                rectifiedLeftImage, rectifiedRightImage);
			
			const auto tp_1 = std::chrono::steady_clock::now();

            // Get timestamps.
            previousTimestamp = currentTimestamp.getNanoseconds();
			currentTimestamp = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);

			if (frame % configuration.FrameSkip == 0) 
			{
				if (previousTimestamp.getMilliseconds() != 0)
				{
					timestep = double(currentTimestamp.getMilliseconds() 
						- previousTimestamp.getMilliseconds()) 
						/ double(1e3);
				}

				SLAM.feed_stereo_frame(rectifiedLeftImage, rectifiedRightImage, 
                    timestep);
			}

			const auto tp_2 = std::chrono::steady_clock::now();

			const auto track_time = std::chrono::duration_cast
				<std::chrono::duration<double>>(tp_2 - tp_1)
				.count();
			if (frame % configuration.FrameSkip == 0) 
			{
				track_times.push_back(track_time);
			}

			// Wait until the timestamp of the next frame.
			if (!configuration.Realtime && frame < framesTotal - 1) 
			{
				const auto wait = 1.0f / double(fps) - track_time;
				if (wait > 0.0) 
				{
					std::this_thread::sleep_for(std::chrono::microseconds(
						static_cast<unsigned int>(wait * 1e6)));
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
	if (configuration.Terminate) 
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
	if (configuration.Evaluate) 
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
	if (!configuration.MapPath.empty()) 
	{
		SLAM.save_map_database(configuration.MapPath);
	}

	std::sort(track_times.begin(), track_times.end());
	const auto total_track_time = 
		std::accumulate(track_times.begin(), track_times.end(), 0.0);
	std::cout << "median tracking time: " 
		<< track_times.at(track_times.size() / 2) << "[s]" << std::endl;
	std::cout << "mean tracking time: " 
		<< total_track_time / track_times.size() << "[s]" << std::endl;
}
