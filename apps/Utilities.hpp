#pragma once

#include <opencv2/core/core.hpp>

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include <sl/Camera.hpp>

cv::Mat StereolabsToOpenCV(const sl::Mat& input);
