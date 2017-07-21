#pragma once

#ifdef _MSC_VER
//To avoid compiler error C2589
#define NOMINMAX
#include <windows.h>
#endif

//C++ includes
#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <atomic>
#include <mutex>
#include <numeric>
#include <random>
// C includes
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
//#include <omp.h>

#include<opencv2/core/version.hpp>

//OPENCV Includes 
#if CV_MAJOR_VERSION == 2
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
//#include <opencv2/gpu/device/common.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/internal.hpp>
//#include <opencv2/gpu/stream_accessor.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <opencv2/core/gpumat.hpp>
#elif CV_MAJOR_VERSION == 3
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#else
#error "Undefined OpenCV Version"
#endif
//ZED includes
//#include <zed/Mat.hpp>
//#include <zed/Camera.hpp>
//#include <zed/utils/GlobalDefine.hpp>