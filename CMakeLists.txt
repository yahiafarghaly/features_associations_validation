project (validation_test)
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_definitions(-std=c++11)

add_executable (validation main.cpp)
TARGET_LINK_LIBRARIES( validation
			${OpenCV_LIBRARIES}
			)
