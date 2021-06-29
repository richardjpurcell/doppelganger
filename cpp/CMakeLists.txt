cmake_minimum_required(VERSION 2.8.12)

PROJECT(doppelganger)
SET(OpenCV_DIR /home/richard/installation/OpenCV/opencv-4.5.0/lib/cmake/opencv4)

###################   ***DO NOT EDIT***  #####################

#############  Common Instructions for all Users  ############
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})
include(../dlib/dlib/cmake)

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib::dlib)
ENDMACRO()

add_example(doppelgangerEnroll)
add_example(doppelgangerRecognize)

