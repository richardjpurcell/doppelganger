/*
 * File:      doppelgangerRecognize.cpp
 * Author:    Richard Purcell
 * Date:      2021-06-28
 * Version:   1.0
 * Purpose:   Find nearest celebrity match for a user submitted image.
 * Usage:     $ ./doppelgangerRecognize <path to image>
 * Usage:     ie: $ ./doppelgangerRecognize ../images/sofia-solares.jpg
 * Notes:     Created for OpenCV's Computer Vision 2 Project 2.
 *            This file is heavily based on testDlibFaceRecImage.cpp,
 *            provided for Computer Vision 2, week 4.
 *            Threshold is currently hard coded at 0.6
 *            
 */

 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <math.h>
 #include <map>
 #include <iomanip> // setprecision
 #include <sstream> // stringstream
#include <string>
#include <string.h>
#include <vector>
#include <dirent.h>
#include <algorithm>

 #include <opencv2/core.hpp>
 #include <opencv2/videoio.hpp>
 #include <opencv2/highgui.hpp>
 #include <opencv2/imgproc.hpp>

 #include <dlib/string.h>
 #include <dlib/dnn.h>
 #include <dlib/image_io.h>
 #include <dlib/opencv.h>
 #include <dlib/image_processing.h>
 #include <dlib/image_processing/frontal_face_detector.h>

 #include "labelData.h"

// dirent.h is pre-included with *nix like systems
// but not for Windows. So we are trying to include
// this header files based on Operating System
#ifdef _WIN32
  #include "dirent.h"
#elif __APPLE__
  #include "TargetConditionals.h"
#if TARGET_OS_MAC
  #include <dirent.h>
#else
  #error "Not Mac. Find an alternative to dirent"
#endif
#elif __linux__
  #include <dirent.h>
#elif __unix__ // all unices not caught above
  #include <dirent.h>
#else
  #error "Unknown compiler"
#endif

using namespace cv;
using namespace dlib;
using namespace std;

#define THRESHOLD 0.6

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network. It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller. Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;
// ----------------------------------------------------------------------------------------

// Reads files in a directory
void listdir(string dirName, std::vector<string>& fileNames) 
{
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(dirName.c_str())) != NULL) 
  {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      // ignore . and ..
      if((strcmp(ent->d_name,".") == 0) || (strcmp(ent->d_name,"..") == 0)) {
      continue;
      }
      string temp_name = ent->d_name;
      // Read more about file types identified by dirent.h here
      // https://www.gnu.org/software/libc/manual/html_node/Directory-Entries.html
      switch (ent->d_type) {
        case DT_REG:
          fileNames.push_back(temp_name);
          break;
      }
    }
    // sort all the files
    sort(fileNames.begin(), fileNames.end());
    closedir(dir);
  }
}

// function to print a vector
template<typename T>
void printVector(std::vector<T>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    cout << i << " " << vec[i] << "; ";
  }
  cout << endl;
}

// read names and labels mapping from file
static void readLabelNameMap(const string& filename, std::vector<string>& names, std::vector<int>& labels,
                             std::map<int, string>& labelNameMap, char separator = ';') {
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }
  string line;
  string name, labelStr;
  // read lines from file one by one
  while (getline(file, line)) {
    stringstream liness(line);
    // read first word which is person name
    getline(liness, name, separator);
    // read second word which is integer label
    getline(liness, labelStr);
    if(!name.empty() && !labelStr.empty()) {
      names.push_back(name);
      // convert label from string format to integer
      int label = atoi(labelStr.c_str());
      labels.push_back(label);
      // add (integer label, person name) pair to map
      labelNameMap[label] = name;
    }
  }
}

// read descriptors saved on disk
static void readDescriptors(const string& filename, std::vector<int>& faceLabels, std::vector<matrix<float,0,1>>& faceDescriptors, char separator = ';') {
  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }
  // each line has:
  // 1st element = face label
  // rest 128 elements = descriptor elements
  string line;
  string faceLabel;
  // valueStr = one element of descriptor in string format
  // value = one element of descriptor in float
  string valueStr;
  float value;
  std::vector<float> faceDescriptorVec;
  // read lines from file one by one
  while (getline(file, line)) {
    stringstream liness(line);
    // read face label
    // read first word on a line till separator
    getline(liness, faceLabel, separator);
    if(!faceLabel.empty()) {
      faceLabels.push_back(std::atoi(faceLabel.c_str()));
    }

    faceDescriptorVec.clear();
    // read rest of the words one by one using separator
    while (getline(liness, valueStr, separator)) {
      if (!valueStr.empty()) {
        // convert descriptor element from string to float
        faceDescriptorVec.push_back(atof(valueStr.c_str()));
      }
    }

    // convert face descriptor from vector of float to Dlib's matrix format
    dlib::matrix<float, 0, 1> faceDescriptor = dlib::mat(faceDescriptorVec);
    faceDescriptors.push_back(faceDescriptor);
  }
}

// find nearest face descriptor from vector of enrolled faceDescriptor
// to a query face descriptor
void nearestNeighbor(dlib::matrix<float, 0, 1>& faceDescriptorQuery,
                    std::vector<dlib::matrix<float, 0, 1>>& faceDescriptors,
                    std::vector<int>& faceLabels, int& label, float& minDistance) {
  int minDistIndex = 0;
  minDistance = 1.0;
  label = -1;
  // Calculate Euclidean distances between face descriptor calculated on face dectected
  // in current frame with all the face descriptors we calculated while enrolling faces
  // Calculate minimum distance and index of this face
  for (int i = 0; i < faceDescriptors.size(); i++) {
    double distance = length(faceDescriptors[i] - faceDescriptorQuery);
    if (distance < minDistance) {
      minDistance = distance;
      minDistIndex = i;
    }
  }
  // Dlib specifies that in general, if two face descriptor vectors have a Euclidean
  // distance between them less than 0.6 then they are from the same
  // person, otherwise they are from different people.

  // This threshold will vary depending upon number of images enrolled
  // and various variations (illuminaton, camera quality) between
  // enrolled images and query image
  // We are using a threshold of 0.5
  // if minimum distance is greater than a threshold
  // assign integer label -1 i.e. unknown face
  if (minDistance > THRESHOLD){
    label = -1;
  } else {
    label = faceLabels[minDistIndex];
  }
}

int main(int argc, char *argv[]) {
  // Initialize face detector, facial landmarks detector and face recognizer
  String predictorPath, faceRecognitionModelPath;
  predictorPath = "../models/shape_predictor_68_face_landmarks.dat";
  faceRecognitionModelPath = "../models/dlib_face_recognition_resnet_model_v1.dat";
  frontal_face_detector faceDetector = get_frontal_face_detector();
  shape_predictor landmarkDetector;
  deserialize(predictorPath) >> landmarkDetector;
  anet_type net;
  deserialize(faceRecognitionModelPath) >> net;

  // read names, labels and labels-name-mapping from file
  std::map<int, string> labelNameMap;
  std::vector<string> names;
  std::vector<int> labels;
  const string labelNameFile = "label_name.txt";
  readLabelNameMap(labelNameFile, names, labels, labelNameMap);

  // read descriptors of enrolled faces from file
  const string faceDescriptorFile = "descriptors.csv";
  std::vector<int> faceLabels;
  std::vector<matrix<float,0,1>> faceDescriptors;
  readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);

  // read query image
  string imagePath;
  if (argc > 1) {
    imagePath = argv[1];
  } else {
    imagePath = "../images/shashikant-pedwal.jpg";
  }
  Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);

  if (im.empty()){
    exit(0);
  }
  double t = cv::getTickCount();
  // convert image from BGR to RGB
  // because Dlib used RGB format
  Mat imRGB = im.clone();
  cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);


  // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
  // Dlib's dnn module doesn't accept Dlib's cv_image template
  dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

  // detect faces in image
  std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
  std::string name;
  std::string nameText;
  float minDistance;
  // Now process each face we found

  // Find facial landmarks for each detected face
  full_object_detection landmarks = landmarkDetector(imDlib, faceRects[0]);

  // object to hold preProcessed face rectangle cropped from image
  matrix<rgb_pixel> face_chip;

  // original face rectangle is warped to 150x150 patch.
  // Same pre-processing was also performed during training.
  extract_image_chip(imDlib, get_face_chip_details(landmarks, 150, 0.25), face_chip);

  // Compute face descriptor using neural network defined in Dlib.
  // It is a 128D vector that describes the face in img identified by shape.
  matrix<float, 0, 1> faceDescriptorQuery = net(face_chip);

  // Find closest face enrolled to face found in frame
  int label;

  nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);
  // Name of recognized person from map
  name = labelNameMap[label];

  cout << "Time taken = " << ((double)cv::getTickCount() - t) / cv::getTickFrequency() << endl;

  Dict celeb = generateLabelMap();
  int size = celeb.size();
  std::cout << "celebrities size:: " << size << std::endl;

  nameText = "undefined";

  if (celeb.find(name) == celeb.end())
  {
    std::cout << "Doppelganger not found." << std::endl;
  }
  else
  {
    nameText = celeb.find(name)->second;
    std::cout << name << " " << nameText << std::endl;
  }

  // Draw a rectangle for detected face
  Point2d p1 = Point2d(faceRects[0].left(), faceRects[0].top());
  Point2d textOrigin = Point2d(10, 20);

  // Write text on image specifying identified person and minimum distance
  stringstream stream;
  stream << nameText << " ";
  stream << fixed << setprecision(4) << minDistance;
  string text = "original";
  cv::putText(im, text, textOrigin, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);

  //find image that of match
  string faceDatasetFolder = "../images/celeb_mini/";
  string matchDatasetFolder = faceDatasetFolder + name + "/";
  std::vector<string> fileNames;

  listdir(matchDatasetFolder, fileNames);
  string matchImage = fileNames.front();
  
  string matchImagePath = matchDatasetFolder + matchImage;
  Mat imMatch = cv::imread(matchImagePath, cv::IMREAD_COLOR);
  
  Point2d textOrigin2 = Point2d(10, 45);
  Point2d textOrigin3 = Point2d(10, 70);
  cv::putText(imMatch, "match", textOrigin, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);
  cv::putText(imMatch, nameText, textOrigin2, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);
  cv::putText(imMatch, std::to_string(minDistance), textOrigin3, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);

  // Show result
  cv::imshow("original", im);
  cv::imshow("match", imMatch);
  int k = cv::waitKey(0);

  cv::destroyAllWindows();
}
