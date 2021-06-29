/*
 * File:      doppelgangerEnroll.cpp
 * Author:    Richard Purcell
 * Date:      2021-06-28
 * Version:   1.0
 * Purpose:   Enroll a set of images and output a set of descriptors.
 * Usage:     $ ./doppelgangerEnroll
 * Notes:     Created for OpenCV's Computer Vision 2 Project 2.
 *            This file is heavily based on enrollDlibFaceRec.cpp,
 *            provided for Computer Vision 2, week 4.
 *            This program expects images in "../images/celeb_mini"
 *            
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>

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

// ----------------------------------------------------------------------------------------
// The next bit of code defines a ResNet network. It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
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

// Reads files, folders and symbolic links in a directory
void listdir(string dirName, std::vector<string>& folderNames, std::vector<string>& fileNames, std::vector<string>& symlinkNames) {
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(dirName.c_str())) != NULL) {
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
        case DT_DIR:
          folderNames.push_back(dirName + "/" + temp_name);
          break;
        case DT_LNK:
          symlinkNames.push_back(temp_name);
          break;
        default:
          break;
      }
      // cout << temp_name << endl;
    }
    // sort all the files
    std::sort(folderNames.begin(), folderNames.end());
    std::sort(fileNames.begin(), fileNames.end());
    std::sort(symlinkNames.begin(), symlinkNames.end());
    closedir(dir);
  }
}

// filter files having extension ext i.e. jpg
void filterFiles(string dirPath, std::vector<string>& fileNames, std::vector<string>& filteredFilePaths, string ext, std::vector<int>& imageLabels, int index){
  for(int i = 0; i < fileNames.size(); i++) {
    string fname = fileNames[i];
    if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos) {
      filteredFilePaths.push_back(dirPath + "/" + fname);
      imageLabels.push_back(index);
    }
  }
}

template<typename T>
void printVector(std::vector<T>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    cout << i << " " << vec[i] << "; ";
  }
  cout << endl;
}

int main() {
  // Initialize face detector, facial landmarks detector and face recognizer
  String predictorPath, faceRecognitionModelPath;
  predictorPath = "../models/shape_predictor_68_face_landmarks.dat";
  faceRecognitionModelPath = "../models/dlib_face_recognition_resnet_model_v1.dat";
  frontal_face_detector faceDetector = get_frontal_face_detector();
  shape_predictor landmarkDetector;
  deserialize(predictorPath) >> landmarkDetector;
  anet_type net;
  deserialize(faceRecognitionModelPath) >> net;

  // Now let's prepare our training data
  // data is organized assuming following structure
  // faces folder has subfolders.
  // each subfolder has images of a person
  string faceDatasetFolder = "../images/celeb_mini";
  std::vector<string> subfolders, fileNames, symlinkNames;
  // fileNames and symlinkNames are useless here
  // as we are looking for sub-directories only
  listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);

  // names: vector containing names of subfolders i.e. persons
  // labels: integer labels assigned to persons
  // labelNameMap: dict containing (integer label, person name) pairs
  std::vector<string> names;
  std::vector<int> labels;
  std::map<int, string> labelNameMap;
  // add -1 integer label for un-enrolled persons
  names.push_back("unknown");
  labels.push_back(-1);

  // imagePaths: vector containing imagePaths
  // imageLabels: vector containing integer labels corresponding to imagePaths
  std::vector<string> imagePaths;
  std::vector<int> imageLabels;

  // variable to hold any subfolders within person subFolders
  std::vector<string> folderNames;
  // iterate over all subFolders within faces folder
  for (int i = 0; i < subfolders.size(); i++) {
    string personFolderName = subfolders[i];
    // remove / or \\ from end of subFolder
    std::size_t found = personFolderName.find_last_of("/\\");
    string name = personFolderName.substr(found+1);
    // assign integer label to person subFolder
    int label = i;
    // add person name and label to vectors
    names.push_back(name);
    labels.push_back(label);
    // add (integer label, person name) pair to map
    labelNameMap[label] = name;

    // read imagePaths from each person subFolder
    // clear vectors
    folderNames.clear();
    fileNames.clear();
    symlinkNames.clear();
    // folderNames and symlinkNames are useless here
    // as we are only looking for files here
    // read all files present in subFolder
    listdir(subfolders[i], folderNames, fileNames, symlinkNames);
    // filter only jpg files
    filterFiles(subfolders[i], fileNames, imagePaths, "JPEG", imageLabels, i);
    }

  // process training data
  // We will store face descriptors in vector faceDescriptors
  // and their corresponding labels in vector faceLabels
  std::vector<matrix<float,0,1>> faceDescriptors;
  // std::vector<cv_image<bgr_pixel> > imagesFaceTrain;
  std::vector<int> faceLabels;

  // iterate over images
  for (int i = 0; i < imagePaths.size(); i++) {
    string imagePath = imagePaths[i];
    int imageLabel = imageLabels[i];

    cout << "processing: " << imagePath << endl;

    // read image using OpenCV
    Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);

    // convert image from BGR to RGB
    // because Dlib used RGB format
    Mat imRGB;
    cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);

    // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
    // Dlib's dnn module doesn't accept Dlib's cv_image template
    dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

    // detect faces in image
    std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
    cout << faceRects.size() << " Face(s) Found" << endl;
    // Now process each face we found
    for (int j = 0; j < faceRects.size(); j++) {

      // Find facial landmarks for each detected face
      full_object_detection landmarks = landmarkDetector(imDlib, faceRects[j]);

      // object to hold preProcessed face rectangle cropped from image
      matrix<rgb_pixel> face_chip;

      // original face rectangle is warped to 150x150 patch.
      // Same pre-processing was also performed during training.
      extract_image_chip(imDlib, get_face_chip_details(landmarks, 150, 0.25), face_chip);

      // Compute face descriptor using neural network defined in Dlib.
      // It is a 128D vector that describes the face in img identified by shape.
      matrix<float,0,1> faceDescriptor = net(face_chip);

      // add face descriptor and label for this face to
      // vectors faceDescriptors and faceLabels
      faceDescriptors.push_back(faceDescriptor);
      // add label for this face to vector containing labels corresponding to
      // vector containing face descriptors
      faceLabels.push_back(imageLabel);
    }
  }

  cout << "number of face descriptors " << faceDescriptors.size() << endl;
  cout << "number of face labels " << faceLabels.size() << endl;

  // write label name map to disk
  const string labelNameFile = "label_name.txt";
  ofstream of;
  of.open (labelNameFile);
  for (int m = 0; m < names.size(); m++) {
    of << names[m];
    of << ";";
    of << labels[m];
    of << "\n";
  }
  of.close();

  // write face labels and descriptor to disk
  // each row of file descriptors.csv has:
  // 1st element as face label and
  // rest 128 as descriptor values
  const string descriptorsPath = "descriptors.csv";
  ofstream ofs;
  ofs.open(descriptorsPath);
  // write descriptors
  for (int m = 0; m < faceDescriptors.size(); m++) {
    matrix<float,0,1> faceDescriptor = faceDescriptors[m];
    std::vector<float> faceDescriptorVec(faceDescriptor.begin(), faceDescriptor.end());
    // cout << "Label " << faceLabels[m] << endl;
    ofs << faceLabels[m];
    ofs << ";";
    for (int n = 0; n < faceDescriptorVec.size(); n++) {
      ofs << std::fixed << std::setprecision(8) << faceDescriptorVec[n];
      // cout << n << " " << faceDescriptorVec[n] << endl;
      if ( n == (faceDescriptorVec.size() - 1)) {
        ofs << "\n";  // add ; if not the last element of descriptor
      } else {
        ofs << ";";  // add newline character if last element of descriptor
      }
    }
  }
  ofs.close();
  return 1;
}
