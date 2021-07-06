The doppelganger project is completed for OpenCV's Computer Vision II, project 2.
The outcome enrolls a provided face data set and finds a match within that set for
a user provided image.

The code relies heavily on provided code from "Computer Vision for Faces" by Satya Mallick.

Paths are hard coded for expediency. The expected directory structure is:
.
├── cpp
│   ├── doppelgangerEnroll.cpp
│   ├── doppelgangerRecognize.cpp
│   ├── labelData.cpp
│   └── labelData.h
├── dlib
│   └── provided_dlib_files
├── images
│   ├── celeb_mini
│   │   ├── n00000001
│   │   ├── n00000002
│   │   ├── n00000003
│   │   └── n0000-etc
│   ├── shashikant-pedwal.jpg
│   └── sofia-solares.jpg
└── models
    └── provided_models

Below is the info from doppelgangerEnroll.cpp and doppelgangerRecognize.cpp:
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
 *            Code adapted from "Computer Vision for Faces" by Satya Mallick
 *            As noted in the original code "for personal non-commercial use".
 *            This program expects images in "../images/celeb_mini" (line 162)
 *            This program is set to filter for only JPEG files (line 209)
 *            Future modifications would see user able to input image path 
 *            and image filter types.          
 */

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
 *            Code adapted from "Computer Vision for Faces" by Satya Mallick
 *            As noted in the original code "for personal non-commercial use".
 *            Threshold is currently hard coded at 0.6 
 *            Future modifications would see user able to input threshold.        
 */


