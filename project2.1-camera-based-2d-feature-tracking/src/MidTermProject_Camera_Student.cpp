#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>

#include "dataStructures.h"
#include "matching2D.hpp"

int main(int argc, const char* argv[]) {
    /* Data stream config */
    std::string dataPath = "../";  // data location
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix =
        "KITTI/2011_09_26/image_00/data/000000";  // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0;  // first file index to load
    int imgEndIndex = 9;    // last file index to load
    int imgFillWidth = 4;   // size for fill up 0s

    /* Loop over data */
    int dataBufferSize = 2;
    std::deque<DataFrame> dataBuffer;
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex;
         imgIndex++) {
        /* Load image into buffer */
        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth)
                  << imgStartIndex + imgIndex;
        std::string imgFullFilename =
            imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // -----STUDENT ASSIGNMENT TASK MP.1-----
        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
        if (dataBuffer.size() > dataBufferSize) {
            dataBuffer.pop_front();
        }

        std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;
        // -----STUDENT ASSIGNMENT-----

        /* Detect keypoints */
        // -----STUDENT ASSIGNMENT TASK MP.2-----
        double detection_time = (double)cv::getTickCount();

        std::vector<cv::KeyPoint> keypoints;
        std::string detectorType =
            "SHITOMASI";  // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0) {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        } else if (detectorType.compare("HARRIS") == 0) {
            detKeypointsHarris(keypoints, imgGray, false);
        } else {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }

        detection_time = ((double)cv::getTickCount() - detection_time) /
                         cv::getTickFrequency();
        std::cout << detectorType << " detection with n=" << keypoints.size()
                  << " keypoints in " << 1000 * detection_time / 1.0 << " ms"
                  << std::endl;      
        // -----STUDENT ASSIGNMENT-----

        // -----STUDENT ASSIGNMENT TASK MP.3-----
        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle) {
            std::vector<cv::KeyPoint> focus_keypoints;
            for (const auto& it : keypoints) {
                if (vehicleRect.contains(it.pt)) {
                    focus_keypoints.push_back(it);
                }
            }
            keypoints = focus_keypoints;
        }
        // -----STUDENT ASSIGNMENT-----

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        std::cout << "#2 : DETECT KEYPOINTS done" << std::endl;

        /* Extract keypoint descriptors */
        // -----STUDENT ASSIGNMENT TASK MP.4-----
        double extraction_time = (double)cv::getTickCount();

        cv::Mat descriptors;
        std::string descriptorType =
            "BRISK";  // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints,
                      (dataBuffer.end() - 1)->cameraImg, descriptors,
                      descriptorType);

        extraction_time =
            ((double)cv::getTickCount() - extraction_time) / cv::getTickFrequency();
        std::cout << descriptorType << " descriptor extraction in "
                  << 1000 * extraction_time / 1.0 << " ms" << std::endl; 
        // -----STUDENT ASSIGNMENT-----

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;
        std::cout << "#3 : EXTRACT DESCRIPTORS done" << std::endl;

        /* Match keypoints */
        if (dataBuffer.size() > 1) {
            std::vector<cv::DMatch> matches;
            std::string matcherType = "MAT_BF";         // MAT_BF, MAT_FLANN
            std::string descriptorType = "DES_BINARY";  // DES_BINARY, DES_HOG
            std::string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            // -----STUDENT ASSIGNMENT TASK MP.5 AND MP.6-----
            matchDescriptors((dataBuffer.end() - 2)->keypoints,
                             (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors,
                             (dataBuffer.end() - 1)->descriptors, matches,
                             descriptorType, matcherType, selectorType);
            // -----STUDENT ASSIGNMENT-----

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
            std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

            // visualize matches between current and previous image
            cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
            cv::drawMatches((dataBuffer.end() - 2)->cameraImg,
                            (dataBuffer.end() - 2)->keypoints,
                            (dataBuffer.end() - 1)->cameraImg,
                            (dataBuffer.end() - 1)->keypoints, matches,
                            matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            std::vector<char>(),
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            std::string windowName =
                "Matching keypoints between two camera images";
            cv::namedWindow(windowName, 7);
            cv::imshow(windowName, matchImg);
            std::cout << "Press key to continue to next image" << std::endl;
            cv::waitKey(0);
        }
    }

    return 0;
}