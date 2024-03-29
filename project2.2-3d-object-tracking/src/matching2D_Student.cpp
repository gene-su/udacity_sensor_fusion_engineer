#include <numeric>

#include "matching2D.hpp"

// Find best matches for keypoints in two camera images based on several
// matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    int normType;
    if (descriptorType.compare("DES_BINARY") == 0) {
        normType = cv::NORM_HAMMING;
    } else {
        normType = cv::NORM_L2;
    }

    if (matcherType.compare("MAT_BF") == 0) {
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType.compare("MAT_FLANN") == 0) {
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
        matcher =
            cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0) {  // nearest neighbor (best match)
        matcher->match(
            descSource, descRef,
            matches);  // Finds the best match for each descriptor in desc1
    } else if (selectorType.compare("SEL_KNN") ==
               0) {  // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches,
                          2);  // finds the 2 best matches
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
                matches.push_back((*it)[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify
// keypoints
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, std::string descriptorType) {
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {
        extractor = cv::BRISK::create();
    } else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create();
    } else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
    } else if (descriptorType.compare("SIFT") == 0) {
        extractor = cv::SIFT::create();
    }

    extractor->compute(img, keypoints, descriptors);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                           bool bVis) {
    // compute detector parameters based on image size
    int blockSize = 4;  //  size of an average block for computing a derivative
                        //  covariation matrix over each pixel neighborhood
    double maxOverlap =
        0.0;  // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols /
                     std::max(1.0, minDistance);  // max. num. of keypoints

    double qualityLevel = 0.01;  // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
                            cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        bool bVis) {
    // Detector parameters
    int blockSize = 2;  // for every pixel, a blockSize × blockSize neighborhood
                        // is considered
    int apertureSize =
        3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse =
        100;  // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;  // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0;  // max. permissible overlap between two features
                              // in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);
            if (response >
                minResponse) {  // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood
                // around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap) {
                        bOverlap = true;
                        if (newKeyPoint.response >
                            (*it).response) {   // if overlap is >t AND response
                                                // is higher for new kpt
                            *it = newKeyPoint;  // replace old key point with
                                                // new one
                            break;              // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap) {  // only add new key point if no overlap has
                                  // been found in previous NMS
                    keypoints.push_back(
                        newKeyPoint);  // store new keypoint in dynamic list
                }
            }
        }
    }

    if (bVis) {
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        std::string detectorType, bool bVis) {
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0) {
        detector = cv::FastFeatureDetector::create();
    } else if (detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create();
    } else if (detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();
    } else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();
    } else if (detectorType.compare("SIFT") == 0) {
        detector = cv::SIFT::create();
    }

    detector->detect(img, keypoints);

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
