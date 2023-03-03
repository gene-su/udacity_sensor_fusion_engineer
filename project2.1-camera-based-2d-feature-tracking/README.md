# Camera Based 2D Feature Tracking
## MP.1 Data Buffer Optimization
Use `std::deque` instead of `std::vector`.
``` c++
dataBuffer.push_back(frame);
if (dataBuffer.size() > dataBufferSize) {
	dataBuffer.pop_front();
}
```

## MP.2 Keypoint Detection
Fill up functions `detKeypointsHarris`, and `detKeypointsModern`.
``` c++
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

```

## MP.3 Keypoint Removal
``` c++
if (bFocusOnVehicle) {
	std::vector<cv::KeyPoint> focus_keypoints;
	for (const auto& it : keypoints) {
		if (vehicleRect.contains(it.pt)) {
			focus_keypoints.push_back(it);
		}
	}
	keypoints = focus_keypoints;
}
```

## MP.4 Keypoint Descriptors
``` c++
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
```

## MP.5 Descriptor Matching & MP.6 Descriptor Distance Ratio
``` c++
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
```

## MP.7 Performance Evaluation 1
Number of keypoints on the preceding vehicle for all 10 images.
| Detector\Image | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|
| ---      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shi-Tomasi   | 125 | 118 | 123 | 120 | 120 | 113 | 114 | 123 | 111 | 112 |
| Harris | 17 | 14 | 18 | 21 | 26 | 43 | 18 | 31 | 26 | 34 |
| FAST     | 419 | 427 | 404 | 423 | 386 | 414 | 418 | 406 | 396 | 401 |
| BRISK    | 264 | 282 | 282 | 277 | 297 | 279 | 289 | 272 | 266 | 254 |
| ORB      | 92  | 102 | 106 | 113 | 109 | 125 | 130 | 129 | 127 | 128 |
| AKAZE    | 166 | 157 | 161 | 155 | 163 | 164 | 173 | 175 | 177 | 179 |
| SIFT     | 138 | 132 | 124 | 137 | 134 | 140 | 137 | 148 | 159 | 137 |

## MP.8 Performance Evaluation 2
Number of matched keypoints for all 10 images using BF matcher with the descriptor distance ratio set to 0.8.
| Detector+Descriptor\Images | 0-1 | 1-2| 2-3| 3-4| 4-5| 5-6| 6-7| 7-8| 8-9|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shi-Tomasi + BRISK | 95 | 88 | 80 | 90 | 82 | 79 | 85 | 86 | 82 |
| Shi-Tomasi + BRIEF | 115 | 111 | 104 | 101 | 102 | 102 | 100 | 109 | 100 |
| Shi-Tomasi + ORB | 104 | 103 | 100 | 102 | 103 | 98 | 98 | 102 | 97 |
| Shi-Tomasi + FREAK | 90 | 88 | 87 | 89 | 83 | 78 | 81 | 86 | 84 |
| Shi-Tomasi + AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Shi-Tomasi + SIFT | 112 | 109 | 104 | 103 | 99 | 101 | 96 | 106 | 97 |
| Harris + BRISK | 12 | 10 | 14 | 15 | 16 | 16 | 15 | 23 | 21 |
| Harris + BRIEF | 14 | 11 | 15 | 20 | 24 | 26 | 16 | 24 | 23 |
| Harris + ORB | 12 | 13 | 16 | 18 | 24 | 18 | 15 | 24 | 20 |
| Harris + FREAK | 13 | 13 | 15 | 15 | 17 | 20 | 14 | 21 | 18 |
| Harris + AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Harris + SIFT | 14 | 11 | 16 | 19 | 22 | 22 | 13 | 24 | 22 |
| FAST + BRISK | 256 | 243 | 241 | 239 | 215 | 251 | 248 | 243 | 247 |
| FAST + BRIEF | 320 | 332 | 299 | 331 | 276 | 327 | 324 | 315 | 307 |
| FAST + ORB | 306 | 314 | 295 | 318 | 284 | 312 | 323 | 306 | 304 |
| FAST + FREAK | 251 | 250 | 228 | 252 | 234 | 269 | 252 | 243 | 246 |
| FAST + AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| FAST + SIFT | 316 | 325 | 297 | 311 | 291 | 326 | 315 | 300 | 301 |
| BRISK + BRISK | 171 | 176 | 157 | 176 | 174 | 188 | 173 | 171 | 184 |
| BRISK + BRIEF | 178 | 205 | 185 | 179 | 183 | 195 | 207 | 189 | 183 |
| BRISK + ORB | 160 | 171 | 157 | 170 | 154 | 180 | 171 | 175 | 172 |
| BRISK + FREAK | 160 | 178 | 156 | 173 | 160 | 183 | 169 | 179 | 168 |
| BRISK + AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| BRISK + SIFT | 182 | 193 | 169 | 183 | 171 | 195 | 194 | 176 | 183 |
| ORB + BRISK | 73 | 74 | 79 | 85 | 79 | 92 | 90 | 88 | 91 |
| ORB + BRIEF | 49 | 43 | 45 | 59 | 53 | 78 | 68 | 84 | 66 |
| ORB + ORB | 65 | 69 | 71 | 85 | 91 | 101 | 95 | 93 | 91 |
| ORB + FREAK | 42 | 36 | 45 | 47 | 44 | 51 | 52 | 49 | 55 |
| ORB + AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ORB + SIFT | 67 | 79 | 78 | 79 | 82 | 95 | 95 | 94 | 94 |
| AKAZE + BRISK | 137 | 125 | 129 | 129 | 131 | 132 | 142 | 146 | 144 |
| AKAZE + BRIEF | 141 | 134 | 131 | 130 | 134 | 146 | 150 | 148 | 152 |
| AKAZE + ORB | 130 | 129 | 128 | 115 | 132 | 132 | 137 | 137 | 146 |
| AKAZE + FREAK | 126 | 128 | 128 | 121 | 123 | 132 | 145 | 148 | 137 |
| AKAZE + AKAZE | 138 | 138 | 133 | 127 | 129 | 146 | 147 | 151 | 150 |
| AKAZE + SIFT | 134 | 134 | 130 | 136 | 137 | 147 | 147 | 154 | 151 |
| SIFT + BRISK | 64 | 66 | 62 | 66 | 59 | 64 | 64 | 67 | 80 |
| SIFT + BRIEF | 86 | 78 | 76 | 85 | 69 | 74 | 76 | 70 | 88 |
| SIFT + ORB | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SIFT + FREAK | 64 | 72 | 65 | 66 | 63 | 58 | 64 | 65 | 79 |
| SIFT + AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SIFT + SIFT | 82 | 81 | 85 | 93 | 90 | 81 | 82 | 102 | 104 |

## MP.9 Performance Evaluation 3
Time for keypoint detection and descriptor extraction.
| Detector+Descriptor\Time(ms) | Detection | Extraction | Sum |
| --- | --- | --- | --- |
| Shi-Tomasi + BRISK | 8.22888 | 22.661 | 31.0236 |
| Shi-Tomasi + BRIEF | 8.20676 | 0.667094 | 8.87386 | 
| Shi-Tomasi + ORB | 8.00189 | 0.907153 | 8.90904 |
| Shi-Tomasi + FREAK | 6.96438 | 27.2902 | 34.2546 |
| Shi-Tomasi + AKAZE | N/A | N/A | N/A |
| Shi-Tomasi + SIFT | 8.04436 | 11.9622 | 20.0065 |
| Harris + BRISK | 9.56001 | 21.4799 | 31.0399 |
| Harris + BRIEF | 9.86325 | 0.40955 | 10.2728 |
| Harris + ORB | 9.1708 | 0.721549 | 9.89235 |
| Harris + FREAK | 8.55559 | 26.2348 | 34.7904 |
| Harris + AKAZE | N/A | N/A | N/A |
| Harris + SIFT | 9.52799 | 11.3419 | 20.8699 |
| FAST + BRISK | 1.44117 | 24.5111 | 25.9523 |
| FAST + BRIEF | 1.47507 | 1.2419 | 2.71697 |
| FAST + ORB | 1.53336 | 1.63575 | 3.16911 |
| FAST + FREAK | 1.48604 | 28.9151 | 30.4011 |
| FAST + AKAZE | N/A | N/A | N/A |
| FAST + SIFT | 1.44784 | 16.8935 | 18.3413 |
| BRISK + BRISK | 48.9428 | 22.4312 | 72.4091 |
| BRISK + BRIEF | 49.2451 | 0.710449 | 49.9556 |
| BRISK + ORB | 49.2736 | 3.97976 | 53.2534 |
| BRISK + FREAK | 49.2108 | 27.2061 | 76.417 |
| BRISK + AKAZE | N/A | N/A | N/A |
| BRISK + SIFT | 54.0114 | 22.9465 | 76.9579 |
| ORB + BRISK | 5.8311 | 21.3405 | 27.1716 |
| ORB + BRIEF | 6.27655 | 0.45145 | 6.728 |
| ORB + ORB | 6.11304 | 3.94289 | 10.0559 |
| ORB + FREAK | 6.13098 | 27.2013 | 33.3323 |
| ORB + AKAZE | N/A | N/A | N/A |
| ORB + SIFT | 6.26056 | 23.2351 | 29.4957 |
| AKAZE + BRISK | 41.1825 | 20.797 | 61.9796 |
| AKAZE + BRIEF | 41.9021 | 0.53127 | 42.4334 |
| AKAZE + ORB | 41.6394 | 2.34117 | 43.9806 |
| AKAZE + FREAK | 40.1799 | 26.4239 | 66.6039 |
| AKAZE + AKAZE | 41.5818 | 33.1663 | 74.7481 |
| AKAZE + SIFT | 41.6546 | 13.3282 | 54.9827 |
| SIFT + BRISK | 61.6025 | 15.2033 | 76.8058 |
| SIFT + BRIEF | 64.6585 | 0.625924 | 65.2845 |
| SIFT + ORB | N/A | N/A | N/A |
| SIFT + FREAK | 59.1388 | 32.9744 | 92.1132 |
| SIFT + AKAZE | N/A | N/A | N/A |
| SIFT + SIFT | 61.2827 | 55.7907 | 117.073 |

TOP3 detector / descriptor combinations
1. FAST + BRIEF
2. FAST + ORB
3. FAST + SIFT

FAST has the most keypoints detected. As for the descriptors, comparing speed between each descriptors, I choose three descriptors with lesser proccessing time.