# 3D Object Tracking
Before using this repository, please download the yolov3 weights first. The link can be found in /dat/yolo/yolov3.weights.

## FP.1 Match 3D Objects
Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.
``` c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
    std::map<std::pair<int, int>, int> matches_count;
    for (const auto &keypoint_match : matches) {
        for (int prev_bb_index = 0;
             prev_bb_index < prevFrame.boundingBoxes.size(); ++prev_bb_index) {
            if (prevFrame.boundingBoxes.at(prev_bb_index)
                    .roi.contains(
                        prevFrame.keypoints.at(keypoint_match.queryIdx).pt)) {
                for (int curr_bb_index = 0;
                     curr_bb_index < currFrame.boundingBoxes.size();
                     ++curr_bb_index) {
                    if (currFrame.boundingBoxes.at(curr_bb_index)
                            .roi.contains(
                                currFrame.keypoints.at(keypoint_match.trainIdx)
                                    .pt)) {
                        if (matches_count.find(
                                {prev_bb_index, curr_bb_index}) ==
                            matches_count.end()) {
                            matches_count.insert(
                                {{prev_bb_index, curr_bb_index}, 1});
                        } else {
                            matches_count.at({prev_bb_index, curr_bb_index}) +=
                                1;
                        }
                    }
                }
            }
        }
    }

    for (int prev_bb_index = 0; prev_bb_index < prevFrame.boundingBoxes.size();
         ++prev_bb_index) {
        std::pair<int, int> best_bb_match;
        int max_count = 0;
        for (const auto &it : matches_count) {
            if (it.first.first == prev_bb_index) {
                if (it.second > max_count) {
                    max_count = it.second;
                    best_bb_match = it.first;
                }
            }
        }

        if (max_count != 0) {
            bbBestMatches.insert({best_bb_match.first, best_bb_match.second});
        }
    }
}
```

## FP.2 Compute Lidar-based TTC
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.
``` c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate,
                     double &TTC) {
    // auxiliary variables
    double laneWidth = 4.0;  // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it) {
        if (it->y > laneWidth / 2. || it->y < -laneWidth / 2.) {
            continue;
        }

        // std::cout << "px: " << it->x << std::endl;
        if (it->x < minXPrev) {
            minXPrev = it->x;
        }
        // minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it) {
        if (it->y > laneWidth / 2. || it->y < -laneWidth / 2.) {
            continue;
        }
        // std::cout << "cx: " << it->x << std::endl;
        if (it->x < minXCurr) {
            minXCurr = it->x;
        }
    }
    std::cout << "minXPrev: " << minXPrev << std::endl;
    std::cout << "minXCurr: " << minXCurr << std::endl;
    std::cout << "minXPrev - minXCurr: " << minXPrev - minXCurr << std::endl;
    std::cout << "TTC: " << TTC << std::endl;
    // compute TTC from both measurements
    TTC = minXCurr * (1. / frameRate) / (minXPrev - minXCurr);
}
```

## FP.3 Associate Keypoint Correspondences with Bounding Boxes
Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.
``` c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {
    std::vector<double> distances;
    for (const auto &keypoint_match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr.at(keypoint_match.trainIdx).pt)) {
            distances.push_back(
                cv::norm(kptsCurr.at(keypoint_match.trainIdx).pt -
                         kptsPrev.at(keypoint_match.queryIdx).pt));
        }
    }
    double mean_distance =
        std::accumulate(distances.begin(), distances.end(), 0.0) /
        distances.size();

    for (const auto &keypoint_match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr.at(keypoint_match.trainIdx).pt)) {
            if (cv::norm(kptsCurr.at(keypoint_match.trainIdx).pt -
                         kptsPrev.at(keypoint_match.queryIdx).pt) <
                mean_distance * 1.5) {
                boundingBox.keypoints.push_back(
                    kptsCurr.at(keypoint_match.trainIdx));
                boundingBox.kptMatches.push_back(keypoint_match);
            }
        }
    }

    std::cout << "mean value: " << mean_distance
              << "Before filtering there are: " << distances.size()
              << " and after filtering, there are "
              << boundingBox.keypoints.size() << std::endl;
}
```

## FP.4 Compute Camera-based TTC
Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.
``` c++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC, cv::Mat *visImg) {
    // compute distance ratios between all matched keypoints
    vector<double> distRatios;  // stores the distance ratios for all keypoints
                                // between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1;
         ++it1) {  // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end();
             ++it2) {  // inner keypoint loop

            double minDist = 100.0;  // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() &&
                distCurr >= minDist) {  // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }  // eof inner loop over all matched kpts
    }      // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0) {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio =
        distRatios.size() % 2 == 0
            ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0
            : distRatios[medIndex];  // compute median dist. ratio to remove
                                     // outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}
```

## FP.5 Performance Evaluation 1
Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.
![image](fp5-1.png)
The TTC Lidar increse to 34s because the car starts to brake.

![image](fp5-2.png)
The TTC Lidar may produce negative values when the car is stationary due to Lidar noise, which can cause the previous position measurement (prev_x) to be smaller than the current position measurement (curr_x).

## FP.6 Performance Evaluation 2
Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

Using Top3 combinations from my mid-term project.
1. FAST + BRIEF
2. FAST + ORB
3. FAST + SIFT

Using the combinations above, the TTC values are stable.

When using a combination of ORB and SIFT, the TTC value may sometimes be 'inf' due to a `medDistRatio` = 1. This indicates that the keypoints are stationary.