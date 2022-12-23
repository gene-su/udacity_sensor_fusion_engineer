// PCL lib Functions for processing point clouds
#include "processPointClouds.h"

// constructor:
template <typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}

// de-constructor:
template <typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}

template <typename PointT>
void ProcessPointClouds<PointT>::numPoints(
    typename pcl::PointCloud<PointT>::Ptr cloud) {
    std::cout << cloud->points.size() << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(
    typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes,
    Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint) {
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // voxel grid point reduction
    typename pcl::PointCloud<PointT>::Ptr filtered_cloud(
        new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> voxel_grid_filter;
    voxel_grid_filter.setInputCloud(cloud);
    voxel_grid_filter.setLeafSize(filterRes, filterRes, filterRes);
    voxel_grid_filter.filter(*filtered_cloud);

    // region based filtering
    typename pcl::PointCloud<PointT>::Ptr intrested_cloud(
        new pcl::PointCloud<PointT>);
    pcl::CropBox<PointT> intrested_region_filter;
    intrested_region_filter.setMin(minPoint);
    intrested_region_filter.setMax(maxPoint);
    intrested_region_filter.setInputCloud(filtered_cloud);
    intrested_region_filter.filter(*intrested_cloud);

    // roof noise extraction
    pcl::CropBox<PointT> roof_region_filter(true);
    intrested_region_filter.setMin({-3.0, -3.0, -1.0, -1.0});
    intrested_region_filter.setMax({3.0, 3.0, 1.0, 1.0});
    intrested_region_filter.setInputCloud(intrested_cloud);
    intrested_region_filter.setNegative(true);
    intrested_region_filter.filter(*intrested_cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds"
              << std::endl;

    return intrested_cloud;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,
          typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::SeparateClouds(
    pcl::PointIndices::Ptr inliers,
    typename pcl::PointCloud<PointT>::Ptr cloud) {
    // TODO: Create two new point clouds, one cloud with obstacles and other
    // with segmented plane
    typename pcl::PointCloud<PointT>::Ptr plane_cloud(
        new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr obstacle_cloud(
        new pcl::PointCloud<PointT>());

    for (const int& index : inliers->indices) {
        plane_cloud->points.push_back(cloud->points[index]);
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstacle_cloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr,
              typename pcl::PointCloud<PointT>::Ptr>
        segResult(obstacle_cloud, plane_cloud);
    return segResult;
}

template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,
          typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::SegmentPlane(
    typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations,
    float distanceThreshold) {
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    /* My Segmentation */
    std::unordered_set<int> inliers =
        Ransac(cloud, maxIterations, distanceThreshold);

    typename pcl::PointCloud<PointT>::Ptr inlier_cloud(
        new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr outlier_cloud(
        new pcl::PointCloud<PointT>());
    for (int index = 0; index < cloud->points.size(); index++) {
        PointT point = cloud->points[index];
        if (inliers.count(index))
            inlier_cloud->points.push_back(point);
        else
            outlier_cloud->points.push_back(point);
    }

    std::pair<typename pcl::PointCloud<PointT>::Ptr,
              typename pcl::PointCloud<PointT>::Ptr>
        segResult = {outlier_cloud, inlier_cloud};

    /* PCL Segmentation */
    // pcl::SACSegmentation<PointT> seg;
    // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    // seg.setOptimizeCoefficients(true);
    // seg.setModelType(pcl::SACMODEL_PLANE);
    // seg.setMethodType(pcl::SAC_RANSAC);
    // seg.setDistanceThreshold(distanceThreshold);
    // seg.setMaxIterations(maxIterations);

    // seg.setInputCloud(cloud);
    // seg.segment(*inliers, *coefficients);

    // if (inliers->indices.size() == 0) {
    //     std::cout << "Could not estimate a planar model for the given
    //     dataset."
    //               << std::endl;
    // }
    // std::pair<typename pcl::PointCloud<PointT>::Ptr,
    //           typename pcl::PointCloud<PointT>::Ptr>
    //     segResult = SeparateClouds(inliers, cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count()
              << " milliseconds" << std::endl;

    return segResult;
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr>
ProcessPointClouds<PointT>::Clustering(
    typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance,
    int minSize, int maxSize) {
    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    /* My Clustering */
    std::vector<std::vector<float>> points;
    for (const auto& point : cloud->points) {
        points.push_back({point.x, point.y, point.z});
    }
    KdTree* tree = new KdTree;
    for (int i = 0; i < points.size(); ++i) {
        tree->insert(points.at(i), i);
    }
    std::vector<std::vector<int>> clustered_indices =
        euclideanCluster(points, tree, clusterTolerance, minSize, maxSize);

    for (std::vector<int> indices : clustered_indices) {
        typename pcl::PointCloud<PointT>::Ptr clustered_cloud(
            new pcl::PointCloud<PointT>);
        for (int index : indices) {
            PointT point;
            point.x = points[index][0];
            point.y = points[index][1];
            point.z = points[index][2];
            clustered_cloud->points.push_back(point);
        }
        clusters.push_back(clustered_cloud);
    }

    /* PCL Clustering */
    // typename pcl::search::KdTree<PointT>::Ptr tree(
    //     new pcl::search::KdTree<PointT>);
    // tree->setInputCloud(cloud);
    // std::vector<pcl::PointIndices> cluster_indices;
    // pcl::EuclideanClusterExtraction<PointT> ec;
    // ec.setClusterTolerance(clusterTolerance);
    // ec.setMinClusterSize(minSize);
    // ec.setMaxClusterSize(maxSize);
    // ec.setSearchMethod(tree);
    // ec.setInputCloud(cloud);
    // ec.extract(cluster_indices);

    // for (const auto& cluster : cluster_indices) {
    //     typename pcl::PointCloud<PointT>::Ptr cloud_cluster(
    //         new pcl::PointCloud<PointT>);
    //     for (const auto& index : cluster.indices) {
    //         cloud_cluster->push_back((*cloud)[index]);
    //     }
    //     cloud_cluster->width = cloud_cluster->size();
    //     cloud_cluster->height = 1;
    //     cloud_cluster->is_dense = true;

    //     clusters.push_back(cloud_cluster);
    // }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count()
              << " milliseconds and found " << clusters.size() << " clusters"
              << std::endl;

    return clusters;
}

template <typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(
    typename pcl::PointCloud<PointT>::Ptr cluster) {
    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

template <typename PointT>
void ProcessPointClouds<PointT>::savePcd(
    typename pcl::PointCloud<PointT>::Ptr cloud, std::string file) {
    pcl::io::savePCDFileASCII(file, *cloud);
    std::cerr << "Saved " << cloud->points.size() << " data points to " + file
              << std::endl;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(
    std::string file) {
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1)  //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size()
              << " data points from " + file << std::endl;

    return cloud;
}

template <typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(
    std::string dataPath) {
    std::vector<boost::filesystem::path> paths(
        boost::filesystem::directory_iterator{dataPath},
        boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;
}

template <typename PointT>
std::unordered_set<int> ProcessPointClouds<PointT>::Ransac(
    typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations,
    float distanceTol) {
    std::unordered_set<int> inliersResult;
    if (cloud->points.empty()) {
        return inliersResult;
    } else if (cloud->points.size() == 1) {
        inliersResult.insert(0);
        return inliersResult;
    } else if (cloud->points.size() == 2) {
        inliersResult.insert(0);
        inliersResult.insert(1);
        return inliersResult;
    }

    srand(time(NULL));
    for (int i = 0; i < maxIterations; ++i) {
        std::unordered_set<int> inliers;
        // Randomly sample subset and fit line
        while (inliers.size() < 3) {
            inliers.insert(rand() % cloud->points.size());
        }

        auto iter = inliers.begin();
        Eigen::Vector3f point1 = {cloud->points[*iter].x,
                                  cloud->points[*iter].y,
                                  cloud->points[*iter].z};
        ++iter;
        Eigen::Vector3f point2 = {cloud->points[*iter].x,
                                  cloud->points[*iter].y,
                                  cloud->points[*iter].z};
        ++iter;
        Eigen::Vector3f point3 = {cloud->points[*iter].x,
                                  cloud->points[*iter].y,
                                  cloud->points[*iter].z};

        Eigen::Vector3f vector1 = point2 - point1;
        Eigen::Vector3f vector2 = point3 - point1;
        Eigen::Vector3f normal_vector = vector1.cross(vector2);

        float a = normal_vector.x();
        float b = normal_vector.y();
        float c = normal_vector.z();
        float d =
            -(normal_vector.x() * point1.x() + normal_vector.y() * point1.y() +
              normal_vector.z() * point1.z());

        // Measure distance between every point and fitted line
        for (int index = 0; index < cloud->points.size(); ++index) {
            if (inliers.count(index) > 0) {
                continue;
            }

            float distance = fabs(a * cloud->points.at(index).x +
                                  b * cloud->points.at(index).y +
                                  c * cloud->points.at(index).z + d) /
                             sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));

            // If distance is smaller than threshold count it as inlier
            if (distance <= distanceTol) {
                inliers.insert(index);
            }
        }

        if (inliers.size() > inliersResult.size()) {
            inliersResult = inliers;
        }
    }

    // Return indicies of inliers from fitted line with most inliers
    return inliersResult;
}

template <typename PointT>
void ProcessPointClouds<PointT>::SearchCluster(
    const int& index, const std::vector<std::vector<float>>& points,
    const float distanceTol, std::vector<bool>& is_processed, KdTree* tree,
    std::vector<int>& cluster) {
    is_processed.at(index) = true;
    cluster.push_back(index);
    std::vector<int> nearby_points =
        tree->search(points.at(index), distanceTol);
    for (const auto& i : nearby_points) {
        if (is_processed.at(i)) {
            continue;
        }

        SearchCluster(i, points, distanceTol, is_processed, tree, cluster);
    }
}

template <typename PointT>
std::vector<std::vector<int>> ProcessPointClouds<PointT>::euclideanCluster(
    const std::vector<std::vector<float>>& points, KdTree* tree,
    float distanceTol, int minSize, int maxSize) {
    std::vector<std::vector<int>> clusters;

    std::vector<bool> is_processed(points.size(), false);
    for (int i = 0; i < points.size(); ++i) {
        if (is_processed.at(i)) {
            continue;
        }

        std::vector<int> cluster;
        SearchCluster(i, points, distanceTol, is_processed, tree, cluster);
        if (cluster.size() >= minSize && cluster.size() <= maxSize) {
            clusters.push_back(cluster);
        }
    }

    return clusters;
}