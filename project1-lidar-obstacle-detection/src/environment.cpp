#include "processPointClouds.cpp"
#include "processPointClouds.h"
#include "render/render.h"

void cityBlock(pcl::visualization::PCLVisualizer::Ptr& viewer,
               ProcessPointClouds<pcl::PointXYZI>& process_point_cloud,
               const pcl::PointCloud<pcl::PointXYZI>::Ptr& inputCloud) {
    // Do voxel grid point reduction and region based filtering
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_intrested_cloud =
        process_point_cloud.FilterCloud(inputCloud, 0.3, {-10, -5, -2, 1},
                                        {30, 8, 1, 1});
    // renderPointCloud(viewer, filtered_intrested_cloud,
    //                  "filtered_intrested_cloud");

    // Segment to obstacle and road clouds
    std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr,
              pcl::PointCloud<pcl::PointXYZI>::Ptr>
        segmented_clouds = process_point_cloud.SegmentPlane(
            filtered_intrested_cloud, 100, 0.2);
    // renderPointCloud(viewer, segmented_clouds.first, "obstacle_cloud",
    //                  Color{1, 0, 0});
    renderPointCloud(viewer, segmented_clouds.second, "road_cloud",
                     Color{0, 1, 0});

    // Cluster obstacles and box them
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters =
        process_point_cloud.Clustering(segmented_clouds.first, 0.5, 30, 1000);

    int clusterId = 0;
    std::vector<Color> colors = {Color(1, 0, 0), Color(1, 1, 0),
                                 Color(0, 0, 1)};

    for (pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : cloudClusters) {
        process_point_cloud.numPoints(cluster);
        renderPointCloud(viewer, cluster,
                         "obstCloud" + std::to_string(clusterId),
                         colors[clusterId]);

        Box box = process_point_cloud.BoundingBox(cluster);
        renderBox(viewer, box, clusterId);
        ++clusterId;
    }
}

// setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle,
                pcl::visualization::PCLVisualizer::Ptr& viewer) {
    viewer->setBackgroundColor(0, 0, 0);

    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 16;

    switch (setAngle) {
        case XY:
            viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0);
            break;
        case TopDown:
            viewer->setCameraPosition(0, 0, distance, 1, 0, 1);
            break;
        case Side:
            viewer->setCameraPosition(0, -distance, 0, 0, 0, 1);
            break;
        case FPS:
            viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }

    if (setAngle != FPS) viewer->addCoordinateSystem(1.0);
}

int main(int argc, char** argv) {
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("3D Viewer"));
    CameraAngle setAngle = FPS;
    initCamera(setAngle, viewer);

    ProcessPointClouds<pcl::PointXYZI> process_point_cloud;
    std::vector<boost::filesystem::path> stream =
        process_point_cloud.streamPcd("../src/data/data1");
    auto streamIterator = stream.begin();
    while (!viewer->wasStopped()) {
        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        
        // Load pcd and run obstacle detection process
        pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI =
            process_point_cloud.loadPcd((*streamIterator).string());
        cityBlock(viewer, process_point_cloud, inputCloudI);

        streamIterator++;
        if (streamIterator == stream.end()) streamIterator = stream.begin();

        viewer->spinOnce();
    }
}