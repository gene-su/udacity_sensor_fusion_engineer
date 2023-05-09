#include "ukf.h"

#include <iostream>

#include "Eigen/Dense"

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    is_initialized_ = false;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    Xsig_pred_ = Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1);

    lambda_ = 3 - n_aug_;

    // initial state vector
    x_ = Eigen::VectorXd(n_x_);

    // initial covariance matrix
    P_ = Eigen::MatrixXd(n_x_, n_x_);

    // Weights of sigma points
    weights_ = Eigen::VectorXd::Constant(2 * n_aug_ + 1,
                                         1. / (2 * (lambda_ + n_aug_)));
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    // Sigma point spreading parameter

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3.;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 2.;

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values
     */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Make sure you switch between lidar and
     * radar measurements.
     */
    if (!is_initialized_) {
        if (meas_package.sensor_type_ ==
            MeasurementPackage::SensorType::LASER) {
            x_ = Eigen::VectorXd::Zero(n_x_);
            x_(0) = meas_package.raw_measurements_(0);
            x_(1) = meas_package.raw_measurements_(1);
            P_ = Eigen::MatrixXd::Identity(n_x_, n_x_);
            P_(0, 0) = pow(std_laspx_, 2);
            P_(1, 1) = pow(std_laspy_, 2);
        } else if (meas_package.sensor_type_ ==
                   MeasurementPackage::SensorType::RADAR) {
            x_ = Eigen::VectorXd::Zero(n_x_);
            double rho = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            x_(0) = rho * cos(phi);
            x_(1) = rho * sin(phi);
            P_ = Eigen::MatrixXd::Identity(n_x_, n_x_);			
        } else {
            std::cout << "Invalid sensor type" << std::endl;
            return;
        }

        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }

    // Predict
    double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);

    // Update
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
        UpdateLidar(meas_package);
    } else if (meas_package.sensor_type_ ==
               MeasurementPackage::SensorType::RADAR) {
        UpdateRadar(meas_package);
    } else {
        std::cout << "Invalid sensor type" << std::endl;
        return;
    }
}

void UKF::Prediction(double delta_t) {
    /**
     * TODO: Complete this function! Estimate the object's location.
     * Modify the state vector, x_. Predict sigma points, the state,
     * and the state covariance matrix.
     */

    /* Generating sigma points */
    Eigen::VectorXd x_aug = Eigen::VectorXd::Zero(n_aug_);
    Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(n_aug_, n_aug_);
    Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

    x_aug.head(n_x_) = x_;

    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_aug_ - 2, n_aug_ - 2) = pow(std_a_, 2);
    P_aug(n_aug_ - 1, n_aug_ - 1) = pow(std_yawdd_, 2);

    Eigen::MatrixXd A = P_aug.llt().matrixL();
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
        Xsig_aug.col(n_aug_ + i + 1) =
            x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
    }

    /* Sigma points prediction */
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        double px = Xsig_aug(0, i);
        double py = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // motion function
        if (fabs(yawd) > 1e-3) {
            Xsig_pred_(0, i) =
                px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            Xsig_pred_(1, i) =
                py + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            Xsig_pred_(0, i) = px + v * delta_t * cos(yaw);
            Xsig_pred_(1, i) = py + v * delta_t * sin(yaw);
        }
        Xsig_pred_(2, i) = v;
        Xsig_pred_(3, i) = yaw + yawd * delta_t;
        Xsig_pred_(4, i) = yawd;

        // add noise
        Xsig_pred_(0, i) += 0.5 * nu_a * pow(delta_t, 2) * cos(yaw);
        Xsig_pred_(1, i) += 0.5 * nu_a * pow(delta_t, 2) * sin(yaw);
        Xsig_pred_(2, i) += nu_a * delta_t;
        Xsig_pred_(3, i) += 0.5 * nu_yawdd * pow(delta_t, 2);
        Xsig_pred_(4, i) += nu_yawdd * delta_t;
    }

    /* Sigma points to prediected mean and covariance */
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;

        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */
    int n_z = meas_package.raw_measurements_.size();

    /* Real measurement */
    Eigen::VectorXd z = Eigen::VectorXd(n_z);
    z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);

    /* Sigma points prediction */
    Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, 2 * n_aug_ + 1);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        // measurement model
        Zsig(0, i) = Xsig_pred_(0, i);
        Zsig(1, i) = Xsig_pred_(1, i);
    }

    /* Sigma points to prediected mean and covariance */
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(n_z);
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_z, n_z);

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        S += weights_(i) * z_diff * z_diff.transpose();
    }
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n_z, n_z);
    R(0, 0) = pow(std_laspx_, 2);
    R(1, 1) = pow(std_laspy_, 2);
    S += R;

    /* Update */
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    Eigen::MatrixXd K = Tc * S.inverse();

    // New mean and covariance
    x_ += K * (z - z_pred);
    P_ -= K * S * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */
    int n_z = meas_package.raw_measurements_.size();

    /* Real measurement */
    Eigen::VectorXd z = Eigen::VectorXd(n_z);
    z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1),
        meas_package.raw_measurements_(2);

    /* Sigma points prediction */
    Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, 2 * n_aug_ + 1);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        // measurement model
        Zsig(0, i) = sqrt(pow(px, 2) + pow(py, 2));
        Zsig(1, i) = atan2(py, px);
        Zsig(2, i) = (px * cos(yaw) * v + py * sin(yaw) * v) /
                     sqrt(pow(px, 2) + pow(py, 2));
    }

    /* Sigma points to prediected mean and covariance */
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(n_z);
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_z, n_z);

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n_z, n_z);
    R(0, 0) = pow(std_radr_, 2);
    R(1, 1) = pow(std_radphi_, 2);
    R(2, 2) = pow(std_radrd_, 2);
    S += R;

    /* Update */
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    Eigen::MatrixXd K = Tc * S.inverse();

    // New mean and covariance
    Eigen::VectorXd z_diff = z - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
}