#pragma once

#include <fstream>
#include <Eigen/Dense>
#include <chrono>

#include "pinocchio/parsers/urdf.hpp"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/joint/joint-collection.hpp>

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/aba.hpp>


class ModelUpdater{
    
    public:
        static constexpr int kDof=7;

        ModelUpdater(){};    

        void initialize(const std::string urdf_path);
        // void getRobotState(); // get robot state information from Mujoco (ex: joint, angular velocity, ee pose ...)
        void updateModel(const Eigen::Ref<const Eigen::VectorXd> &q,
                         const Eigen::Ref<const Eigen::VectorXd> &qd,
                         const Eigen::Ref<const Eigen::VectorXd> &gw,
                         const Eigen::Ref<const Eigen::VectorXd> &tau); // update robot dynimics base on the current observation
        void updateKinematics();
        void updateDynamics();

        // void setTorque(const Eigen::Matrix<double, 7, 1> &torque_command);
        // void setPosition(const Eigen::Matrix<double, 7, 1> &position_command, bool idle_control = false);
        void setInitialValues();

        pinocchio::Model model_;
        pinocchio::Data data_;

        std::string base_frame_;
        std::string ee_frame_;
        int ee_frame_id_;        
        
        // fr3 parameters --
        Eigen::Matrix<double, 7, 7> M_; // joint mass_matrix_;
        Eigen::Matrix<double, 7, 7> M_inv_; // inverse of joint mass_matrix_;
        Eigen::Matrix<double, 6, 6> A_; // lambda_matrix_;
        Eigen::Matrix<double, 7, 1> NLE_; // non-linear effect (corriolis + gravity)
        Eigen::Matrix<double, 7, 7> C_; // coriolis_;
        Eigen::Matrix<double, 7, 1> G_; // gravity_;

        Eigen::Matrix<double, 7, 1> q_init_; //initial joint angle; 
        Eigen::Matrix<double, 7, 1> q_; // current joint angle
        Eigen::Matrix<double, 7, 1> qd_; // current angular velocity
        Eigen::Matrix<double, 7, 1> qdd_; // current angluar acceleration

        Eigen::Matrix<double, 7, 1> tau_measured_; // from torque sensor
        Eigen::Matrix<double, 7, 1> tau_d_; // command torque
        Eigen::Matrix<double, 7, 1> tau_ext_;

        Eigen::Matrix<double, 6, 1> f_ext_;
        Eigen::Matrix<double, 6, 1> f_ee_ext_; // external force w.r.t end-effector frame
        Eigen::Matrix<double, 6, 1> f_measured_;

        Eigen::Matrix<double, 6, 7> J_;     // jacobian
        Eigen::Matrix<double, 7, 6> J_bar_; // dynamically consistent inverse

        Eigen::Matrix<double, 7, 7> I_ = Eigen::Matrix<double, 7,7>::Identity();
        Eigen::Matrix<double, 7, 7> N_; /// null space projector

        Eigen::Isometry3d initial_transform_;   ///< initial transform for idle control
        Eigen::Isometry3d transform_;

        Eigen::Matrix<double, 3, 1> p_; // end-effector position
        Eigen::Matrix<double, 3, 3> r_; // roation matrix of end-effector

        Eigen::Matrix<double, 6, 1> xd_; // end-effector velocity
        Eigen::Matrix<double, 6, 1> xd_prev_;
        Eigen::Matrix<double, 6, 1> xd_lpf_;

        Eigen::Matrix<double, 2, 1> gw_init_; // gripper initial width
        Eigen::Matrix<double, 2, 1> gw_; // griiper width
        
};

    
