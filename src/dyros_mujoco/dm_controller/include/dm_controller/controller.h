#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <unordered_map>

#include "robot_model/robot_updater.h"
#include "utils/dyros_math.h"

using namespace dyros_math;

namespace DmController{
    
    class Controller{
        public:
        Controller(const std::string urdf_path);

            void initialize(const std::string urdf_path);

            void updateModel(const Eigen::Ref<const Eigen::VectorXd> &q,
                             const Eigen::Ref<const Eigen::VectorXd> &qd,
                             const Eigen::Ref<const Eigen::VectorXd> &tau,
                             const Eigen::Ref<const Eigen::VectorXd> &ft,
                             const double t); // update kinematics and dynamics

            Eigen::VectorXd setIdleConfig(const Eigen::Ref<const Eigen::Vector2d> &time);
            Eigen::VectorXd gripperOpen(const double target_width, const Eigen::Ref<const Eigen::Vector2d> &time);
            Eigen::VectorXd gripperClose();

            Eigen::VectorXd taskMove(const Eigen::Ref<const Eigen::Vector3d> &x_goal, // goal task position
                                     const Eigen::Ref<const Eigen::MatrixXd> &r_goal, // goal task orientation
                                     const Eigen::Ref<const Eigen::Vector2d> &time);  

            Eigen::VectorXd jointMove(const Eigen::Ref<const Eigen::VectorXd> &goal,  // goal joint position
                                      const Eigen::Ref<const Eigen::Vector2d> &time); // time information [t_0, t_f]

            // whether the given task is completed or not     
            Eigen::VectorXd idleState();
            Eigen::VectorXd initState();
            
            void updateInitialValues();
            

        private:
            ModelUpdater model_;
            std::string urdf_path_;
            
            Eigen::Vector7d q_idle_, q_init_; 
            Eigen::Vector7d q_desired_;
            Eigen::Vector7d q_, qd_; // current joint and velocity
            Eigen::Vector7d tau_d_; // robot torque command
            Eigen::Vector7d tau_d_lpf_; // prevent large transition torque
            Eigen::Vector7d tau_measured_; // from joint torque sensor
            Eigen::Vector7d tau_ext_; // external torque (friction + contact)
            
            Eigen::Vector3d x_; // end_effector position w.r.t base
            Eigen::Vector3d x_ee_; // end_effector position w.r.t ee frame
            Eigen::Vector6d v_, v_lpf_; // end_effector velocity w.r.t base
            Eigen::Vector6d v_ee_, v_ee_lpf_;
            Eigen::Matrix3d R_; // end_effector rotation w.r.t base
            Eigen::Matrix3d R_ee_;
            Eigen::Vector6d f_ext_init_, f_ext_, f_ext_lpf_;
            Eigen::Vector6d f_contact_;
            Eigen::Vector6d f_bias_;
            Eigen::Vector6d pose_d_;

            Eigen::Vector2d gw_; //grasp width
            Eigen::Vector2d gw_idle_, gw_init_;       
            Eigen::Vector2d gw_desired_;
            Eigen::Vector2d gv_; //grasp velocity 
            Eigen::Vector2d gf_; //grasp force

            Eigen::Vector8d ctrl_;
            double t_;
            double t_init_;

            int dof_ = 7;
            double hz_ = 1000.0;

            bool set_init_;

    };

}
