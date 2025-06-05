#ifndef MOVEIT_CONTROLLOER_H
#define MOVEIT_CONTROLLOER_H

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "control_msgs/action/gripper_command.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "Eigen/Dense"

using namespace std::placeholders;

class ArmController : public rclcpp::Node
{
public:
    using JT = control_msgs::action::FollowJointTrajectory;
    using GoalHandleJT = rclcpp_action::ServerGoalHandle<JT>;

    ArmController(std::string node_name);
    ~ArmController();

    void compute(const std::shared_ptr<GoalHandleJT> goal_handle);

private:
    sensor_msgs::msg::JointState joint_command_msg;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_command_pub;
    rclcpp_action::Server<JT>::SharedPtr as_;    
    std::vector<trajectory_msgs::msg::JointTrajectoryPoint> joint_trajectory_;

    rclcpp::Time goal_start_time;    
    float traj_time;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const JT::Goal> goal
    )
    {
        goal_start_time = rclcpp::Clock().now();
        RCLCPP_INFO(this->get_logger(), "Arm goal is received at %.4f", goal_start_time.seconds());

        feedback_header_stamp_ = 0;
        traj_time = goal->trajectory.points.back().time_from_start.sec;
        as_joint_size = goal->trajectory.points[0].positions.size();        
        (void) uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleJT> goal_handle
    )
    {
        RCLCPP_INFO(this->get_logger(), "Cancel arm request");
        (void) goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(
        const std::shared_ptr<GoalHandleJT> goal_handle
    )
    {
        std::thread{std::bind(&ArmController::compute, this, _1), goal_handle}.detach();
    }

    int feedback_header_stamp_;
    int as_joint_size;
};


class GripperController : public rclcpp::Node
{
public:

    using GC = control_msgs::action::GripperCommand;
    using GoalHandleGC = rclcpp_action::ServerGoalHandle<GC>;

    GripperController(std::string node_name);
    ~GripperController();

    void compute(const std::shared_ptr<GoalHandleGC> goal_handle);

private:
    sensor_msgs::msg::JointState joint_command_msg;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_command_pub;
    
    rclcpp_action::Server<GC>::SharedPtr as_;    
 
    rclcpp::Time goal_start_time;    
    
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID& uuid
    )
    {
        goal_start_time = rclcpp::Clock().now();
        RCLCPP_INFO(this->get_logger(), "Gripper Goal is received at %.4f", goal_start_time.seconds());
        
        feedback_header_stamp_ = 0;        
        
        (void) uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleGC> goal_handle
    )
    {
        RCLCPP_INFO(this->get_logger(), "Cancel gripper request");
        (void) goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(
        const std::shared_ptr<GoalHandleGC> goal_handle
    )
    {
        std::thread{std::bind(&GripperController::compute, this, _1), goal_handle}.detach();
    }

    int feedback_header_stamp_;    
};

#endif //MOVEIT_CONTROLLER_H