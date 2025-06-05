
#include "moveit_controller.hpp"

int main(int argc, char** argv)
{

    rclcpp::init(argc, argv);
    
    std::string arm_node = "moveit_arm_interface";
    std::string gripper_node = "moveit_gripper_interface";

    auto arm_action_server = std::make_shared<ArmController>(arm_node);
    auto gripper_action_server = std::make_shared<GripperController>(gripper_node);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(arm_action_server);
    executor.add_node(gripper_action_server);
    executor.spin();

    // rclcpp::spin({arm_action_server,gripper_action_server});

    
    rclcpp::shutdown();
    return 0;
}