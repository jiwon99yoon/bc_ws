#include "moveit_controller.hpp"

static Eigen::Vector3d QuinticSpline(
                   double time,       ///< Current time
                   double time_0,     ///< Start time
                   double time_f,     ///< End time
                   double x_0,        ///< Start state
                   double x_dot_0,    ///< Start state dot
                   double x_ddot_0,   ///< Start state ddot
                   double x_f,        ///< End state
                   double x_dot_f,    ///< End state
                   double x_ddot_f )  ///< End state ddot
{
  double a1,a2,a3,a4,a5,a6;
  double time_s;

  Eigen::Vector3d result;

  if(time < time_0)
  {
    result << x_0, x_dot_0, x_ddot_0;
    return result;
  }
  else if (time > time_f)
  {
    result << x_f, x_dot_f, x_ddot_f;
    return result;
  }


  time_s = time_f - time_0;
  a1=x_0;
  a2=x_dot_0;
  a3=x_ddot_0/2.0;

  Eigen::Matrix3d Temp;
  Temp<<pow(time_s, 3), pow(time_s, 4), pow(time_s, 5),
        3.0 * pow(time_s, 2), 4.0 * pow(time_s, 3), 5.0 * pow(time_s, 4),
        6.0 * time_s, 12.0 * pow(time_s, 2), 20.0 * pow(time_s, 3);

  Eigen::Vector3d R_temp;
  R_temp<<x_f-x_0-x_dot_0*time_s-x_ddot_0*pow(time_s,2)/2.0,
        x_dot_f-x_dot_0-x_ddot_0*time_s,
        x_ddot_f-x_ddot_0;

  Eigen::Vector3d RES;

  RES = Temp.inverse()*R_temp;

  a4=RES(0);
  a5=RES(1);
  a6=RES(2);

  double time_fs = time - time_0;

  double position = a1+a2*pow(time_fs,1)+a3*pow(time_fs,2)+a4*pow(time_fs,3)+a5*pow(time_fs,4)+a6*pow(time_fs,5);
  double velocity = a2+2.0*a3*pow(time_fs,1)+3.0*a4*pow(time_fs,2)+4.0*a5*pow(time_fs,3)+5.0*a6*pow(time_fs,4);
  double acceleration =2.0*a3+6.0*a4*pow(time_fs,1)+12.0*a5*pow(time_fs,2)+20.0*a6*pow(time_fs,3);


  result<<position,velocity,acceleration;

  return result;
}

ArmController::ArmController(std::string node_name):Node(node_name)
{
    this->as_ = rclcpp_action::create_server<JT>(
        this,
        "panda_arm_controller/follow_joint_trajectory",
        std::bind(&ArmController::handle_goal, this, _1, _2),
        std::bind(&ArmController::handle_cancel, this, _1),
        std::bind(&ArmController::handle_accepted, this, _1)
    );

    joint_command_pub = this->create_publisher<sensor_msgs::msg::JointState>("/panda/joint_set", 10);
    joint_command_msg.name.resize(7);
    joint_command_msg.position.resize(7);
    joint_command_msg.header.frame_id="panda_link0";
    
    for(int i=0;i<7;i++){
        joint_command_msg.name[i] = "panda_joint" +std::to_string(i+1);
    }

}

ArmController::~ArmController(){};


void ArmController::compute(const std::shared_ptr<GoalHandleJT> goal_handle)
{
    RCLCPP_INFO(this->get_logger(), "Executing arm goal");
    const auto goal = goal_handle->get_goal();

    rclcpp::Time currentTime;
    double passedTime; // float -> double로 변경
    auto feedback = std::make_shared<JT::Feedback>();
    auto result = std::make_shared<JT::Result>();

    feedback->joint_names.resize(as_joint_size);
    feedback->actual.positions.resize(as_joint_size);
    feedback->actual.velocities.resize(as_joint_size);
    feedback->actual.accelerations.resize(as_joint_size);
    rclcpp::Rate loop_rate(100); // 100Hz # 100hz이상 올라가가지 않게 제어 - mujoco 렉
    while(rclcpp::ok())
    {
        if(!goal_handle->is_active())
            return;
        
        currentTime = rclcpp::Clock().now();
        passedTime = (currentTime - goal_start_time).seconds();
        
        //If, current time is between start time of trajectory and end time of trajectory
        if((currentTime >=goal_start_time)&&(passedTime<traj_time))
        { 
            int point_size = static_cast<int>(goal->trajectory.points.size());
            joint_command_msg.header.stamp=currentTime;

            Eigen::Vector3d position_now;            
            for(int j=0;j<as_joint_size;j++){ // j = joint number
                
                for(int i=0;i<point_size-1;i++){
                    // 초단위 update to mujoco
                    //  if((passedTime>=goal->trajectory.points[i].time_from_start.sec)&&(passedTime<goal->trajectory.points[i+1].time_from_start.sec))
                    //  {
                    //      position_now=QuinticSpline(passedTime,goal->trajectory.points[i].time_from_start.sec, goal->trajectory.points[i+1].time_from_start.sec,
                    //      goal->trajectory.points[i].positions[j],goal->trajectory.points[i].velocities[j],goal->trajectory.points[i].accelerations[j],
                    //      goal->trajectory.points[i+1].positions[j],goal->trajectory.points[i+1].velocities[j],goal->trajectory.points[i+1].accelerations[j]);
                    //  }            
                    // trajectory i 의 time_from_start (sec, nanosec)
                    const auto &pt_i  = goal->trajectory.points[i].time_from_start;
                    const auto &pt_i1 = goal->trajectory.points[i+1].time_from_start;

                    // “double t_i = sec_i + nanosec_i * 1e-9”
                    double t_i  = static_cast<double>(pt_i.sec)
                                 + static_cast<double>(pt_i.nanosec) * 1e-9;
                    double t_i1 = static_cast<double>(pt_i1.sec)
                                 + static_cast<double>(pt_i1.nanosec) * 1e-9;
                    // passedTime(이미 소수점 이하 포함) 과 비교
                    if (passedTime >= t_i && passedTime < t_i1) {
                        // trajectory i, i+1 의 위치·속도·가속도 할당
                        double pos_i   = goal->trajectory.points[i]  .positions[j];
                        double vel_i   = goal->trajectory.points[i]  .velocities[j];
                        double acc_i   = goal->trajectory.points[i]  .accelerations[j];
                        double pos_i1  = goal->trajectory.points[i+1].positions[j];
                        double vel_i1  = goal->trajectory.points[i+1].velocities[j];
                        double acc_i1  = goal->trajectory.points[i+1].accelerations[j];

                        position_now = QuinticSpline(passedTime,  // 현재 시간(초 실수, nanosec 포함)
                            t_i,         // i 구간 시작 시간
                            t_i1,        // i+1 구간 끝 시간
                            pos_i, vel_i, acc_i,        // pos_i, vel_i, acc_i 는 i 구간의 위치·속도·가속도
                            pos_i1, vel_i1, acc_i1      // pos_i1, vel_i1, acc_i1 는 i+1 구간의 위치·속도·가속도
                        );
                        break; 
                    }                      
                }
                feedback->joint_names = goal->trajectory.joint_names;                
                feedback->actual.positions[j] = position_now(0);                
                feedback->actual.velocities[j] = position_now(1);
                feedback->actual.accelerations[j] = position_now(2);

                joint_command_msg.position[j]=position_now(0);
            }            
            joint_command_pub->publish(joint_command_msg); //기존 for loop에서 publish를 하지 않고, 계산된 결과 pub
            feedback -> actual.time_from_start= currentTime - goal_start_time;
            feedback -> header.stamp.sec = feedback_header_stamp_;            
            feedback_header_stamp_++;    
            goal_handle->publish_feedback(feedback);
        }

        if(passedTime > traj_time + 0.5 )
        {
            // as_.setSucceeded();
            result->error_code = 0; // 0 : SUCCESSFUL
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Arm goal succeeded");
        }
        loop_rate.sleep(); //100hz 이상 올라가지않게 제어
    }
}


GripperController::GripperController(std::string node_name):Node(node_name)
{
    this->as_ = rclcpp_action::create_server<GC>(
        this,
        "panda_gripper_controller/gripper_cmd",
        std::bind(&GripperController::handle_goal, this, _1),
        std::bind(&GripperController::handle_cancel, this, _1),
        std::bind(&GripperController::handle_accepted, this, _1)
    );

    joint_command_pub = this->create_publisher<sensor_msgs::msg::JointState>("/panda/joint_set", 10);
    
    joint_command_msg.name.resize(2);
    joint_command_msg.position.resize(2);

    joint_command_msg.header.frame_id="panda_link0";
    joint_command_msg.name[0]="panda_finger_joint1";
    joint_command_msg.name[1]="panda_finger_joint2";
}

GripperController::~GripperController(){};


void GripperController::compute(const std::shared_ptr<GoalHandleGC> goal_handle)
{
    RCLCPP_INFO(this->get_logger(), "Executing gripper goal");
    //rclcpp::Rate loop_rate(1);
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<GC::Result>();

    rclcpp::Time currentTime;
    float passedTime;
    rclcpp::Rate loop_rate(100); // 100Hz # 100hz이상 올라가가지 않게 제어
    while(rclcpp::ok())
    {
        if(!goal_handle->is_active())
            return;
        
        currentTime = rclcpp::Clock().now();
        passedTime = (currentTime - goal_start_time).seconds();
        
        //If, current time is between start time of trajectory and end time of trajectory
        if((passedTime <= 1.0 ))
        { 
            joint_command_msg.header.stamp=goal_start_time;
            joint_command_msg.position[0] = goal->command.position;
            joint_command_msg.position[1] = goal->command.position;
            joint_command_pub -> publish(joint_command_msg);
        }
        else{

            result->reached_goal = 1; // 1 : SUCCESSFUL
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
        }
        loop_rate.sleep(); //100hz 이상 올라가지않게 제어
    }
}