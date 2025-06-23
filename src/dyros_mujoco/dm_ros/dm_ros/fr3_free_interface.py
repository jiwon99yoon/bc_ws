import rclpy
import os
from prc import Fr3Controller
from .utils.multi_thread_interface import MujocoROSBridge

def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    rclpy.init()
   
    xml_path = os.path.join(current_dir, '../robots', "fr3_w_hand.xml")
    urdf_path = os.path.join(current_dir, '../robots', 'fr3/fr3_hand.urdf')

    rc = Fr3Controller(urdf_path) 

    robot_info = [xml_path, urdf_path, 1000]
    camera_info = ['hand_eye', 320, 240, 30]

    bridge = MujocoROSBridge(robot_info, camera_info, rc)

    # time.sleep(2.0)
    bridge.run()
    
    bridge.destroy_node()
    rclpy.shutdown()


    
if __name__=="__main__":    
    main()
