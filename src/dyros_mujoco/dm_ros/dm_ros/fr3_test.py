import mujoco
import mujoco.viewer as mj_view
import time
import rclpy
import os

def main():
    rclpy.init()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    xml_path = os.path.join(current_dir, '../robots', "fr3_w_hand.xml")

    # MuJoCo XML 생성 및 로드
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Viewer 실행
    try:
        with mj_view.launch_passive(model, data) as viewer:
            while rclpy.ok() and viewer.is_running():
                mujoco.mj_step(model, data)  # 시뮬레이션 실행
                viewer.sync()  # 화면 업데이트
                time.sleep(0.001)  # 1ms 대기

    except KeyboardInterrupt:
        print("\nSimulation interrupted. Closing viewer...")
        # mj_view.close()  # 시뮬레이터 창 닫기

    rclpy.shutdown()

if __name__=="__main__":
    main()