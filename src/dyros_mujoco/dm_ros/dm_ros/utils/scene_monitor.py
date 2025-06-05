from rclpy.node import Node
import mujoco
from scipy.spatial.transform import Rotation as R

class SceneMonitor(Node):
    def __init__(self, model, data):
        super().__init__("scene_monitor")
        self.model = model # mujoco model
        self.data = data  # mujoco data

        self.object_types = [
            mujoco.mjtObj.mjOBJ_UNKNOWN,       
            mujoco.mjtObj.mjOBJ_BODY,                     
            mujoco.mjtObj.mjOBJ_XBODY,                    
            mujoco.mjtObj.mjOBJ_JOINT,                    
            mujoco.mjtObj.mjOBJ_DOF,                      
            mujoco.mjtObj.mjOBJ_GEOM,                     
            mujoco.mjtObj.mjOBJ_SITE,                     
            mujoco.mjtObj.mjOBJ_CAMERA,                   
            mujoco.mjtObj.mjOBJ_LIGHT,                    
            mujoco.mjtObj.mjOBJ_FLEX,                     
            mujoco.mjtObj.mjOBJ_MESH,                     
            mujoco.mjtObj.mjOBJ_SKIN,                     
            mujoco.mjtObj.mjOBJ_HFIELD,                   
            mujoco.mjtObj.mjOBJ_TEXTURE,                  
            mujoco.mjtObj.mjOBJ_MATERIAL,                   
            mujoco.mjtObj.mjOBJ_PAIR,                        
            mujoco.mjtObj.mjOBJ_EXCLUDE,                    
            mujoco.mjtObj.mjOBJ_EQUALITY,                  
            mujoco.mjtObj.mjOBJ_TENDON,                   
            mujoco.mjtObj.mjOBJ_ACTUATOR,                 
            mujoco.mjtObj.mjOBJ_SENSOR,
            mujoco.mjtObj.mjOBJ_NUMERIC,                  
            mujoco.mjtObj.mjOBJ_TEXT,                     
            mujoco.mjtObj.mjOBJ_TUPLE,                    
            mujoco.mjtObj.mjOBJ_KEY,                      
            mujoco.mjtObj.mjOBJ_PLUGIN,  
        ]

        self.get_logger().info("Scene Monitor is ready")

    def getObjectInfo(self, obj_id):
        position = self.data.xpos[obj_id]
        quaternion = self.data.xquat[obj_id]  # [w, x, y, z]
        return position, quaternion

    def getAllObject(self):
        for obj_type in self.object_types:
            print(f"\n{obj_type.name} Objects:")
            for obj_id in range(self.model.nbody if obj_type == mujoco.mjtObj.mjOBJ_BODY else self.model.ngeom):
                name = mujoco.mj_id2name(self.model, obj_type, obj_id)
                if name:
                    print(f"  - {name}({obj_id})")

    def getSensor(self):
        for i in range(self.model.nsensor):  # 모델에 포함된 센서 개수만큼 반복
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)     
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)  # 센서 ID 가져오기
            print(f"Sensor Name: {name}, Sensor ID: {sensor_id}")
