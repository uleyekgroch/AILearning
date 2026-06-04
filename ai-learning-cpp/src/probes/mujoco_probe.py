"""
MuJoCo 外接探针 (Embodied Probe) — 连接 AILearning C++ 核心与物理世界
通过 /api/perceive 接口将 MuJoCo 传感器数据（视觉、关节角度、触觉）输入到 AI 系统，
并通过 /api/autonomous 接收反馈或主动探索动作，实现真正的“具身认知”。
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import requests
import json
import threading
import sys

API_URL = "http://localhost:8080/api/perceive"
CHAT_URL = "http://localhost:8080/api/chat"

# 我们构建一个简单的 XML 模型：一个可以自由运动的机械臂(手臂)去触碰一个苹果(红色球体)
xml = """
<mujoco>
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="32"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom type="plane" size="2 2 .1" rgba=".9 .9 .9 1"/>
    
    <!-- Apple (红苹果) -->
    <body name="apple" pos="0.5 0.5 0.1">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1" mass="0.1"/>
    </body>

    <!-- Simple Arm -->
    <body name="arm_base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.1" rgba="0.2 0.2 0.2 1"/>
      <body name="arm_link1" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0  0 0 0.4" size="0.04" rgba="0.8 0.6 0.4 1"/>
        <body name="arm_link2" pos="0 0 0.4">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0  0 0 0.4" size="0.03" rgba="0.8 0.6 0.4 1"/>
          <!-- Hand with tactile sensor -->
          <body name="hand" pos="0 0 0.4">
            <geom name="hand_geom" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/>
            <site name="touch_site" pos="0 0 0.05" type="sphere" size="0.05" rgba="0 0 1 0.5"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor joint="shoulder" ctrlrange="-1 1" gear="10"/>
    <motor joint="elbow" ctrlrange="-1 1" gear="5"/>
  </actuator>
  
  <sensor>
    <touch site="touch_site" name="hand_touch"/>
    <jointpos joint="shoulder" name="shoulder_pos"/>
    <jointpos joint="elbow" name="elbow_pos"/>
  </sensor>
</mujoco>
"""

class EmbodiedProbe:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.running = True
        self.last_sync_time = time.time()
        
    def get_sensory_state(self):
        """采集多模态传感器数据"""
        # 1. 本体感觉 (Proprioception: 关节角度)
        shoulder_pos = self.data.sensor("shoulder_pos").data[0]
        elbow_pos = self.data.sensor("elbow_pos").data[0]
        
        # 2. 触觉 (Haptic: 接触力)
        touch_force = self.data.sensor("hand_touch").data[0]
        
        # 3. 视觉特征提取 (此处我们做一个简化的"目标对象中心位置"作为视觉潜向量)
        apple_pos = self.data.body("apple").xpos
        hand_pos = self.data.body("hand").xpos
        distance = np.linalg.norm(apple_pos - hand_pos)
        
        # 编码为 AILearning 所需的 raw_input (将各模态打平为 float 数组)
        # 注意：真实的系统这里会传图片像素给 /api/perceive，但由于我们目前用的是Stub，我们直接传特征向量
        sensory_vector = [
            float(shoulder_pos), float(elbow_pos), # 关节状态
            float(touch_force),                    # 触觉
            float(apple_pos[0]), float(apple_pos[1]), float(apple_pos[2]), # 视觉空间定位
            float(distance)                        # 相对距离感知
        ]
        
        # 补充到 128 维 (由于目前 C++ 的 input_dim 默认配置是 128)
        padding = [0.0] * (128 - len(sensory_vector))
        return sensory_vector + padding

    def send_to_brain(self, sensory_vector):
        """将感知发送给 AILearning 核心大脑"""
        try:
            # 1. 感知
            resp = requests.post(API_URL, json={"raw_input": {"visual": sensory_vector}}, timeout=0.1)
            # 在一个成熟的系统中，这里应该是调用强化学习的 step/learn_from_experience
            # 但当前我们只是展示信号通路的连通性
        except Exception:
            pass

    def run(self):
        print("启动 MuJoCo 仿真物理世界探针...")
        print("尝试向 C++ 核心投喂具身数据...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step = 0
            while viewer.is_running() and self.running:
                step_start = time.time()

                # 给一点微小的随机扰动/控制，模拟探索
                self.data.ctrl[0] = np.sin(step * 0.01) * 0.5 + np.random.normal(0, 0.1)
                self.data.ctrl[1] = np.cos(step * 0.02) * 0.5 + np.random.normal(0, 0.1)
                
                # 物理引擎步进
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 每 100ms 将世界感知投喂给大脑
                if time.time() - self.last_sync_time > 0.1:
                    state = self.get_sensory_state()
                    self.send_to_brain(state)
                    self.last_sync_time = time.time()

                # 保持稳定的渲染帧率 (60fps)
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                step += 1

if __name__ == "__main__":
    probe = EmbodiedProbe()
    try:
        probe.run()
    except KeyboardInterrupt:
        probe.running = False
        print("探针停止。")
