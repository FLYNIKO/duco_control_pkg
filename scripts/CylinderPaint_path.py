import math
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from DucoCobot import DucoCobot


class compute_pose:
    def __init__(self, ip):
        self.ip = ip
        self.vel = 0.3
        self.acc = 1.2
        self.init_pos = [0.41, 0.18, 1, -1.57, 0.0, -1.57]
        self.init_ajpos = [0.04, -0.59, 2.26, -1.70, -1.61, 0]
        self.stopheartthread = False
        self.spray_distance = 0.5 #喷涂距离
        self.paint_width = 0.2 #喷涂宽度
        self.hight_limit = 1.1 #机械臂最高z坐标
        self.lower_limit = 0.1 #机械臂最低z坐标
        self.paint_angle = 90 #喷涂角度
        self.duco_cobot = DucoCobot(ip, 7003)

    def get_cylinder_param(self):
        # TODO: 后期可从ROS topic获取
        cx, cy, cz = 1.8, 0.3, 1.8   # 圆心坐标
        cy_radius = 0.5              # 圆柱半径
        return cx, cy, cz, cy_radius

    def normalize(self,v):
        return v / np.linalg.norm(v)

    def robot_connect(self):
        rlt = self.duco_cobot.open()
        print("open:", rlt)
        rlt = self.duco_cobot.power_on(True)
        print("power_on:", rlt)
        rlt = self.duco_cobot.enable(True)
        print("enable:", rlt)
        self.duco_cobot.switch_mode(1)

    def compute_spray_pose(self, cx, cy, cz, cylinder_radius, spray_distance, theta_deg):
        # 合成半径
        r = cylinder_radius + spray_distance
        theta_rad = math.radians(theta_deg)

        # 坐标
        x = cx - r * math.cos(theta_rad)
        y = cy - r * math.sin(theta_rad)
        z = cz

        # 方向向量 Z：指向圆心
        normal = self.normalize(np.array([cx - x, cy - y, cz - z]))
        up = np.array([0, 0, 1])  # 世界坐标系的Z轴

        # 构造旋转基
        x_axis = self.normalize(np.cross(up, normal))   # 末端X轴（朝右）
        y_axis = np.cross(normal, x_axis)          # 末端Y轴
        rot_matrix = np.column_stack((x_axis, y_axis, normal))  # 列向量方式构造旋转矩阵

        # 转为 RPY
        rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)

        return [x, y, z, rpy[0], rpy[1], rpy[2]]
    
    def generate_spray_path(self, cx, cy, cz, radius, distance, start_angle, end_angle, step):
        path = []
        start_angle = int(start_angle)
        end_angle = int(end_angle)
        step = int(step)
        for angle in range(start_angle, end_angle, step):
            pose = self.compute_spray_pose(cx, cy, cz, radius, distance, angle)
            path.append((angle, pose))  
        return path

    def run(self):
        cx, cy, cz, cy_radius = self.get_cylinder_param()  # 获取圆柱 半径 和 圆心相对于机械臂底座坐标
        cz = self.hight_limit
        self.robot_connect()
        # 初始位置
        self.duco_cobot.movej_pose2(self.init_pos, self.vel, self.acc, 1, '', '', '', True)
        # 喷涂左侧初始位置
        pose = self.compute_spray_pose(cx, cy, cz, cy_radius, self.spray_distance, -(self.paint_angle/2))
        self.duco_cobot.movej_pose2([self.init_ajpos[0], pose[1], pose[2], self.init_ajpos[3], self.init_ajpos[4], self.init_ajpos[5]], self.vel, self.acc, 1, '', '', '', True)
        self.duco_cobot.movej_pose2(pose, self.vel, self.acc, 1, '', '', '', True)
        print("移动到喷涂左侧初始位置")

        while 1:
            # for angle in range(-45, 46, 2):
            #     pose = self.compute_spray_pose(cx, cy, cz, cylinder_radius, spray_distance, angle)
            #     self.duco_cobot.movel(pose, self.vel, self.acc, 1, '', '', '', True)
            #     print(f"{angle}°点位: {pose}")

            # 从左到右移动
            path = self.generate_spray_path(cx, cy, cz, cy_radius, self.spray_distance, -(self.paint_angle/2), self.paint_angle/2 + 1, 2)
            for angle, pose in path:
                self.duco_cobot.movel(pose, self.vel, self.acc, 1, '', '', '', True)
            #     print(f"{angle}° 点位: {pose}")
            # 到右侧后下降
            if cz > self.lower_limit:
                cz -= self.paint_width
            else:
                break
            # 从右到左移动
            path = self.generate_spray_path(cx, cy, cz, cy_radius, self.spray_distance, self.paint_angle/2, -(self.paint_angle/2 + 1), -2)
            for angle, pose in path:
                self.duco_cobot.movel(pose, self.vel, self.acc, 1, '', '', '', True)
            #     print(f"{angle}° 点位: {pose}")
            # 到左侧后下降
            if cz > self.lower_limit:
                cz -= self.paint_width
            else:
                break
        
        # 结束后回到初始位置
        self.duco_cobot.movej2(self.init_ajpos, self.vel*2, self.acc * 1.5, 0, True)

    def main(self):
        self.run()
    
if __name__ == '__main__':
    test = compute_pose("192.168.100.10")
    test.main()
