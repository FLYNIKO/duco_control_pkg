import math
import time

class CylinderAutoPaint:
    def __init__(self, duco_cobot, init_pos, theta_deg, distance_to_cylinder, painting_width, vel, acc):
        self.duco_cobot = duco_cobot
        self.init_pos = init_pos
        self.theta_deg = theta_deg
        self.distance_to_cylinder = distance_to_cylinder
        self.painting_width = painting_width
        self.vel = vel
        self.acc = acc
        self.z_hi = 1.0 # 机械臂最高z坐标
        self.z_lo = 0.1 # 机械臂最低z坐标

    def get_cylinder_param(self):
        # TODO: 获取圆柱圆心坐标及半径，可根据实际需求修改
        cx, cy, cz = 1.8, 0.3, 1.2   # 圆心坐标
        cy_radius = 0.5               # 圆柱半径
        return cx, cy, cz, cy_radius

    def auto_paint(self):
        self.duco_cobot.switch_mode(1)
        cx, cy, cz, cy_radius = self.get_cylinder_param()
        z_val = self.z_hi
        radius = cy_radius + self.distance_to_cylinder
        theta_rad = math.radians(self.theta_deg)
        mid_angle = theta_rad / 2

        # 移动到中间初始位置
        pm = [cx - radius, cy, z_val, self.init_pos[3], self.init_pos[4], self.init_pos[5]]
        self.duco_cobot.movel(pm, self.vel, self.acc, 0, '', '', '', True)
        tcp_pos = self.duco_cobot.get_tcp_pose()
        print("pm: ", tcp_pos)

        pl_1 = [cx - radius * math.cos(mid_angle),
                cy + radius * math.sin(mid_angle),
                z_val,
                tcp_pos[3], tcp_pos[4], tcp_pos[5]]
        pl_2 = [cx - radius * math.cos(theta_rad),
                cy + radius * math.sin(theta_rad),
                z_val,
                tcp_pos[3], tcp_pos[4], tcp_pos[5]]

        # 左到右弧线
        self.duco_cobot.movec(pl_1, pl_2, self.vel, self.acc, 0, 2, '', '', '', True)
        tcp_pos = self.duco_cobot.get_tcp_pose()
        print("pl_2: ", tcp_pos)

        while z_val > self.z_lo:
            # 右下
            tcp_pos = self.duco_cobot.get_tcp_pose()
            pl_2 = [cx - radius * math.cos(theta_rad),
                    cy + radius * math.sin(theta_rad),
                    z_val,
                    tcp_pos[3], tcp_pos[4], tcp_pos[5]]
            pm = [cx - radius, cy, z_val, tcp_pos[3], tcp_pos[4], tcp_pos[5]]
            pr_2 = [cx - radius * math.cos(theta_rad),
                    cy - radius * math.sin(theta_rad),
                    z_val,
                    tcp_pos[3], tcp_pos[4], tcp_pos[5]]

            self.duco_cobot.movel(pl_2, self.vel, self.acc, 0, '', '', '', True)
            self.duco_cobot.movec(pm, pr_2, self.vel, self.acc, 0, 2, '', '', '', True)
            tcp_pos = self.duco_cobot.get_tcp_pose()
            print("pr_2: ", tcp_pos)

            z_val -= self.painting_width
            if z_val < self.z_lo:
                break

            # 左下
            tcp_pos = self.duco_cobot.get_tcp_pose()
            pl_2 = [cx - radius * math.cos(theta_rad),
                    cy + radius * math.sin(theta_rad),
                    z_val,
                    tcp_pos[3], tcp_pos[4], tcp_pos[5]]
            pm = [cx - radius, cy, z_val, tcp_pos[3], tcp_pos[4], tcp_pos[5]]
            pr_2 = [cx - radius * math.cos(theta_rad),
                    cy - radius * math.sin(theta_rad),
                    z_val,
                    tcp_pos[3], tcp_pos[4], tcp_pos[5]]

            self.duco_cobot.movel(pr_2, self.vel, self.acc, 0, '', '', '', True)
            self.duco_cobot.movec(pm, pl_2, self.vel, self.acc, 0, 2, '', '', '', True)

            z_val -= self.painting_width
            if z_val < self.z_lo:
                break

        time.sleep(0.5)
        self.duco_cobot.movel(self.init_pos, self.vel, self.acc, 0, '', '', '', True)