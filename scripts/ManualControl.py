import time
import rospy
import math
import threading

from DucoCobot import DucoCobot
from key_input_pkg.msg import KeyInput
from duco_control_pkg.msg import LineDetectionArray, LineInfo, ObstacleFlags
from CylinderPaint_duco import CylinderAutoPaint
from collections import deque
from config import *
from triple_radar_front_distance import DirectionalLaser

class KeyInputStruct:
    def __init__(self, x0=0, x1=0, y0=0, y1=0, z0=0, z1=0,
                 init=0, serv=0, multi=0, start=0,
                 rx0=0, rx1=0, ry0=0, ry1=0, rz0=0, rz1=0,
                 clog=0, find=0, high=0, center=0, low=0,
                 top=0, bottom=0):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.init = init
        self.serv = serv
        self.multi = multi
        self.start = start
        self.rx0 = rx0
        self.rx1 = rx1
        self.ry0 = ry0
        self.ry1 = ry1
        self.rz0 = rz0
        self.rz1 = rz1
        self.clog = clog
        self.find = find
        self.high = high
        self.center = center
        self.low = low
        self.top = top
        self.bottom = bottom
        # TODO: 添加更多按键位的解析

class SimplePID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target, current, dt):
        error = current - target
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class system_control:
    def __init__(self, duco_cobot, app=None):
        self.ip = IP
        self.duco_cobot = duco_cobot
        self.app = app
        self.auto_vel = AUTOSPEED # 自动喷涂速度
        self.vel = DEFAULT_VEL # 机械臂手动末端速度
        self.ob_vel = OB_VELOCITY # 障碍物避障速度
        self.ob_acc = OB_ACC # 障碍物避障加速度
        self.acc = DEFAULT_ACC # 机械臂末端加速度
        self.aj_pos = [] # 当前关节角度
        self.tcp_pos = [] # 当前末端位姿
        self.sysrun = True
        self.autopaint_flag = False
        self.clog_flag = False
        self.H_find_flag = False
        self.position_flag = False
        self.find_mode = False

        self.ob_status = 1 # 避障模式状态，0-全向，1-仅提示，2-自动sync, 3-自动interval
        self.paint_motion = 0 # 喷涂位置点，0-其他位置，1-喷涂顶部，2-喷涂下翼板，3-喷涂中腹板，4-喷涂上翼板，5-喷涂底部
        self.init_pos = INIT_POS # 初始位置
        self.serv_pos = SERV_POS # 维修位置
        self.safe_pos = SAFE_POS # 安全位置
        self.clog_pos = CLOG_POS # 堵枪位置
        self.pid = SimplePID(kp=1, ki=0.0, kd=0.2)
        self.pid_z = SimplePID(kp=KP, ki=KI, kd=KD)
        self.dist_history = deque(maxlen=5) # 滤波队列


        self.scan_range = 0.7
        self.step_size = 0.02
        self.min_jump_threshold = 0.05
        self.center_z = 0.0
        self.center_x = 0.0
        self.painting_deg_surface = 70
        self.painting_deg_flange = 60
        self.web_top_point = [] # 钢梁顶部点
        self.paint_center = [] # 喷涂中心点
        self.paint_top = [] # 喷涂顶部
        self.paint_bottom = [] # 喷涂底部
        self.paint_high = [] # 喷涂上姿态
        self.paint_low = [] # 喷涂下姿态


        self.web_height = 0 # 钢梁高度
        self.flange_up_width = 0 # 上翼子板宽度
        self.flange_down_width = 0 # 下翼子板宽度

        self.theta_deg = PAINTDEG / 2  # 喷涂角度的一半
        self.distance_to_cylinder = 0  # 末端与圆柱表面距离
        self.painting_width = PAINTWIDTH  # 喷涂宽度

        # 目标线检测相关变量
        self.target_dist_in_surface = 0.0
        self.target_dist_in_flange = 0.0
        self.painting_dist = 0 # 喷涂距离
        self.target_dist_in_web = self.painting_dist
        self.surface_painting_dist = 0 # 钢梁表面喷涂距离
        self.flange_painting_dist = 0 # 翼子板喷涂距离
        self.flange_painting_dist_adjust = 0.2 # 翼子板喷涂距离调整
        self.surface_distance = 0.0
        self.surface_angle_deg = 0.0

        # 障碍物检测相关变量
        self.left_front_obstacle = False
        self.left_mid_obstacle = False
        self.left_rear_obstacle = False
        self.right_front_obstacle = False
        self.right_mid_obstacle = False
        self.right_rear_obstacle = False
        self.center_obstacle = False
        self.up_obstacle = False
        self.down_obstacle = False
        self.obstacle_stamp = None
        self.safe_distance = 0.0
        
        # 障碍物检测过滤队列（连续3帧检测到才认为是真的障碍物）
        self.obstacle_filter_frames = 1  # 需要连续检测到的帧数
        self.left_front_history = deque(maxlen=self.obstacle_filter_frames)
        self.left_mid_history = deque(maxlen=self.obstacle_filter_frames)
        self.left_rear_history = deque(maxlen=self.obstacle_filter_frames)
        self.right_front_history = deque(maxlen=self.obstacle_filter_frames)
        self.right_mid_history = deque(maxlen=self.obstacle_filter_frames)
        self.right_rear_history = deque(maxlen=self.obstacle_filter_frames)
        self.center_history = deque(maxlen=self.obstacle_filter_frames)
        self.up_history = deque(maxlen=self.obstacle_filter_frames)
        self.down_history = deque(maxlen=self.obstacle_filter_frames)

        # H检测历史信息存储（用于3次取平均）
        self.H_detection_history = deque(maxlen=3)  # 存储最近3次的有效H检测信息
        self.H_processing_ready = False  # 标记是否已经收集到3次有效信息

        self.latest_keys = [0] * 20
        self.last_key_time = time.time()
        self.last_line_time = time.time()
        self.last_H_time = time.time()
        self.last_ob_time = time.time()
        self.keys_subscriber = rospy.Subscriber('/key_input', KeyInput, self._keys_callback)
        self.H_info_subscriber = rospy.Subscriber('/left_radar/H_detection_info', LineDetectionArray, self._H_info_callback)
        self.line_detection_subscriber = rospy.Subscriber('/twin_radar/line_detection_info', LineDetectionArray, self._line_detection_callback)
        self.obstacle_flags_subscriber = rospy.Subscriber('/obstacle_flags', ObstacleFlags, self._obstacle_flags_callback)

        self.emergency_stop_flag = False
        self.emergency_thread = threading.Thread(target=self.emergency_stop_thread, daemon=True)
        self.emergency_thread.start()
        self.ob_flag = False # 障碍物检测标志
        self.ob_thread = threading.Thread(target=self.ob_thread, daemon=True)
        self.ob_thread.start()
                
        # 初始化方向性激光雷达读取器
        self.directional_laser = DirectionalLaser('/right_radar/filtered_scan')
        
    # 急停线程
    def emergency_stop_thread(self):
        self._stop_event = threading.Event()
        self.duco_stop = DucoCobot(self.ip, PORT)
        self.duco_stop.open()
        while self.sysrun and not rospy.is_shutdown():
            self.emergency_stop_flag = False
            key_input = self.get_key_input()
            state = self.duco_stop.get_robot_state()
            if key_input.multi or self.emergency_stop_flag:
                rospy.logwarn("| 检测到紧急停止按键，正在执行紧急停止！ |")
                self.ob_status = 1
                self.emergency_stop_flag = True
                self.autopaint_flag = False
                self.find_mode = False
                self.duco_stop.stop(True)                
                if state[0] != 6:
                    rospy.loginfo("机械臂已重置")
                    if state[0] == 5:
                        self.duco_stop.enable(True)
                    if state[0] == 4:
                        self.duco_stop.power_on(True)
                        self.duco_stop.enable(True)
                    self.duco_stop.switch_mode(1)          
            self._stop_event.wait(0.05)

    def ob_safe_pos(self):
        # 1. 记录当前点位
        rospy.logwarn("| 检测到侧障碍物，回收机械臂到安全位置 |")
        original_pos = self.duco_ob.get_tcp_pose()
        # 2. 先移动到安全位置
        self.duco_ob.servoj_pose(self.safe_pos, self.ob_vel, self.acc, '', '', '', True)
        # 3. 等待 1 秒
        rospy.sleep(1)
        # 4. 最多尝试 3 次恢复
        max_attempts = 3
        attempt = 0
        move = False
        while attempt < max_attempts:
            if move:
                break
            attempt += 1
            rospy.loginfo(f"尝试恢复到原点位，第 {attempt} 次")
            # 开始移动到 original_pos
            self.duco_ob.servoj_pose(original_pos, self.vel, self.acc, '', '', '', False)
            # 在移动过程中持续判断
            while self.ob_flag:
                current_pos = self.duco_ob.get_tcp_pose()
                if current_pos[0] - original_pos[0] < 0.01 and current_pos[1] - original_pos[1] < 0.01 and current_pos[2] - original_pos[2] < 0.01:
                    rospy.loginfo("成功回到原点位")
                    move = True
                    break  # 成功恢复，直接结束逻辑
                elif self.is_obstacle_detected():
                    self.duco_ob.stop(True)
                    rospy.logwarn("| 检测到障碍，暂停 0.5s |")
                    rospy.sleep(0.5)
                    break  # 跳出本轮 while，进行下一次尝试
            # 如果循环结束还没成功，会进入下一轮 attempt
        rospy.logwarn("| 超过最大尝试次数，请等待通过障碍后手动恢复 |")

    def ob_thread(self):
        self._ob_event = threading.Event()
        self.duco_ob = DucoCobot(self.ip, PORT)
        self.duco_ob.open()
        rospy.loginfo("ob_start")
        while self.sysrun and not rospy.is_shutdown():
            if time.time() - self.last_ob_time > TIMEOUT:
                rospy.logwarn("| 障碍物检测超时，请检查传感器连接，此时无法避障！ |")
                rospy.sleep(1)
                continue
            else:
                ob_data = self.get_obstacle_status()
                tcp_pos = self.duco_ob.get_tcp_pose()
                if self.is_obstacle_detected():
                    
                    if self.ob_status == 0: # 全向避障
                        self.ob_flag = True
                        self.duco_ob.stop(True)
                        if (ob_data['left_mid'] and ob_data['left_rear']) or (ob_data['right_mid'] and ob_data['right_rear']):
                            self.ob_safe_pos()
                        elif ob_data['left_mid'] or ob_data['right_mid'] or ob_data['center']:
                            # self.duco_ob.servoj_pose([tcp_pos[0] + 0.2, tcp_pos[1], tcp_pos[2], self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            self.duco_ob.movel([tcp_pos[0] + 0.2, tcp_pos[1], tcp_pos[2], self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, 0, '', '', '', False)

                            rospy.logwarn("| 检测到障碍物，向后躲避！ |")
                        elif ob_data['up']:
                           #  self.duco_ob.servoj_pose([tcp_pos[0], tcp_pos[1], tcp_pos[2] - 0.1, self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            self.duco_ob.movel([tcp_pos[0], tcp_pos[1], tcp_pos[2] - 0.1, self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, 0, '', '', '', False)

                            rospy.logwarn("| 检测到上方障碍物障碍物，向下躲避！ |")
                        elif ob_data['down']:
                            # self.duco_ob.servoj_pose([tcp_pos[0], tcp_pos[1], tcp_pos[2] + 0.1, self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            self.duco_ob.movel([tcp_pos[0], tcp_pos[1], tcp_pos[2] + 0.1, self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, 0, '', '', '', False)

                            rospy.logwarn("| 检测到下方障碍物，向上躲避！ |")

                    elif self.ob_status == 1: # 无避障
                        self.ob_flag = True
                        obstacle_keys = ['left_front', 'left_mid', 'left_rear', 'right_front', 'right_mid', 'right_rear', 'center', 'up', 'down']
                        rospy.loginfo("--------------------------------")
                        for key in obstacle_keys:
                            if ob_data.get(key):
                                rospy.logwarn(f"| 检测到{key}障碍物，注意操作！ |")
                        rospy.loginfo("--------------------------------")

                    elif self.ob_status == 2: # 自动sync 避障逻辑
                        self.duco_ob.stop(True)
                        if self.paint_motion == 1 or self.paint_motion == 5:
                            if (ob_data['left_mid'] and ob_data['left_rear']) or (ob_data['right_mid'] and ob_data['right_rear']):
                                self.ob_flag = True
                                self.ob_safe_pos()
                            elif ob_data['left_mid'] or ob_data['right_mid'] or ob_data['center']:
                                self.ob_flag = True
                                # self.duco_ob.servoj_pose([tcp_pos[0] + 0.2, tcp_pos[1], tcp_pos[2], self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                                self.duco_ob.movel([tcp_pos[0] + 0.2, tcp_pos[1], tcp_pos[2], self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, 0, '', '', '', False)
                                rospy.logwarn("| 检测到障碍物，向后躲避！ |")
                            # elif ob_data['up']:
                            #     self.ob_flag = True
                            #     self.duco_ob.servoj_pose([tcp_pos[0], tcp_pos[1], tcp_pos[2] - 0.15, tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            #     rospy.logwarn("| 检测到上方障碍物障碍物，向下躲避！ |")
                            # elif ob_data['down']:
                            #     self.ob_flag = True
                            #     self.duco_ob.servoj_pose([tcp_pos[0], tcp_pos[1], tcp_pos[2] + 0.15, tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            #     rospy.logwarn("| 检测到下方障碍物，向上躲避！ |")
                            else:
                                self.ob_flag = False

                        elif self.paint_motion == 2 or self.paint_motion == 4:
                            if (ob_data['left_mid'] and ob_data['left_rear']) or (ob_data['right_mid'] and ob_data['right_rear']):
                                self.ob_flag = True
                                self.ob_safe_pos()
                            elif ob_data['left_mid'] or ob_data['right_mid']:
                                self.ob_flag = True
                                # self.duco_ob.servoj_pose([tcp_pos[0] + 0.2, tcp_pos[1], tcp_pos[2], self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                                self.duco_ob.movel([tcp_pos[0] + 0.2, tcp_pos[1], tcp_pos[2], self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, 0, '', '', '', False)
                                rospy.logwarn("| 检测到障碍物，向后躲避！ |")
                            # elif ob_data['up']:
                            #     self.ob_flag = True
                            #     self.duco_ob.servoj_pose([tcp_pos[0], tcp_pos[1], tcp_pos[2] - 0.15, tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            #     rospy.logwarn("| 检测到上方障碍物障碍物，向下躲避！ |")
                            # elif ob_data['down']:
                            #     self.ob_flag = True
                            #     self.duco_ob.servoj_pose([tcp_pos[0], tcp_pos[1], tcp_pos[2] + 0.15, tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            #     rospy.logwarn("| 检测到下方障碍物，向上躲避！ |")
                            else:
                                self.ob_flag = False

                        elif self.paint_motion == 3:
                            if (ob_data['left_mid'] and ob_data['left_rear']) or (ob_data['right_mid'] and ob_data['right_rear']):
                                self.ob_flag = True
                                self.ob_safe_pos()
                            elif ob_data['left_mid'] or ob_data['right_mid'] or ob_data['center']:
                                self.ob_flag = True
                                # self.duco_ob.servoj_pose([tcp_pos[0] + 0.2, tcp_pos[1], tcp_pos[2], self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                                self.duco_ob.movel([tcp_pos[0] + 0.2, tcp_pos[1], tcp_pos[2], self.init_pos[3], self.init_pos[4], self.init_pos[5]], self.ob_vel, self.ob_acc, 0, '', '', '', False)
                                rospy.logwarn("| 检测到障碍物，向后躲避！ |")
                            # elif ob_data['up']:
                            #     self.ob_flag = True
                            #     self.duco_ob.servoj_pose([tcp_pos[0], tcp_pos[1], tcp_pos[2] - 0.15, tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            #     rospy.logwarn("| 检测到上方障碍物障碍物，向下躲避！ |")
                            # elif ob_data['down']:
                            #     self.ob_flag = True
                            #     self.duco_ob.servoj_pose([tcp_pos[0], tcp_pos[1], tcp_pos[2] + 0.15, tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.ob_vel, self.ob_acc, '', '', '', True)
                            #     rospy.logwarn("| 检测到下方障碍物，向上躲避！ |")
                            else:
                                self.ob_flag = False
                        

                else:
                    self.ob_flag = False
            
            rospy.sleep(0.5)

    def get_cylinder_param(self):
        # TODO: 获取圆柱圆心坐标及圆柱半径
        cx, cy, cz = 1.8, 0.3, 1.2   # 圆心坐标
        cy_radius = 0.5               # 圆柱半径
        return cx, cy, cz, cy_radius
    
    def _H_info_callback(self, msg):
        self.duco_H = DucoCobot(self.ip, PORT)
        self.duco_H.open()
        self.H_info = msg
        self.last_H_time = time.time()
        
        # 检查是否收到有效信息（3条线）
        if len(msg.lines) == 3:
            # 将当前有效信息添加到历史队列
            self.H_detection_history.append(msg)
            
            # 检查是否已经收集到3次有效信息
            if len(self.H_detection_history) == 3:
                self.H_processing_ready = True
                rospy.loginfo("已收集到3次有效H检测信息，开始处理...")
                
                # 计算3次检测结果的平均值
                avg_msg = self._calculate_average_H_detection()
                
                # 使用平均值进行原来的处理逻辑
                self._process_H_detection(avg_msg)
                
                # 处理完成后重置状态
                self.H_processing_ready = False
                self.H_detection_history.clear()
            else:
                rospy.loginfo(f"正在收集H检测信息... ({len(self.H_detection_history)}/3)")
        else:
            # 如果收到的信息无效，清空历史记录
            self.H_detection_history.clear()
            self.H_processing_ready = False

    def _calculate_average_H_detection(self):
        """
        计算3次H检测结果的平均值
        返回: 包含平均值的LineDetectionArray消息
        """
        if len(self.H_detection_history) != 3:
            return None
            
        # 创建平均消息
        avg_msg = LineDetectionArray()
        avg_msg.header = self.H_detection_history[0].header
        
        # 对每条线分别计算平均值
        for line_idx in range(3):  # 3条线
            avg_line = LineInfo()
            avg_line.id = self.H_detection_history[0].lines[line_idx].id
            avg_line.type = self.H_detection_history[0].lines[line_idx].type
            
            # 计算start_point的平均值
            avg_line.start_point.x = sum(msg.lines[line_idx].start_point.x for msg in self.H_detection_history) / 3
            avg_line.start_point.y = sum(msg.lines[line_idx].start_point.y for msg in self.H_detection_history) / 3
            avg_line.start_point.z = sum(msg.lines[line_idx].start_point.z for msg in self.H_detection_history) / 3
            
            # 计算end_point的平均值
            avg_line.end_point.x = sum(msg.lines[line_idx].end_point.x for msg in self.H_detection_history) / 3
            avg_line.end_point.y = sum(msg.lines[line_idx].end_point.y for msg in self.H_detection_history) / 3
            avg_line.end_point.z = sum(msg.lines[line_idx].end_point.z for msg in self.H_detection_history) / 3
            
            # 计算其他属性的平均值
            avg_line.length = sum(msg.lines[line_idx].length for msg in self.H_detection_history) / 3
            avg_line.distance = sum(msg.lines[line_idx].distance for msg in self.H_detection_history) / 3
            avg_line.angle_deg = sum(msg.lines[line_idx].angle_deg for msg in self.H_detection_history) / 3
            
            avg_msg.lines.append(avg_line)
            rospy.loginfo("3次H完成处理")
        return avg_msg

    def _process_H_detection(self, msg):
        """
        处理H检测信息的核心逻辑（原来的处理代码）
        """
        if self.find_mode:
            rospy.loginfo("start process")
            tcp_pos = self.duco_H.get_tcp_pose()
            self.H_find_flag = True
            for line in msg.lines:
                if line.type == 0:
                    web_line = line
                elif line.type == 1:
                    if line.start_point.z > 0:
                        up_line = line
                    elif line.start_point.z < 0:
                        down_line = line

            if web_line is not None:
                if web_line.start_point.z > web_line.end_point.z:
                    self.start_z = web_line.start_point.z
                    self.web_length = web_line.length
                    self.web_distance = web_line.distance
                    self.web_top_point = [web_line.start_point.x + LEFT_RADAR_OFFSET[2] + tcp_pos[0], web_line.start_point.y + LEFT_RADAR_OFFSET[0] + tcp_pos[1], web_line.start_point.z + LEFT_RADAR_OFFSET[1] + tcp_pos[2]]
                
                else:
                    self.start_z = web_line.end_point.z
                    self.web_length = web_line.length
                    self.web_distance = web_line.distance
                    self.web_top_point = [web_line.end_point.x + LEFT_RADAR_OFFSET[2] + tcp_pos[0], web_line.end_point.y + LEFT_RADAR_OFFSET[0] + tcp_pos[1], web_line.end_point.z + LEFT_RADAR_OFFSET[1] + tcp_pos[2]]

                
                self.web_height = web_line.length
                
            if up_line is not None:
                self.flange_up_width = up_line.length
                
            if down_line is not None:
                self.flange_down_width = down_line.length
            rospy.loginfo("有数据")
        else:
            if self.position_flag:
                tcp_pos = self.duco_H.get_tcp_pose()
                if len(msg.lines) == 3:
                    for line in msg.lines:
                        if line.type == 0:
                            web_line = line
                        elif line.type == 1:
                            if line.start_point.z > 0:
                                up_line = line
                            elif line.start_point.z < 0:
                                down_line = line

                    if web_line is not None:
                        self.distance_to_web = abs(web_line.start_point.x)
                        
                else:
                    self.distance_to_web = -1

    def _line_detection_callback(self, msg):
        # 处理 /twin_radar/line_detection_info 消息
        # 只保留 id=999 的 line 信息，并提取 distance 和 angle_deg
        self.line_detection = msg
        self.last_line_time = time.time()
        
        # 查找 id=999 的 line
        target_line = None
        for line in msg.lines:
            if line.id == 999:
                target_line = line
                break
        
        if target_line is not None and self.position_flag:
            # 提取 distance 和 angle_deg
            self.surface_distance = abs(target_line.start_point.x)
            self.surface_angle_deg = target_line.angle_deg
            self.target_dist_in_surface = self.painting_dist - abs(abs(self.paint_top[0]) - abs(self.paint_center[0]))

            # rospy.loginfo(f"找到目标线 (id=999): distance={self.target_distance:.3f}, angle_deg={self.target_angle_deg:.3f}")
        else:
            self.surface_distance = -1
            # rospy.logwarn("未找到 id=999 的目标线")
            pass

    def _obstacle_flags_callback(self, msg):
        self.obstacle_flags = msg
        self.last_ob_time = time.time()
        
        # 处理障碍物标志信息
        # 将当前帧的状态添加到历史队列中
        self.left_front_history.append(msg.left_front)
        self.left_mid_history.append(msg.left_mid)
        self.left_rear_history.append(msg.left_rear)
        self.right_front_history.append(msg.right_front)
        self.right_mid_history.append(msg.right_mid)
        self.right_rear_history.append(msg.right_rear)
        self.center_history.append(msg.center)
        self.up_history.append(msg.up)
        self.down_history.append(msg.down)
        
        # 应用过滤逻辑：只有当连续3帧都检测到障碍物时才设置为true
        self.left_front_obstacle = self._check_continuous_detection(self.left_front_history)
        self.left_mid_obstacle = self._check_continuous_detection(self.left_mid_history)
        self.left_rear_obstacle = self._check_continuous_detection(self.left_rear_history)
        self.right_front_obstacle = self._check_continuous_detection(self.right_front_history)
        self.right_mid_obstacle = self._check_continuous_detection(self.right_mid_history)
        self.right_rear_obstacle = self._check_continuous_detection(self.right_rear_history)
        self.center_obstacle = self._check_continuous_detection(self.center_history)
        self.up_obstacle = self._check_continuous_detection(self.up_history)
        self.down_obstacle = self._check_continuous_detection(self.down_history)
        
        # 读取时间戳和安全距离
        self.obstacle_stamp = msg.stamp
        self.safe_distance = msg.safe_distance
        
    
    def _check_continuous_detection(self, history_queue):
        """
        检查历史队列中是否连续检测到障碍物
        参数:
            history_queue: 包含最近几帧检测结果的队列
        返回:
            bool: 如果连续3帧都检测到则返回True，否则返回False
        """
        # 如果队列还没有填满，返回False
        if len(history_queue) < self.obstacle_filter_frames:
            return False
        
        # 检查最近3帧是否都为True
        return all(history_queue)
    
    def get_obstacle_status(self):
        """
        获取过滤后的障碍物状态信息
        返回: 包含所有障碍物状态的字典
        """
        return {
            'left_front': self.left_front_obstacle,
            'left_mid': self.left_mid_obstacle,
            'left_rear': self.left_rear_obstacle,   
            'right_front': self.right_front_obstacle,
            'right_mid': self.right_mid_obstacle,
            'right_rear': self.right_rear_obstacle,
            'center': self.center_obstacle,
            'up': self.up_obstacle,
            'down': self.down_obstacle,
            'stamp': self.obstacle_stamp,
            'safe_distance': self.safe_distance
        }
    
    def get_raw_obstacle_status(self):
        """
        获取原始（未过滤）的障碍物状态信息
        返回: 包含所有原始障碍物状态的字典
        """
        return {
            'left_front': self.left_front_history[-1] if self.left_front_history else False,
            'left_mid': self.left_mid_history[-1] if self.left_mid_history else False,
            'left_rear': self.left_rear_history[-1] if self.left_rear_history else False,
            'right_front': self.right_front_history[-1] if self.right_front_history else False,
            'right_mid': self.right_mid_history[-1] if self.right_mid_history else False,
            'right_rear': self.right_rear_history[-1] if self.right_rear_history else False,
            'center': self.center_history[-1] if self.center_history else False,
            'up': self.up_history[-1] if self.up_history else False,
            'down': self.down_history[-1] if self.down_history else False,
            'stamp': self.obstacle_stamp,
            'safe_distance': self.safe_distance
        }
    
    def is_obstacle_detected(self):
        """
        检查是否有任何障碍物被检测到
        返回: True 如果有障碍物，False 如果没有
        """
        return any([
            self.left_front_obstacle,
            self.left_mid_obstacle,
            self.left_rear_obstacle,
            self.right_front_obstacle,
            self.right_rear_obstacle,
            self.right_mid_obstacle,
            self.center_obstacle,
            self.up_obstacle,
            self.down_obstacle
        ])
    
    def get_directional_distance(self, direction):
        """
        获取指定方向的距离值
        Args:
            direction: 方向名称 ("front", "down", "up")
        Returns:
            float: 距离值（米），-1表示无效或超时
        """
        return self.directional_laser.get_front_distance(direction)
    
    def get_all_directional_distances(self):
        """
        获取所有方向的距离值
        Returns:
            dict: 包含所有方向距离值的字典
        """
        return self.directional_laser.get_all_front_distances()
        
        # 读取/topic中的按键输入
    def _keys_callback(self, msg):
        self.latest_keys = list(msg.keys)
        self.last_key_time = time.time()

    # 第一个元素为按钮
    def get_key_input(self):
        if time.time() - self.last_key_time > KEYTIMEOUT:
            self.emergency_stop_flag = True
            key_bits = 0
            rospy.logerr("与控制端连接丢失，执行急停，阈值使用默认值！")
        else:
            key_bits = self.latest_keys[0]
            self.painting_deg_surface = abs(self.latest_keys[7])
            self.painting_deg_flange = abs(self.latest_keys[8])
            self.painting_dist = self.latest_keys[9]/1000

        # 按位解析
        return KeyInputStruct(
            x0      = (key_bits >> 0) & 1,
            x1      = (key_bits >> 1) & 1,
            y0      = (key_bits >> 2) & 1,
            y1      = (key_bits >> 3) & 1,
            z0      = (key_bits >> 4) & 1,
            z1      = (key_bits >> 5) & 1,
            init    = (key_bits >> 6) & 1,
            serv    = (key_bits >> 7) & 1,
            multi   = (key_bits >> 8) & 1,
            start   = (key_bits >> 9) & 1,
            rx0     = (key_bits >> 10) & 1,
            rx1     = (key_bits >> 11) & 1,
            ry0     = (key_bits >> 12) & 1,
            ry1     = (key_bits >> 13) & 1,
            rz0     = (key_bits >> 14) & 1,
            rz1     = (key_bits >> 15) & 1,
            clog    = (key_bits >> 16) & 1,
            find    = (key_bits >> 17) & 1,
            high     = (key_bits >> 18) & 1,
            center  = (key_bits >> 19) & 1,
            low  = (key_bits >> 20) & 1,
            top = (key_bits >> 21) & 1,
            bottom = (key_bits >> 22) & 1,
            # TODO: 添加更多按键位的解析_max=31
        )

    # 堵枪清理动作
    def clog_function(self):
        if not self.clog_flag:      # 发生堵枪时第一次按下，转到清理位置，执行清理工作
            self.clog_flag = True
            self.ob_status = 1
            self.tcp_pos = self.duco_cobot.get_tcp_pose()
            self.duco_cobot.servoj_pose(self.clog_pos, self.vel * 1.5, self.acc, '', '', '', True)
            rospy.loginfo("已移动到清理堵枪位置: %s" % self.clog_pos)
            if self.autopaint_flag:
                rospy.loginfo("自动喷涂模式已暂停，待再次按下堵枪按钮时恢复位置并继续自动喷涂模式")
        else:                       # 堵枪清理结束之后按下按钮，回到之前位置继续工作
            self.clog_flag = False
            self.duco_cobot.servoj_pose(self.tcp_pos, self.vel * 1.5, self.acc, '', '', '', True)
            rospy.loginfo("已回到堵枪前位置: %s" % self.tcp_pos)
            if self.autopaint_flag:
                rospy.loginfo("自动喷涂模式已恢复")

    def point_on_circle(self, center_x, center_z, radius, angle_deg):
        """计算圆上指定角度的点坐标"""
        theta = math.radians(angle_deg)  # 角度转弧度
        x = center_x + radius * math.cos(theta)
        z = center_z + radius * math.sin(theta)
        return x, z

    # 寻找五个喷涂位姿
    def find_central_pos(self):
        self.position_flag = False
        self.ob_status = 1
        
        self.H_find_flag = False
        
        tcp_pos = self.duco_cobot.get_tcp_pose()
        start_z = tcp_pos[2]  # 当前 z 方向为上下

        scan_data = []  # 保存 [z坐标, 距离值]

        rospy.loginfo("从上到下开始扫描...")

        steps = int(self.scan_range / self.step_size)
        for i in range(steps):
            # 向下移动一小步
            if self.emergency_stop_flag:
                rospy.logwarn("检测到急停信号，停止寻找钢梁中心位置！")
                return
            tcp_pos[2] = start_z - i * self.step_size
            self.duco_cobot.servoj_pose(tcp_pos, self.vel, self.acc, '', '', '', True)
            rospy.sleep(0.05)

            # 读取前向激光传感器
            dist = self.get_directional_distance("right")
            if dist > 0:
                scan_data.append((tcp_pos[2], dist))  # 记录当前高度和距离值
                rospy.loginfo(f"scan z={tcp_pos[2]:.3f}m, front={dist:.3f}m")

        rospy.loginfo("扫描完成，开始检测突变边缘...")

        edge_positions = []
        for i in range(1, len(scan_data)):
            if self.emergency_stop_flag:
                print("检测到急停信号，停止寻找钢梁中心位置！")
                return
            prev = scan_data[i - 1][1]
            curr = scan_data[i][1]
            if abs(curr - prev) > self.min_jump_threshold:
                z_pos = scan_data[i][0]
                dist = scan_data[i][1]
                edge_positions.append((z_pos, dist))

        if len(edge_positions) >= 2:
            top_edge = edge_positions[0][0]
            bottom_edge = edge_positions[-1][0]
            center_z = (top_edge + bottom_edge) / 2

            # 计算目标末端位置
            center_pos = list(self.duco_cobot.get_tcp_pose())
            center_pos[2] = center_z

            # 移动到目标位置
            self.duco_cobot.servoj_pose(center_pos, self.vel, self.acc, '', '', '', True)
            # 计算当前喷涂距离，并移动到目标喷涂距离
            dist = self.get_directional_distance("right")
            center_pos[0] -= dist - self.painting_dist + 0.2
            self.duco_cobot.servoj_pose(center_pos, self.vel, self.acc, '', '', '', True)
            self.paint_center = center_pos

        # 设置寻找模式
        self.find_mode = True
        rospy.loginfo(" |-| 进入寻找模式，等待收集H检测信息...")
        self.find_mode = True
        rospy.sleep(5)
        # 第二步：等待H_find_flag为True
        start_time = time.time()
        while not self.H_find_flag:
            if self.emergency_stop_flag:
                rospy.logwarn("检测到急停信号，停止寻找钢梁中心位置！")
                self.find_mode = False
                return
            # if self.ob_flag:
            #     rospy.logwarn(" |-| 检测到障碍物，无法寻找喷涂位姿！")
            #     self.find_mode = False
            #     return
            
            # 检查是否超时（10秒）
            if time.time() - start_time > 60:
                rospy.logwarn(" |-| 等待H检测处理结果超时（60秒），无法寻找喷涂位姿！")
                self.find_mode = False
                return
                
            rospy.sleep(0.001)  # 短暂等待，避免占用过多CPU
        
        if not self.find_mode:
            return
            
        # 第三步：开始执行寻找喷涂位姿的动作
        rospy.loginfo(" |-| 开始寻找喷涂位姿,请确保机械臂在目标梁中间或上方")
        
        center_pos = []
        tcp_pos = self.duco_cobot.get_tcp_pose()

        self.center_z = tcp_pos[2] - self.start_z + self.web_length / 2
        self.center_x = tcp_pos[0] + self.painting_dist - self.web_distance
        rospy.loginfo(f"center_x: {self.center_x}, center_z: {self.center_z}")
        self.duco_cobot.servoj_pose([self.center_x, tcp_pos[1], self.center_z, tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.vel, self.acc, '', '', '', True)
        self.paint_motion = 3
        rospy.sleep(0.5)
        rospy.loginfo(" |-| 移动到center位姿完成，准备计算其余位姿")

        tcp_pos = self.duco_cobot.get_tcp_pose()
        self.paint_center = tcp_pos # 喷涂中心点

        # 喷涂上表面
        web_top_x = tcp_pos[0] - self.painting_dist + 0.04
        web_top_z = tcp_pos[2] + self.web_height/2
        web_bottom_x = web_top_x
        web_bottom_z = web_top_z - self.web_height
        print(f"web_top_x: {web_top_x}, web_top_z: {web_top_z}")
        self.surface_painting_dist = self.painting_dist
        self.flange_painting_dist = self.painting_dist - self.flange_painting_dist_adjust
        # 喷涂上表面
        self.paint_top = [
            (self.point_on_circle(web_top_x, web_top_z, self.surface_painting_dist, self.painting_deg_surface)[0]),
            tcp_pos[1],
            (self.point_on_circle(web_top_x, web_top_z, self.surface_painting_dist, self.painting_deg_surface)[1]),
            tcp_pos[3], tcp_pos[4], tcp_pos[5]] 
        # 喷涂上翼面
        self.paint_low = [
            (self.point_on_circle(web_top_x, web_top_z, self.flange_painting_dist, -self.painting_deg_flange)[0]),
            tcp_pos[1],
            (self.point_on_circle(web_top_x, web_top_z, self.flange_painting_dist, -self.painting_deg_flange)[1]),
            tcp_pos[3], tcp_pos[4], tcp_pos[5]] 

        # 喷涂下表面
        self.paint_bottom = [
            (self.point_on_circle(web_bottom_x, web_bottom_z, self.surface_painting_dist, -self.painting_deg_surface)[0]),
            tcp_pos[1],
            (self.point_on_circle(web_bottom_x, web_bottom_z, self.surface_painting_dist, -self.painting_deg_surface)[1]),
            tcp_pos[3], tcp_pos[4], tcp_pos[5]]
        # 喷涂下翼面
        self.paint_high = [
            (self.point_on_circle(web_bottom_x, web_bottom_z, self.flange_painting_dist, self.painting_deg_flange)[0]),
            tcp_pos[1],
            (self.point_on_circle(web_bottom_x, web_bottom_z, self.flange_painting_dist, self.painting_deg_flange)[1]),
            tcp_pos[3], tcp_pos[4], tcp_pos[5]] 

        self.target_dist_in_flange = self.painting_dist - abs(abs(self.paint_high[0]) - abs(self.paint_center[0]))
        self.target_dist_in_web = self.painting_dist
        rospy.loginfo("------ |-| 已找到5个喷涂位姿，选择位置开始喷涂！--------\n")
        rospy.loginfo("   ↓       喷涂上表面位姿： %s" % self.paint_top)
        rospy.loginfo("===== ↙    喷涂下翼面位姿： %s" % self.paint_high)
        rospy.loginfo("  |     ←  喷涂中心位姿：  %s" % self.paint_center)
        rospy.loginfo("===== ↖    喷涂上翼面位姿： %s" % self.paint_low)
        rospy.loginfo("   ↑       喷涂下表面位姿： %s" % self.paint_bottom)
        rospy.loginfo("\n------------------------------------------------")
        self.position_flag = True
        self.find_mode = False
        rospy.loginfo(" |-| 寻找喷涂位姿完成！")
                        
    def pid_dist_control(self, distance, target_dist, dt):
        v2 = 0.0  # x轴默认速度为0
        if distance != -1 and not self.clog_flag:
            # 1. 数据滤波
            raw_front_dist = distance
            if raw_front_dist > 0:  # 确保是有效读数
                self.dist_history.append(raw_front_dist)

            if len(self.dist_history) > 0:
                filtered_front_dist = sum(self.dist_history) / len(self.dist_history)

                # 2. 控制死区
                deadband_threshold = DEADZONE  # 单位: mm, 可根据实际情况调整
                error = filtered_front_dist - target_dist

                # 3. PID计算 (仅在死区外)
                if abs(error) > deadband_threshold:
                    v2 = self.pid_z.compute(target_dist, filtered_front_dist, dt)
                    # 限制最大速度
                    v2 = max(min(v2, 0.15), -0.15)  
        return v2

    # 自动喷边走边喷
    def auto_paint_sync(self):
        rospy.loginfo("-----进入自动模式-----")
        self.autopaint_flag = True
        v2 = 0.0  # 初始化前后速度
        cur_time = time.time()
        last_time = cur_time

        if self.paint_motion == 0:            
            rospy.logwarn("| 当前机械臂不在喷涂位置，请先选择一个喷涂位置再开始自动程序 |")
            return

        else:
            while self.autopaint_flag:
                if self.clog_flag:
                    self.ob_status = 1
                else:
                    self.ob_status = 2
                if self.ob_flag:
                    rospy.loginfo("obobobobob")
                    rospy.sleep(2.5)
                    continue
                tcp_pos = self.duco_cobot.get_tcp_pose()
                key_input = self.get_key_input()
                now = time.time()
                dt = now - last_time
                last_time = now

                if key_input.clog:
                    self.clog_function()
                    continue
                # 喷涂上下翼板
                if (self.paint_motion == 2 or self.paint_motion == 4) and (self.last_H_time - time.time() < 1):
                    target_dist = self.target_dist_in_flange
                    now_dist =(self.get_directional_distance("left") + self.get_directional_distance("right")) / 2
                    v2 = self.pid_dist_control(now_dist, target_dist, dt)
                    rospy.loginfo(f"v2: {v2}\ntarget_dist_in_flange: {self.target_dist_in_flange}, \ndistance_now: {now_dist}")

                # 喷涂上下表面
                elif (self.paint_motion == 1 or self.paint_motion == 5) and (self.last_line_time - time.time() < 1):
                    target_dist = self.target_dist_in_surface
                    now_dist = self.surface_distance
                    v2 = self.pid_dist_control(now_dist, target_dist, dt)
                    rospy.loginfo(f"v2: {v2}\ntarget_dist_in_surface: {self.target_dist_in_surface}, \ndistance_now: {now_dist}")
                # 喷涂中腹板
                elif self.paint_motion == 3 and (self.last_H_time - time.time() < 1):
                    target_dist = self.target_dist_in_web
                    now_dist = (self.get_directional_distance("left") + self.get_directional_distance("right")) / 2
                    v2 = self.pid_dist_control(now_dist, target_dist, dt)
                    rospy.loginfo(f"v2: {v2}\ntarget_dist_web: {self.target_dist_in_web}, \ndistance_now: {now_dist}")
                else:
                    v2 = 0.0
                # rospy.logdebug("v2: %f" % v2)
                self.duco_cobot.speedl([0, 0, v2, 0, 0, 0], self.acc * 0.9, -1, False)
                # self.duco_cobot.speedl([0, 0, 0, 0, 0, 0], self.acc * 0.9, -1, False)


        rospy.loginfo("-----退出自动模式-----")

    # 自动喷涂，车辆不动机械臂动
    def auto_paint_interval(self):
        # TODO: 自动喷涂，车辆不动机械臂动
        pass

    def pos_move(self, aim_pos):
        if self.position_flag and aim_pos is not None:
            tcp_pos = self.duco_cobot.get_tcp_pose()
            self.duco_cobot.servoj_pose([self.paint_center[0], tcp_pos[1], tcp_pos[2], tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.vel, self.acc, '', '', '', True)
            self.duco_cobot.servoj_pose([self.paint_center[0], tcp_pos[1], aim_pos[2], tcp_pos[3], tcp_pos[4], tcp_pos[5]], self.vel, self.acc, '', '', '', True)
            self.duco_cobot.servoj_pose(aim_pos, self.vel, self.acc, '', '', '', True)
            if aim_pos == self.paint_top:
                self.paint_motion = 1
                rospy.loginfo("移动到喷涂上表面位置： %s" % aim_pos)
            elif aim_pos == self.paint_high:
                self.paint_motion = 2
                rospy.loginfo("移动到喷涂下翼板位置： %s" % aim_pos)
            elif aim_pos == self.paint_center:
                self.paint_motion = 3
                rospy.loginfo("移动到喷涂中腹板位置： %s" % aim_pos)
            elif aim_pos == self.paint_low:
                self.paint_motion = 4
                rospy.loginfo("移动到喷涂上翼板位置： %s" % aim_pos)
            elif aim_pos == self.paint_bottom:
                self.paint_motion = 5
                rospy.loginfo("移动到喷涂下表面位置： %s" % aim_pos)
            elif aim_pos == self.init_pos:
                self.paint_motion = 0
                rospy.loginfo("移动到初始位置： %s" % aim_pos)
            elif aim_pos == self.serv_pos:
                self.paint_motion = 0
                rospy.loginfo("移动到维修位置： %s" % aim_pos)
            else:
                self.paint_motion = 0
                rospy.loginfo("移动到目标位置： %s" % aim_pos)
        else:
            rospy.logwarn("| 无效位置坐标！请先执行自动寻找程序 |")

    def paint_pos_adjust(self):
        if self.paint_center is not None:
            if self.paint_motion == 1:
                tcp_pos = self.duco_cobot.get_tcp_pose()
                self.paint_adjust[0] = tcp_pos[0] - self.paint_top[0]
                self.paint_adjust[1] = tcp_pos[2] - self.paint_top[2]
        else:
            rospy.logwarn("| 请先执行自动寻找程序 |")

    def run(self):
        rospy.loginfo("等待移动到初始位置...")
        
        self.paint_motion = 0
        # self.duco_cobot.movej2(self.init_pos, 2*self.vel, self.acc, 0, True)
        self.duco_cobot.servoj_pose(self.init_pos, self.vel, self.acc, '', '', '', True)
        rospy.loginfo("移动到初始位置: %s" % self.init_pos)
        rospy.loginfo("================================")
        rospy.loginfo("'/Duco_state'                        话题： 机械臂状态")
        rospy.loginfo("'/left_radar/scan'                   话题： 左雷达数据")
        rospy.loginfo("'/left_radar/filtered_scan'          话题： 左雷达数据(滤波后)")
        rospy.loginfo("'/right_radar/scan'                  话题： 右雷达数据")
        rospy.loginfo("'/right_radar/filtered_scan'         话题： 右雷达数据(滤波后)")
        rospy.loginfo("'/main_radar/scan'                   话题： 主雷达数据")
        rospy.loginfo("'/main_radar/filtered_scan'          话题： 主雷达数据(滤波后)")
        rospy.loginfo("'/obstacle_avoidance/pointcloud'     话题： 避障点云")
        rospy.loginfo("'/obstacle_flags'                    话题： 避障标志(自定义消息)")
        rospy.loginfo("'/twin_radar/line_detection_info'    话题： 双线检测信息(自定义消息)")
        rospy.loginfo("'/left_radar/H_detection_info'       话题： 左雷达H检测信息(自定义消息)")
        rospy.loginfo("================================")
        
        try:
            while self.sysrun and not rospy.is_shutdown():
                key_input = self.get_key_input()
                self.duco_cobot.switch_mode(1)
                self.autopaint_flag = False
                self.find_mode = False
                self.ob_status = 1

                v0 = self.auto_vel                # arm left-/right+
                v1 = self.auto_vel                 # arm up+/down-
                v2 = self.auto_vel                 # arm forward+/backward-
                v3 = -self.auto_vel * 2           # arm head up+/down-
                v4 = self.auto_vel  * 2            # arm head left+/right-
                v5 = self.auto_vel  * 2            # arm head rotate left-/right+
                
                #自动喷涂
                if key_input.start:
                    if not self.emergency_stop_flag:
                        self.auto_paint_sync()
                        # self.auto_paint_interval()
                #堵枪清理
                elif key_input.clog:
                    if not self.emergency_stop_flag:
                        self.clog_function()
                #寻找五个位姿
                elif key_input.find:
                    if not self.emergency_stop_flag:
                        self.find_central_pos()
                #机械臂末端向  前
                elif key_input.x0:
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, 0, v2, 0, 0, 0],self.acc ,-1, False)
                    now_dist = self.get_directional_distance("right")
                    rospy.loginfo(f"distance_now: {now_dist}")
                #机械臂末端向  后
                elif key_input.x1:
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, 0, -v2, 0, 0, 0],self.acc ,-1, False)
                    now_dist = self.get_directional_distance("right")
                    rospy.loginfo(f"distance_now: {now_dist}")
                #机械臂末端向  右
                elif key_input.y1:
                    self.ob_status = 1
                    self.duco_cobot.speedl([v0, 0, 0, 0, 0, 0], self.acc, -1, False)
                #机械臂末端向  左
                elif key_input.y0: 
                    self.ob_status = 1
                    self.duco_cobot.speedl([-v0, 0, 0, 0, 0, 0], self.acc, -1, False)
                #机械臂末端向  上
                elif key_input.z1: 
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, v1, 0, 0, 0, 0],self.acc ,-1, False)
                    now_dist = self.get_directional_distance("right")
                    rospy.loginfo(f"distance_now: {now_dist}")
                #机械臂末端向  下
                elif key_input.z0:
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, -v1, 0, 0, 0, 0],self.acc ,-1, False)
                    now_dist = self.get_directional_distance("right")
                    rospy.loginfo(f"distance_now: {now_dist}")
                #初始化位置
                elif key_input.init:                    
                    self.ob_status = 1
                    self.duco_cobot.servoj_pose(self.init_pos, self.vel, self.acc, '', '', '', True)
                    rospy.loginfo("移动到初始位置")
                    self.paint_motion = 0
                #维修位置
                elif key_input.serv:                    
                    self.ob_status = 1
                    self.duco_cobot.servoj_pose(self.serv_pos, self.vel, self.acc, '', '', '', True)
                    rospy.loginfo("移动到维修位置")
                    self.paint_motion = 0
                #喷涂顶部
                elif key_input.top:
                    self.pos_move(self.paint_top)
                #喷涂下翼板
                elif key_input.high:
                    self.pos_move(self.paint_high)
                #喷涂中腹板
                elif key_input.center:
                    self.pos_move(self.paint_center)
                #喷涂上翼板
                elif key_input.low:
                    self.pos_move(self.paint_low)
                #喷涂底部
                elif key_input.bottom:
                    self.pos_move(self.paint_bottom)
                #机械臂末端转  pitch上
                elif key_input.rx0: 
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, 0, 0, 0, 0, v5], self.acc, -1, False)
                #机械臂末端转  pitch下
                elif key_input.rx1: 
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, 0, 0, 0, 0, -v5], self.acc, -1, False)
                #机械臂末端转  roll左
                elif key_input.ry0: 
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, 0, 0, v3, 0, 0], self.acc, -1, False)
                #机械臂末端转  roll右
                elif key_input.ry1: 
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, 0, 0, -v3, 0, 0], self.acc, -1, False)
                #机械臂末端转  yaw左
                elif key_input.rz0: 
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, 0, 0, 0, v4, 0], self.acc, -1, False)
                #机械臂末端转  yaw右
                elif key_input.rz1: 
                    self.ob_status = 1
                    self.duco_cobot.speedl([0, 0, 0, 0, -v4, 0], self.acc, -1, False)

                # TODO 圆柱喷涂
                # elif btn_y:
                #     self.duco_cobot.switch_mode(1)
                #     auto_painter = CylinderAutoPaint(self.duco_cobot, self.init_pos, self.theta_deg, self.distance_to_cylinder, self.painting_width, self.vel, self.acc)
                #     auto_painter.auto_paint()

                else:
                    self.duco_cobot.speed_stop(False)
            
        except Exception as e:
            rospy.logerr(f"An unexpected error occurred: {e}")
            self.sysrun = False
            self.autopaint_flag = False

        except KeyboardInterrupt:
            rospy.loginfo("KeyboardInterrupt")
            self.sysrun = False
            self.autopaint_flag = False
            return

        finally:
            self.emergency_thread.join()
            self.ob_thread.join()
