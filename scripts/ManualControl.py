import time
import rospy
import threading

from DucoCobot import DucoCobot
from s21c_receive_data.msg import STP23
from key_input_pkg.msg import KeyInput
from CylinderPaint_duco import CylinderAutoPaint
from collections import deque
from config import *

class KeyInputStruct:
    def __init__(self, x0=0, x1=0, y0=0, y1=0, z0=0, z1=0,
                 init=0, serv=0, multi=0, start=0,
                 rx0=0, rx1=0, ry0=0, ry1=0, rz0=0, rz1=0,
                 clog=0, find=0):
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
        self.vel = DEFAULT_VEL # 机械臂末端速度
        self.acc = DEFAULT_ACC # 机械臂末端加速度
        self.aj_pos = [] # 当前关节角度
        self.tcp_pos = [] # 当前末端位姿
        self.history_pos = [] # 历史位置
        self.sysrun = True
        self.autopaint_flag = True
        self.clog_flag = False
        self.afterclog_auto = False
        self.init_pos = INIT_POS # 初始位置
        self.serv_pos = SERV_POS # 维修位置
        self.safe_pos = SAFE_POS # 安全位置
        self.clog_pos = CLOG_POS # 堵枪位置
        self.pid = SimplePID(kp=1, ki=0.0, kd=0.2)
        self.pid_z = SimplePID(kp=KP, ki=KI, kd=KD)
        self.front_sensor_history = deque(maxlen=5) # 滤波队列

        self.anticrash_up = ANTICRASH_UP
        self.anticrash_front = ANTICRASH_FRONT
        self.anticrash_left = ANTICRASH_LEFT
        self.anticrash_right = ANTICRASH_RIGHT

        self.theta_deg = PAINTDEG / 2  # 喷涂角度的一半
        self.distance_to_cylinder = self.anticrash_front  # 末端与圆柱表面距离
        self.painting_width = PAINTWIDTH  # 喷涂宽度

        self.latest_sensor_data = {"up": -1, "front": -1, "left_side": -1, "right_side": -1}
        self.sensor_subscriber = rospy.Subscriber('/STP23', STP23, self._sensor_callback)
        self.latest_keys = [0] * 20
        self.last_key_time = time.time()
        self.last_sensor_time = time.time()
        self.keys_subscriber = rospy.Subscriber('/key_input', KeyInput, self._keys_callback)

        self.emergency_stop_flag = False
        self.emergency_thread = threading.Thread(target=self.emergency_stop_thread, daemon=True)
        self.emergency_thread.start()
    
    # 急停线程
    def emergency_stop_thread(self):
        self._stop_event = threading.Event()
        self.duco_stop = DucoCobot(self.ip, PORT)
        self.duco_stop.open()
        while self.sysrun and not rospy.is_shutdown():
            self.emergency_stop_flag = False
            key_input = self.get_key_input()
            state = self.duco_stop.get_robot_state()
            if key_input.multi:
                print("检测到紧急停止按键，正在执行紧急停止！")
                self.emergency_stop_flag = True
                self.autopaint_flag = False
                self.duco_stop.stop(True)                
                if key_input.start:
                    self.autopaint_flag = False
                    self.sysrun = False
                    self.duco_stop.stop(True)
                    self.duco_stop.disable(True)
                    print("terminate robot")
                    break
                elif state[0] != 6:
                    print("restart robot")
                    if state[0] == 5:
                        self.duco_stop.enable(True)
                    if state[0] == 4:
                        self.duco_stop.power_on(True)
                        self.duco_stop.enable(True)
                    self.duco_stop.switch_mode(1)          
            self._stop_event.wait(0.05)

    def get_cylinder_param(self):
        # TODO: 获取圆柱圆心坐标及圆柱半径
        cx, cy, cz = 1.8, 0.3, 1.2   # 圆心坐标
        cy_radius = 0.5               # 圆柱半径
        return cx, cy, cz, cy_radius
    
    # 读取/topic中的按键输入
    def _keys_callback(self, msg):
        self.latest_keys = list(msg.keys)
        self.last_key_time = time.time()

    # 第一个元素为按钮
    def get_key_input(self):
        if time.time() - self.last_key_time > KEYTIMEOUT:
            key_bits = 0
            self.anticrash_up = ANTICRASH_UP
            self.anticrash_front = ANTICRASH_FRONT
            self.anticrash_left = ANTICRASH_LEFT
            self.anticrash_right = ANTICRASH_RIGHT
        else:
            key_bits = self.latest_keys[0]
            if len(self.latest_keys) > 1 and all(self.latest_keys[i] >= 0 for i in [1, 2, 3, 4]):
                self.anticrash_up = self.latest_keys[1]
                self.anticrash_front = self.latest_keys[2]
                self.anticrash_left = self.latest_keys[3]
                self.anticrash_right = self.latest_keys[4]
            else:
                self.anticrash_up = ANTICRASH_UP
                self.anticrash_front = ANTICRASH_FRONT
                self.anticrash_left = ANTICRASH_LEFT
                self.anticrash_right = ANTICRASH_RIGHT
        # 按位解析
        return KeyInputStruct(
            x0 = (key_bits >> 0) & 1,
            x1 = (key_bits >> 1) & 1,
            y0 = (key_bits >> 2) & 1,
            y1 = (key_bits >> 3) & 1,
            z0 = (key_bits >> 4) & 1,
            z1 = (key_bits >> 5) & 1,
            init = (key_bits >> 6) & 1,
            serv = (key_bits >> 7) & 1,
            multi = (key_bits >> 8) & 1,
            start = (key_bits >> 9) & 1,
            rx0 = (key_bits >> 10) & 1,
            rx1 = (key_bits >> 11) & 1,
            ry0 = (key_bits >> 12) & 1,
            ry1 = (key_bits >> 13) & 1,
            rz0 = (key_bits >> 14) & 1,
            rz1 = (key_bits >> 15) & 1,
            clog = (key_bits >> 16) & 1,
            find = (key_bits >> 17) & 1,
            # TODO: 添加更多按键位的解析
        )
    # 第二个及以后的元素为数据
    def get_ctrl_msg(self):
        return self.latest_keys[1:]
    
    # 读取/topic中的传感器数据
    def _sensor_callback(self, msg):
        self.latest_sensor_data = {
            "up": msg.Distance_1,
            "front": msg.Distance_2,
            "left": msg.Distance_3,
            "right": msg.Distance_4
        }
        self.last_sensor_time = time.time()

    def get_sensor_data(self):
        if time.time() - self.last_sensor_time > SENSORTIMEOUT:
            self.latest_sensor_data = {"up": -1, "front": -1, "left_side": -1, "right_side": -1}
        return self.latest_sensor_data
    
    # 记录当前位置
    def get_replay_pos(self, tcp_pos, history_pos=None):
        history_pos.append(tcp_pos)
        print("new position: %s counter: %d" % (tcp_pos, len(history_pos)))
        time.sleep(0.5)

    # 回放历史位置
    def position_replay(self, tcp_pos, history_pos=None):
        for tcp_pos in history_pos:
            try:
                index = history_pos.index(tcp_pos) + 1
                task_id = self.duco_cobot.movel(tcp_pos, self.vel, self.acc, 0, '', '', '', False)
                print("move to no.%d position: %s" % (index, tcp_pos))
                cur_time = time.time()
                while self.duco_cobot.get_noneblock_taskstate(task_id) != 4:
                    if time.time() - cur_time > 10:
                        print("Timeout.Move to no.%d position failed." % index)
                        break
                    
                    if self.duco_cobot.get_noneblock_taskstate(task_id) == 4:
                        break
                                
            except ValueError:
                print("Position %s not found in history." % tcp_pos)
        # 回到起始位置
        if len(history_pos) > 1:
            self.duco_cobot.movel(history_pos[0], self.vel, self.acc, 0, '', '', '', False)
            print("move to no.1 position: %s" % history_pos[0])
        # 无点位可回放
        elif len(history_pos) == 0:
            print("No position in history.Press LB to record position.")
            time.sleep(0.5)

    # 堵枪清理动作
    def clog_function(self):
        if not self.clog_flag:      # 发生堵枪时第一次按下，转到清理位置，执行清理工作
            self.clog_flag = True
            self.tcp_pose = self.duco_cobot.get_tcp_pose()
            self.duco_cobot.servoj_pose(self.clog_pos, self.vel * 1.5, self.acc, '', '', '', True)
            print("已移动到清理堵枪位置: %s" % self.clog_pos)
        else:                       # 堵枪清理结束之后按下按钮，回到之前位置继续工作
            self.clog_flag = False
            self.duco_cobot.servoj_pose(self.tcp_pose, self.vel * 1.5, self.acc, '', '', '', True)
            print("已回到堵枪前位置")

    def find_central_pos(self):
        print("开始寻找钢梁中心位置...")
        # TODO:
        # 1.上移至边缘并记录top_pos
        # 2.下移至边缘并记录bottom_pos
        # 3.计算中心位置_喷嘴和传感器的偏差以及喷嘴和机械臂末端的偏差
        # 4.移动到中心位置
        pass

    def find_central_pos(self):
        print("开始寻找钢梁中心位置...")

        scan_range = 1  # 扫描总行程，单位：米
        step_size = 0.01  # 每次移动的步长，单位：米
        pause_time = 0.05  # 每次读取后的停顿时间
        min_jump_threshold = 0.2  # 突变阈值，单位：米

        tcp_pos = self.duco_cobot.get_tcp_pose()
        start_z = tcp_pos[1]  # 当前 y 方向为上下

        # 保存 [y坐标, 距离值] 对
        scan_data = []

        print("从上到下开始扫描...")

        steps = int(scan_range / step_size)
        for i in range(steps):
            # 向下移动一小步
            tcp_pos[1] = start_z - i * step_size
            self.duco_cobot.servoj_pose(tcp_pos, self.vel, self.acc, '', '', '', True)
            time.sleep(pause_time)

            # 读取前向激光传感器
            sensor_data = self.get_sensor_data()
            dist = sensor_data["front"]
            if dist > 0:
                scan_data.append((tcp_pos[1], dist))  # 记录当前高度和距离值
                print(f"scan y={tcp_pos[1]:.3f}m, front={dist:.3f}m")

        print("扫描完成，开始检测突变边缘...")

        edge_positions = []
        for i in range(1, len(scan_data)):
            prev = scan_data[i - 1][1]
            curr = scan_data[i][1]
            if abs(curr - prev) > min_jump_threshold:
                y_pos = scan_data[i][0]
                edge_positions.append(y_pos)
                print(f"检测到突变边缘在 y={y_pos:.3f}m")

        if len(edge_positions) >= 2:
            top_edge = edge_positions[0]
            bottom_edge = edge_positions[-1]
            center_y = (top_edge + bottom_edge) / 2
            print(f"中心位置位于 y = {(center_y):.3f}m")

            # 计算目标末端位置
            center_pos = list(self.duco_cobot.get_tcp_pose())
            center_pos[1] = center_y

            # 补偿喷嘴与传感器之间的偏移（如果你有）
            # center_pos[2] += 0.02  # 举例：末端前移 2cm

            self.duco_cobot.servoj_pose(center_pos, self.vel, self.acc, '', '', '', True)
            print(f"机械臂已移动到中心位置：{center_pos}")

        else:
            print("未能检测到两个明显边缘，可能钢梁异常或测距异常。")


    # 自动喷涂，边走边喷
    def auto_paint_sync(self):
        print("-----进入自动模式-----")
        self.autopaint_flag = True
        time.sleep(0.1)  # 防止和退出冲突
        self.front_sensor_history.clear()
        v2 = 0.0  # 初始化前后速度
        cur_time = time.time()
        last_time = cur_time

        while self.autopaint_flag:
            sensor_data = self.get_sensor_data()
            tcp_pos = self.duco_cobot.get_tcp_pose()
            now = time.time()
            dt = now - last_time
            last_time = now

            if sensor_data["front"] == -1:
                print("传感器数据异常无法启动自动程序！")
                self.autopaint_flag = False
                self.duco_cobot.speed_stop(True)
                break

            side_count = 0
            side_count_threshold = 7 
            while self.anticrash_left != 0 and sensor_data["left"] < self.anticrash_left:
                v2 = self.auto_vel * 2
                self.duco_cobot.speedl([0, 0, -v2, 0, 0, 0], self.acc, -1, False)
                time.sleep(0.1)
                sensor_data = self.get_sensor_data()
                side_count += 1
                if side_count > side_count_threshold:
                    print("可能是个梁！")
                    default_pos = self.duco_cobot.get_tcp_pose()
                    self.duco_cobot.servoj_pose(self.safe_pos, self.vel * 1.5, self.acc, '', '', '', True)
                    time.sleep(0.1)
                    sensor_data = self.get_sensor_data()
                    # 检测是否通过梁
                    while self.anticrash_up != 0 and sensor_data["up"] < self.anticrash_up:
                        sensor_data = self.get_sensor_data()
                        time.sleep(0.05)
                    # 通过梁后，回到下降前的最后位置
                    self.duco_cobot.servoj_pose(default_pos, self.vel, self.acc, '', '', '', True)
                    break
            
            # PID with filter
            v2 = 0.0  # x轴默认速度为0
            if self.anticrash_front != 0:
                # 1. 数据滤波
                raw_front_dist = sensor_data["front"]
                if raw_front_dist > 0:  # 确保是有效读数
                    self.front_sensor_history.append(raw_front_dist)

                if len(self.front_sensor_history) > 0:
                    filtered_front_dist = sum(self.front_sensor_history) / len(self.front_sensor_history)

                    # 2. 控制死区
                    target_dist = self.anticrash_front
                    deadband_threshold = DEADZONE  # 单位: mm, 可根据实际情况调整
                    error = filtered_front_dist - target_dist

                    # 3. PID计算 (仅在死区外)
                    if abs(error) > deadband_threshold:
                        v2 = self.pid_z.compute(target_dist, filtered_front_dist, dt)
                        # 限制最大速度
                        v2 = max(min(v2, 0.15), -0.15)  

            print("v2: %f" % v2)
            self.duco_cobot.speedl([0, 0, v2, 0, 0, 0], self.acc * 0.9, -1, False)
        print("-----退出自动模式-----")

    # 自动喷涂，车辆不动机械臂动
    def auto_paint_interval(self):
        self.autopaint_flag = True
        time.sleep(0.1)  # 防止和退出冲突
        self.front_sensor_history.clear()
        v0 = self.auto_vel
        v2 = 0.0  # 初始化前后速度
        cur_time = time.time()
        last_time = cur_time

        while self.autopaint_flag:
            sensor_data = self.get_sensor_data()
            tcp_pos = self.duco_cobot.get_tcp_pose()
            now = time.time()
            dt = now - last_time
            last_time = now
            # 防撞保护
            if (self.anticrash_left != 0 and sensor_data["left"] < self.anticrash_left) or tcp_pos[1] > 1:
                print("jobs done")
                self.duco_cobot.speed_stop(True)
                break
            # PID with filter
            v2 = 0.0  # x轴默认速度为0
            if self.anticrash_front != 0:
                # 1. 数据滤波
                raw_front_dist = sensor_data["front"]
                if raw_front_dist > 0:  # 确保是有效读数
                    self.front_sensor_history.append(raw_front_dist)

                if len(self.front_sensor_history) > 0:
                    filtered_front_dist = sum(self.front_sensor_history) / len(self.front_sensor_history)

                    # 2. 控制死区
                    target_dist = self.anticrash_front
                    deadband_threshold = 10  # 单位: mm, 可根据实际情况调整
                    error = filtered_front_dist - target_dist

                    # 3. PID计算 (仅在死区外)
                    if abs(error) > deadband_threshold:
                        v2 = self.pid_z.compute(target_dist, filtered_front_dist, dt)
                        # 限制最大速度
                        v2 = max(min(v2, 0.1), -0.1) 

            print("v2: %f" % v2)
            task_id = self.duco_cobot.speedl([-v0, 0, v2, 0, 0, 0], self.acc, -1, False)

            if now - cur_time > 100:
                self.duco_cobot.speed_stop(True)
                print("Timeout.")
                break

    def run(self):
        print("等待移动到初始位置...")
        # self.duco_cobot.movej2(self.init_pos, 2*self.vel, self.acc, 0, True)
        self.duco_cobot.servoj_pose(self.init_pos, self.vel, self.acc, '', '', '', True)
        print("移动到初始位置: %s" % self.init_pos)
        time.sleep(1)
        
        try:
            while self.sysrun and not rospy.is_shutdown():
                key_input = self.get_key_input()
                sensor_data = self.get_sensor_data()
                self.duco_cobot.switch_mode(1)

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
                #寻找梁头的中间位置
                elif key_input.find:
                    if not self.emergency_stop_flag:
                        self.find_central_pos()
                #机械臂末端向  前
                elif key_input.x0:
                    self.duco_cobot.speedl([0, 0, v2, 0, 0, 0],self.acc ,-1, False)
                #机械臂末端向  后
                elif key_input.x1:
                    self.duco_cobot.speedl([0, 0, -v2, 0, 0, 0],self.acc ,-1, False)
                #机械臂末端向  右
                elif key_input.y1:
                    self.duco_cobot.speedl([v0, 0, 0, 0, 0, 0], self.acc, -1, False)
                #机械臂末端向  左
                elif key_input.y0: 
                    self.duco_cobot.speedl([-v0, 0, 0, 0, 0, 0], self.acc, -1, False)
                #机械臂末端向  上
                elif key_input.z1: 
                    self.duco_cobot.speedl([0, v1, 0, 0, 0, 0],self.acc ,-1, False)
                #机械臂末端向  下
                elif key_input.z0:
                    self.duco_cobot.speedl([0, -v1, 0, 0, 0, 0],self.acc ,-1, False)
                #初始化位置
                elif key_input.init:
                    self.duco_cobot.servoj_pose(self.init_pos, self.vel, self.acc, '', '', '', True)
                    print("移动到初始位置： %s" % self.init_pos)
                #维修位置
                elif key_input.serv:
                    self.duco_cobot.servoj_pose(self.serv_pos, self.vel, self.acc, '', '', '', True)
                    print("移动到维修位置： %s" % self.serv_pos)
                #机械臂末端转  pitch上
                elif key_input.rx0: 
                    self.duco_cobot.speedl([0, 0, 0, 0, 0, v5], self.acc, -1, False)
                    time.sleep(0.05)
                #机械臂末端转  pitch下
                elif key_input.rx1: 
                    self.duco_cobot.speedl([0, 0, 0, 0, 0, -v5], self.acc, -1, False)
                    time.sleep(0.05)
                #机械臂末端转  roll左
                elif key_input.ry0: 
                    self.duco_cobot.speedl([0, 0, 0, v3, 0, 0], self.acc, -1, False)
                #机械臂末端转  roll右
                elif key_input.ry1: 
                    self.duco_cobot.speedl([0, 0, 0, -v3, 0, 0], self.acc, -1, False)
                #机械臂末端转  yaw左
                elif key_input.rz0: 
                    self.duco_cobot.speedl([0, 0, 0, 0, v4, 0], self.acc, -1, False)
                #机械臂末端转  yaw右
                elif key_input.rz1: 
                    self.duco_cobot.speedl([0, 0, 0, 0, -v4, 0], self.acc, -1, False)

                # TODO 圆柱喷涂
                # elif btn_y:
                #     self.duco_cobot.switch_mode(1)
                #     auto_painter = CylinderAutoPaint(self.duco_cobot, self.init_pos, self.theta_deg, self.distance_to_cylinder, self.painting_width, self.vel, self.acc)
                #     auto_painter.auto_paint()

                else:
                    self.duco_cobot.speed_stop(False)
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.sysrun = False
            self.autopaint_flag = False

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            self.sysrun = False
            self.autopaint_flag = False
            return

        finally:
            self.emergency_thread.join()
