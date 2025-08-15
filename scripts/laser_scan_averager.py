#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
from collections import deque

class KalmanFilter:
    """单个雷达点的卡尔曼滤波器"""
    def __init__(self, dt=0.1, process_noise=0.02, measurement_noise=0.15):
        self.dt = dt
        self.initialized = False
        
        # 状态向量: [距离, 速度]
        self.x = np.zeros(2)
        
        # 状态转移矩阵
        self.F = np.array([[1, dt], [0, 1]])
        
        # 测量矩阵
        self.H = np.array([[1, 0]])
        
        # 过程噪声协方差
        self.Q = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * process_noise
        
        # 测量噪声协方差
        self.R = np.array([[measurement_noise]])
        
        # 估计误差协方差
        self.P = np.eye(2) * 10
        
        # 异常值检测阈值
        self.innovation_threshold = 3.0
        
    def predict(self):
        """预测步骤"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        """更新步骤，返回滤波后的值"""
        if not self.initialized:
            self.x[0] = measurement
            self.x[1] = 0
            self.initialized = True
            return measurement
            
        # 预测
        self.predict()
        
        # 处理无效测量值
        if np.isnan(measurement) or np.isinf(measurement):
            return self.x[0]  # 返回预测值
            
        # 计算创新
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # 异常值检测
        innovation_normalized = abs(y[0]) / np.sqrt(S[0, 0])
        if innovation_normalized < self.innovation_threshold:
            # 更新
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            I = np.eye(2)
            self.P = (I - K @ self.H) @ self.P
            
        return self.x[0]

class LaserScanAverager:
    def __init__(self):
        rospy.init_node('laser_scan_averager')

        self.buffer_size = 5  # 每0.5秒 取5帧 (10Hz)
        
        # 卡尔曼滤波器参数
        self.use_kalman = rospy.get_param('~use_kalman_filter', False)
        self.process_noise = rospy.get_param('~process_noise', 0.01)
        self.measurement_noise = rospy.get_param('~measurement_noise', 0.1)
        self.dt = 0.1  # 10Hz对应0.1秒
        
        rospy.loginfo(f"Using Kalman filter: {self.use_kalman}")
        if self.use_kalman:
            rospy.loginfo(f"Process noise: {self.process_noise}, Measurement noise: {self.measurement_noise}")

        # 为每个雷达维护 buffer、订阅和发布器
        self.radar_topics = {
            "left": {
                "scan_topic": "/left_radar/scan",
                "filtered_topic": "/left_radar/filtered_scan",
                "buffer": deque(maxlen=self.buffer_size),
                "kalman_filters": {}  # 为每个激光点维护一个卡尔曼滤波器
            },
            "main": {
                "scan_topic": "/main_radar/scan",
                "filtered_topic": "/main_radar/filtered_scan",
                "buffer": deque(maxlen=self.buffer_size),
                "kalman_filters": {}
            },
            "right": {
                "scan_topic": "/right_radar/scan",
                "filtered_topic": "/right_radar/filtered_scan",
                "buffer": deque(maxlen=self.buffer_size),
                "kalman_filters": {}
            }
        }

        # 分别创建订阅和发布器
        for key, radar in self.radar_topics.items():
            radar["sub"] = rospy.Subscriber(radar["scan_topic"], LaserScan, self.make_callback(key))
            radar["pub"] = rospy.Publisher(radar["filtered_topic"], LaserScan, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(0.5), self.publish_filtered_scan)

    def make_callback(self, key):
        def callback(msg):
            self.radar_topics[key]["buffer"].append(msg)
        return callback

    def apply_kalman_filter(self, radar_key, ranges):
        """对单个雷达的ranges应用卡尔曼滤波"""
        kalman_filters = self.radar_topics[radar_key]["kalman_filters"]
        filtered_ranges = []
        
        for i, range_val in enumerate(ranges):
            # 为每个激光点创建卡尔曼滤波器（如果不存在）
            if i not in kalman_filters:
                # 根据雷达类型调整噪声参数
                if radar_key == "main":
                    # 主雷达通常精度更高
                    measurement_noise = self.measurement_noise * 0.7
                else:
                    measurement_noise = self.measurement_noise
                    
                kalman_filters[i] = KalmanFilter(
                    dt=self.dt,
                    process_noise=self.process_noise,
                    measurement_noise=measurement_noise
                )
            
            # 应用卡尔曼滤波
            filtered_val = kalman_filters[i].update(range_val)
            filtered_ranges.append(filtered_val)
            
        return filtered_ranges

    def publish_filtered_scan(self, event):
        for key, radar in self.radar_topics.items():
            buffer = radar["buffer"]
            if len(buffer) < self.buffer_size:
                continue  # 数据不够，不处理

            if self.use_kalman:
                # 使用卡尔曼滤波器处理最新的扫描数据
                latest_scan = buffer[-1]
                filtered_ranges = self.apply_kalman_filter(key, latest_scan.ranges)
                
                # 对intensities也可以应用简单的滤波（如果存在）
                if latest_scan.intensities:
                    # 对intensities使用简单的移动平均
                    avg_intensities = np.mean([np.array(scan.intensities) for scan in buffer], axis=0)
                    filtered_intensities = avg_intensities.tolist()
                else:
                    filtered_intensities = []
                    
            else:
                # 使用原来的滑动平均方法
                avg_ranges = np.mean([np.array(scan.ranges) for scan in buffer], axis=0)
                filtered_ranges = avg_ranges.tolist()
                
                avg_intensities = np.mean([np.array(scan.intensities) if scan.intensities else np.zeros(len(scan.ranges)) for scan in buffer], axis=0)
                filtered_intensities = avg_intensities.tolist()

            # 创建滤波后的扫描消息
            filtered_scan = LaserScan()
            latest_scan = buffer[-1]
            
            # 复制header和配置信息
            filtered_scan.header = latest_scan.header
            filtered_scan.angle_min = latest_scan.angle_min
            filtered_scan.angle_max = latest_scan.angle_max
            filtered_scan.angle_increment = latest_scan.angle_increment
            filtered_scan.time_increment = latest_scan.time_increment
            filtered_scan.scan_time = latest_scan.scan_time
            filtered_scan.range_min = latest_scan.range_min
            filtered_scan.range_max = latest_scan.range_max
            
            # 设置滤波后的数据
            filtered_scan.ranges = filtered_ranges
            filtered_scan.intensities = filtered_intensities

            radar["pub"].publish(filtered_scan)
            
    def adjust_noise_for_painted_surface(self, increase_factor=2.0):
        """动态调整噪声参数以适应涂漆表面"""
        rospy.loginfo(f"Adjusting noise parameters for painted surface (factor: {increase_factor})")
        
        for radar_key, radar in self.radar_topics.items():
            for kalman_filter in radar["kalman_filters"].values():
                kalman_filter.R *= increase_factor
                
    def reset_noise_parameters(self):
        """重置噪声参数到默认值"""
        rospy.loginfo("Resetting noise parameters to default values")
        
        for radar_key, radar in self.radar_topics.items():
            for kalman_filter in radar["kalman_filters"].values():
                if radar_key == "main":
                    measurement_noise = self.measurement_noise * 0.7
                else:
                    measurement_noise = self.measurement_noise
                kalman_filter.R = np.array([[measurement_noise]])

if __name__ == '__main__':
    try:
        # 可以通过ROS参数配置滤波器
        # rosrun your_package laser_scan_averager.py _use_kalman_filter:=true _process_noise:=0.01 _measurement_noise:=0.1
        
        averager = LaserScanAverager()
        
        # 示例：如何在运行时调整参数
        # 可以通过服务调用或其他方式触发
        # averager.adjust_noise_for_painted_surface(2.0)  # 涂漆表面时增加噪声
        # averager.reset_noise_parameters()  # 重置到默认值
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass