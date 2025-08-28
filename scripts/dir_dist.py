#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
import math
import threading
import time

class DirectionalLaser:
    def __init__(self, topic_name='/right_radar/filtered_scan'):
        """
        初始化方向性激光雷达读取器
        Args:
            topic_name: 激光雷达话题名称
        """
        self.topic_name = topic_name
        self.latest_scan = None
        self.last_scan_time = 0
        self.scan_timeout = 1.0  # 1秒超时
        
        # 目标方向角（单位：弧度）
        self.angles = {
            "front": math.pi,
            "down": -math.pi / 2,
            "up": math.pi / 2,
        }
        
        # 存储各个方向的最新距离值
        self.distances = {
            "front": -1,
            "down": -1,
            "up": -1,
        }
        
        # 初始化ROS订阅者
        self.scan_subscriber = rospy.Subscriber(topic_name, LaserScan, self.scan_callback)
        
        # 启动处理线程
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        rospy.loginfo(f"DirectionalLaser 初始化完成，订阅话题: {topic_name}")

    def scan_callback(self, scan):
        """激光雷达数据回调函数"""
        self.latest_scan = scan
        self.last_scan_time = time.time()

    def _processing_loop(self):
        """处理循环，持续更新各个方向的距离值"""
        while self.running and not rospy.is_shutdown():
            if self.latest_scan is not None:
                self._update_distances()
            time.sleep(0.1)  # 100ms更新一次

    def _update_distances(self):
        """更新各个方向的距离值"""
        scan = self.latest_scan
        if scan is None:
            return
            
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        ranges = scan.ranges
        num_ranges = len(ranges)
        window_size = 3  # 左右各取3个，总共7个点

        for direction, target_angle in self.angles.items():
            # 把目标角度归一化到 [-π, π]
            angle = math.atan2(math.sin(target_angle), math.cos(target_angle))
            
            # 检查是否在扫描范围内
            if angle < scan.angle_min or angle > scan.angle_max:
                self.distances[direction] = -1
                continue

            index = int((angle - angle_min) / angle_increment)

            # 获取窗口内的索引范围
            start = max(0, index - window_size)
            end = min(num_ranges, index + window_size + 1)
            window_ranges = ranges[start:end]

            # 去掉 inf 和 0 的无效值
            valid_ranges = [r for r in window_ranges if not math.isinf(r) and r > 0.01]

            if valid_ranges:
                avg_dist = sum(valid_ranges) / len(valid_ranges)
                self.distances[direction] = avg_dist
            else:
                self.distances[direction] = -1

    def get_distance(self, direction):
        """
        获取指定方向的距离值
        Args:
            direction: 方向名称 ("front", "down", "up")
        Returns:
            float: 距离值（米），-1表示无效或超时
        """
        if direction not in self.distances:
            rospy.logwarn(f"未知方向: {direction}")
            return -1
            
        # 检查数据是否超时
        if time.time() - self.last_scan_time > self.scan_timeout:
            rospy.logwarn(f"激光雷达数据超时，最后更新时间: {self.last_scan_time}")
            return -1
            
        return self.distances[direction]

    def get_all_distances(self):
        """
        获取所有方向的距离值
        Returns:
            dict: 包含所有方向距离值的字典
        """
        # 检查数据是否超时
        if time.time() - self.last_scan_time > self.scan_timeout:
            rospy.logwarn(f"激光雷达数据超时，最后更新时间: {self.last_scan_time}")
            return {direction: -1 for direction in self.distances}
            
        return self.distances.copy()

    def is_data_valid(self):
        """
        检查数据是否有效（未超时）
        Returns:
            bool: True表示数据有效，False表示数据超时
        """
        return time.time() - self.last_scan_time <= self.scan_timeout

    def shutdown(self):
        """关闭服务"""
        self.running = False
        if hasattr(self, 'scan_subscriber'):
            self.scan_subscriber.unregister()

# 如果直接运行此文件，则启动独立节点
if __name__ == "__main__":
    rospy.init_node('directional_laser_reader')
    
    # 创建实例
    directional_laser = DirectionalLaser()
    
    try:
        # 独立运行时的循环
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            if directional_laser.is_data_valid():
                distances = directional_laser.get_all_distances()
                print("当前距离值:")
                for direction, distance in distances.items():
                    if distance > 0:
                        print(f"  {direction}: {distance:.3f} m")
                    else:
                        print(f"  {direction}: 无效")
            else:
                print("激光雷达数据无效或超时")
            print("---")
            rate.sleep()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        directional_laser.shutdown()