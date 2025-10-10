#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
import math
import threading
import time

class TripleRadarFrontDistance:
    def __init__(self, topic_name='/right_radar/filtered_scan'):
        """
        初始化三个雷达的前方距离读取器
        保持与原有DirectionalLaser类相同的接口
        Args:
            topic_name: 主雷达话题名称（保持向后兼容）
        """
        self.topic_name = topic_name
        self.radars = {
            'left': {
                'topic': '/left_radar/filtered_scan',
                'latest_scan': None,
                'last_scan_time': 0,
                'distances': {
                    'front': -1,
                    'up': -1,
                    'down': -1
                }
            },
            'right': {
                'topic': '/right_radar/filtered_scan',
                'latest_scan': None,
                'last_scan_time': 0,
                'distances': {
                    'front': -1,
                    'up': -1,
                    'down': -1
                }
            },
            'main': {
                'topic': '/main_radar/filtered_scan',
                'latest_scan': None,
                'last_scan_time': 0,
                'distances': {
                    'front': -1,
                    'up': -1,
                    'down': -1
                }
            }
        }
        
        self.scan_timeout = 1.0  # 1秒超时
        
        # 目标方向角（单位：弧度）- 保持"front": math.pi不变
        self.angles = {
            "front": math.pi,
            "up": -math.pi / 2,
            "down": math.pi / 2,
        }
        
        # 存储各个方向的最新距离值（保持向后兼容）
        self.distances = {
            "front": -1,
            "down": -1,
            "up": -1,
        }
        
        # 初始化ROS订阅者
        self.scan_subscribers = {}
        for radar_name, radar_info in self.radars.items():
            self.scan_subscribers[radar_name] = rospy.Subscriber(
                radar_info['topic'], 
                LaserScan, 
                lambda msg, name=radar_name: self.scan_callback(msg, name)
            )
            rospy.loginfo(f"订阅雷达话题: {radar_info['topic']}")
        
        # 启动处理线程
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        rospy.loginfo("TripleRadarFrontDistance 初始化完成")

    def scan_callback(self, scan, radar_name):
        """激光雷达数据回调函数"""
        self.radars[radar_name]['latest_scan'] = scan
        self.radars[radar_name]['last_scan_time'] = time.time()

    def _processing_loop(self):
        """处理循环，持续更新各个雷达的所有方向距离值"""
        while self.running and not rospy.is_shutdown():
            for radar_name in self.radars.keys():
                self._update_all_directions(radar_name)
            
            # 更新兼容性距离值（使用left雷达的数据）
            self._update_compatibility_distances()
            
            time.sleep(0.1)  # 100ms更新一次

    def _update_all_directions(self, radar_name):
        """更新指定雷达的所有方向距离值"""
        radar_info = self.radars[radar_name]
        scan = radar_info['latest_scan']
        
        if scan is None:
            for direction in radar_info['distances']:
                radar_info['distances'][direction] = -1
            return
            
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        ranges = scan.ranges
        num_ranges = len(ranges)
        window_size = 3  # 左右各取3个，总共7个点

        # 遍历所有方向
        for direction, target_angle in self.angles.items():
            # 把目标角度归一化到 [-π, π]
            angle = math.atan2(math.sin(target_angle), math.cos(target_angle))
            
            # 检查是否在扫描范围内
            if angle < scan.angle_min or angle > scan.angle_max:
                radar_info['distances'][direction] = -1
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
                radar_info['distances'][direction] = avg_dist
            else:
                radar_info['distances'][direction] = -1

    def _update_compatibility_distances(self):
        """更新兼容性距离值，使用left雷达的数据"""
        # 使用left雷达的数据来更新兼容性接口
        left_radar = self.radars['left']
        for direction in self.distances:
            self.distances[direction] = left_radar['distances'][direction]

    def get_distance(self, radar, direction):
        """
        获取指定雷达的指定方向的距离值
        Args:
            radar: 雷达名称 ("left", "right", "main")
            direction: 方向名称 ("front", "up", "down")
        Returns:
            float: 距离值（米），-1表示无效或超时
        """
        # 检查雷达名称
        if radar not in self.radars:
            rospy.logwarn(f"未知雷达: {radar}")
            return -1
            
        # 检查方向名称
        if direction not in self.angles:
            rospy.logwarn(f"未知方向: {direction}")
            return -1
            
        radar_info = self.radars[radar]
        
        # 检查数据是否超时
        if time.time() - radar_info['last_scan_time'] > self.scan_timeout:
            rospy.logwarn(f"雷达 {radar} 数据超时，最后更新时间: {radar_info['last_scan_time']}")
            return -1
            
        return radar_info['distances'][direction]

    def get_front_distance(self, radar_name):
        """
        获取指定雷达的前方距离值
        Args:
            radar_name: 雷达名称 ("left", "right", "main")
        Returns:
            float: 距离值（米），-1表示无效或超时
        """
        # 直接调用 get_distance 方法
        return self.get_distance(radar_name, "front")

    def get_all_front_distances(self):
        """
        获取所有雷达的前方距离值
        Returns:
            dict: 包含所有雷达前方距离值的字典
        """
        distances = {}
        for radar_name in self.radars.keys():
            distances[radar_name] = self.get_front_distance(radar_name)
        return distances

    def get_all_distances(self):
        """
        获取所有方向的距离值（保持向后兼容，使用left雷达数据）
        Returns:
            dict: 包含所有方向距离值的字典
        """
        # 检查数据是否超时（使用left雷达的时间）
        left_radar = self.radars['left']
        if time.time() - left_radar['last_scan_time'] > self.scan_timeout:
            rospy.logwarn(f"left雷达数据超时，最后更新时间: {left_radar['last_scan_time']}")
            return {direction: -1 for direction in self.distances}
            
        return self.distances.copy()

    def get_radar_all_distances(self, radar_name):
        """
        获取指定雷达的所有方向距离值
        Args:
            radar_name: 雷达名称 ("left", "right", "main")
        Returns:
            dict: 包含所有方向距离值的字典 {"front": x, "up": y, "down": z}
        """
        if radar_name not in self.radars:
            rospy.logwarn(f"未知雷达: {radar_name}")
            return {"front": -1, "up": -1, "down": -1}
            
        radar_info = self.radars[radar_name]
        
        # 检查数据是否超时
        if time.time() - radar_info['last_scan_time'] > self.scan_timeout:
            rospy.logwarn(f"雷达 {radar_name} 数据超时，最后更新时间: {radar_info['last_scan_time']}")
            return {"front": -1, "up": -1, "down": -1}
            
        return radar_info['distances'].copy()

    def is_data_valid(self, radar_name=None):
        """
        检查数据是否有效（未超时）
        Args:
            radar_name: 雷达名称，如果为None则检查left雷达（保持向后兼容）
        Returns:
            bool: True表示数据有效，False表示数据超时
        """
        if radar_name is None:
            # 检查left雷达（保持向后兼容）
            left_radar = self.radars['left']
            return time.time() - left_radar['last_scan_time'] <= self.scan_timeout
        else:
            # 检查指定雷达
            if radar_name not in self.radars:
                return False
            return time.time() - self.radars[radar_name]['last_scan_time'] <= self.scan_timeout

    def shutdown(self):
        """关闭服务"""
        self.running = False
        for subscriber in self.scan_subscribers.values():
            if hasattr(subscriber, 'unregister'):
                subscriber.unregister()

# 为了保持向后兼容，创建一个别名
DirectionalLaser = TripleRadarFrontDistance

# 如果直接运行此文件，则启动独立节点
if __name__ == "__main__":
    rospy.init_node('triple_radar_front_distance')
    
    # 创建实例
    triple_radar = TripleRadarFrontDistance()
    
    try:
        # 独立运行时的循环
        rate = rospy.Rate(2)  # 2Hz
        while not rospy.is_shutdown():
            print("\n" + "="*50)
            
            # 显示每个雷达的所有方向距离
            for radar_name in ['left', 'right', 'main']:
                if triple_radar.is_data_valid(radar_name):
                    print(f"\n=== {radar_name.upper()} 雷达 - 所有方向 ===")
                    distances = triple_radar.get_radar_all_distances(radar_name)
                    for direction, distance in distances.items():
                        if distance > 0:
                            print(f"  {direction}: {distance:.3f} m")
                        else:
                            print(f"  {direction}: 无效")
                else:
                    print(f"\n=== {radar_name.upper()} 雷达 ===")
                    print("  数据无效或超时")
            
            # 显示所有雷达的前方距离
            front_distances = triple_radar.get_all_front_distances()
            print("\n=== 所有雷达前方距离 ===")
            for radar_name, distance in front_distances.items():
                if distance > 0:
                    print(f"  {radar_name}: {distance:.3f} m")
                else:
                    print(f"  {radar_name}: 无效")
            
            # 测试新的 get_distance 方法
            print("\n=== 测试 get_distance 方法 ===")
            left_up = triple_radar.get_distance('left', 'up')
            right_front = triple_radar.get_distance('right', 'front')
            main_down = triple_radar.get_distance('main', 'down')
            print(f"  left雷达 up方向: {left_up:.3f} m" if left_up > 0 else "  left雷达 up方向: 无效")
            print(f"  right雷达 front方向: {right_front:.3f} m" if right_front > 0 else "  right雷达 front方向: 无效")
            print(f"  main雷达 down方向: {main_down:.3f} m" if main_down > 0 else "  main雷达 down方向: 无效")
            
            print("="*50)
            rate.sleep()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        triple_radar.shutdown()
