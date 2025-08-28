#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Pose
from std_msgs.msg import Header, Float64MultiArray
import sensor_msgs.point_cloud2 as pc2

from config import *   

class RadarObstacleAvoidance:
    def __init__(self):
        """
        机械臂末端三雷达避障系统初始化 - 基于末端位姿的简化版本
        """
        rospy.init_node('radar_obstacle_avoidance_node')
        
        # 可调节参数 - 雷达相对机械臂末端的位置偏移
        # 坐标系：X轴前后(后为正)，Y轴左右(右为正)，Z轴上下(上为正)
        self.main_radar_offset = np.array([MAIN_RADAR_OFFSET[0], MAIN_RADAR_OFFSET[1], MAIN_RADAR_OFFSET[2]])
        self.left_radar_offset = np.array([LEFT_RADAR_OFFSET[0], LEFT_RADAR_OFFSET[1], LEFT_RADAR_OFFSET[2]])
        self.right_radar_offset = np.array([RIGHT_RADAR_OFFSET[0], RIGHT_RADAR_OFFSET[1], RIGHT_RADAR_OFFSET[2]])

        # 雷达旋转角度配置（弧度）
        # 主雷达横置（水平扫描），左右雷达竖置（垂直扫描）
        self.main_radar_rotation = np.array([MAIN_RADAR_OFFSET[3], MAIN_RADAR_OFFSET[4], MAIN_RADAR_OFFSET[5]])  # 无旋转 [RX, RY, RZ]
        self.left_radar_rotation = np.array([LEFT_RADAR_OFFSET[3], LEFT_RADAR_OFFSET[4], LEFT_RADAR_OFFSET[5]])  # 绕Y轴旋转90度
        self.right_radar_rotation = np.array([RIGHT_RADAR_OFFSET[3], RIGHT_RADAR_OFFSET[4], RIGHT_RADAR_OFFSET[5]])  # 绕Y轴旋转-90度
        
        # 订阅机械臂末端位姿数据
        # 数据格式：[X, Y, Z, RX, RY, RZ] - 位置+欧拉角
        self.arm_pose_sub = rospy.Subscriber('/Duco_state', Float64MultiArray, self.arm_pose_callback)

        
        # 订阅三个雷达数据
        self.main_radar_sub = rospy.Subscriber('/main_radar/scan', LaserScan, self.main_radar_callback)
        self.left_radar_sub = rospy.Subscriber('/left_radar/scan', LaserScan, self.left_radar_callback)
        self.right_radar_sub = rospy.Subscriber('/right_radar/scan', LaserScan, self.right_radar_callback)
        
        # 发布融合后的点云
        self.pointcloud_pub = rospy.Publisher('/obstacle_avoidance/pointcloud', PointCloud2, queue_size=1)
        
        # 存储最新数据
        self.current_arm_pose = None  # [X, Y, Z, RX, RY, RZ]
        self.main_radar_data = None
        self.left_radar_data = None
        self.right_radar_data = None
        
        # 数据时间戳
        self.last_pose_time = rospy.Time(0)
        self.last_main_time = rospy.Time(0)
        self.last_left_time = rospy.Time(0)
        self.last_right_time = rospy.Time(0)
        
        # 性能参数
        self.max_data_age = rospy.Duration(rospy.get_param('~max_data_age', 2))
        fusion_freq = rospy.get_param('~fusion_frequency', 20.0)
        
        # 定时器用于定期融合发布点云
        self.fusion_timer = rospy.Timer(rospy.Duration(1.0/fusion_freq), self.timer_fusion_callback)
        self.last_publish_time = rospy.Time(0)
        
        rospy.loginfo("雷达避障系统初始化完成")
    
    def arm_pose_callback(self, msg):
        """
        机械臂末端位姿回调
        数据格式：[X, Y, Z, RX, RY, RZ]
        """
        if len(msg.data) >= 6:
            arm_pose_raw = np.array(msg.data[:10])
            # self.current_arm_pose = [arm_pose_raw[0], arm_pose_raw[1], arm_pose_raw[2], arm_pose_raw[3]+np.pi/2, arm_pose_raw[4], arm_pose_raw[5]-np.pi/2]
            self.current_arm_pose = [arm_pose_raw[4], arm_pose_raw[5], arm_pose_raw[6], arm_pose_raw[7], arm_pose_raw[8], arm_pose_raw[9]]
            self.last_pose_time = rospy.Time.now()
    
    def main_radar_callback(self, msg):
        """主雷达数据回调"""
        self.main_radar_data = msg
        self.last_main_time = msg.header.stamp
    
    def left_radar_callback(self, msg):
        """左雷达数据回调"""
        self.left_radar_data = msg
        self.last_left_time = msg.header.stamp
    
    def right_radar_callback(self, msg):
        """右雷达数据回调"""
        self.right_radar_data = msg
        self.last_right_time = msg.header.stamp
    
    def euler_to_rotation_matrix(self, rx, ry, rz):
        """
        欧拉角转旋转矩阵
        
        Args:
            rx, ry, rz: 绕X、Y、Z轴的旋转角度（弧度）
        
        Returns:
            3x3旋转矩阵
        """
        # 绕X轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # 绕Y轴旋转
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # 绕Z轴旋转
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转：R = Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def create_transform_matrix(self, translation, rotation_euler):
        """
        创建4x4变换矩阵
        
        Args:
            translation: 平移向量 [x, y, z]
            rotation_euler: 欧拉角 [rx, ry, rz]
        
        Returns:
            4x4变换矩阵
        """
        rotation_matrix = self.euler_to_rotation_matrix(*rotation_euler)
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation
        
        return transform_matrix
    
    def laser_scan_to_pointcloud(self, scan_msg, radar_offset, radar_rotation, arm_pose):
        """
        将激光雷达扫描数据转换为机械臂底座坐标系下的点云
        
        Args:
            scan_msg: LaserScan消息
            radar_offset: 雷达相对末端的位置偏移 [x, y, z]
            radar_rotation: 雷达相对末端的旋转角度 [rx, ry, rz]
            arm_pose: 机械臂末端位姿 [X, Y, Z, RX, RY, RZ]
        
        Returns:
            points: 机械臂底座坐标系下的点云数组 (N, 3)
        """
        if scan_msg is None or arm_pose is None:
            return np.array([]).reshape(0, 3)
        
        points_base = []
        angle = scan_msg.angle_min
        
        # 机械臂末端的位置和姿态
        arm_position = arm_pose[:3]  # [X, Y, Z]
        arm_rotation = arm_pose[3:]  # [RX, RY, RZ]
        
        # 创建末端执行器的变换矩阵
        T_base_to_ee = self.create_transform_matrix(arm_position, arm_rotation)
        
        # 创建雷达相对末端执行器的变换矩阵
        T_ee_to_radar = self.create_transform_matrix(radar_offset, radar_rotation)
        
        # 组合变换：底座 -> 末端 -> 雷达
        T_base_to_radar = T_base_to_ee @ T_ee_to_radar
        
        for i, distance in enumerate(scan_msg.ranges):
            if scan_msg.range_min <= distance <= scan_msg.range_max:
                # 在雷达坐标系中的点（雷达扫描平面）
                x_radar = distance * np.cos(angle)
                y_radar = distance * np.sin(angle)
                z_radar = 0.0
                
                # 齐次坐标
                point_radar = np.array([x_radar, y_radar, z_radar, 1.0])
                
                # 变换到底座坐标系
                point_base = T_base_to_radar @ point_radar
                
                points_base.append(point_base[:3])
            
            angle += scan_msg.angle_increment
        
        return np.array(points_base)
    
    def timer_fusion_callback(self, event):
        """定时器回调函数，定期融合所有雷达数据并发布"""
        self.process_radar_fusion()
    
    def process_radar_fusion(self):
        """处理三个雷达数据融合"""
        current_time = rospy.Time.now()
        
        # 检查机械臂位姿数据是否有效
        if (self.current_arm_pose is None or 
            (current_time - self.last_pose_time) > self.max_data_age):
            rospy.logwarn_throttle(2.0, "机械臂位姿数据无效或过期")
            return
        
        # 避免过于频繁的发布
        if (current_time - self.last_publish_time).to_sec() < 0.02:
            return
        
        # 收集所有有效的雷达点云
        all_points_list = []
        valid_radars = []
        
        # 处理主雷达
        if (self.main_radar_data is not None and 
            (current_time - self.last_main_time) < self.max_data_age):
            main_points = self.laser_scan_to_pointcloud(
                self.main_radar_data, 
                self.main_radar_offset, 
                self.main_radar_rotation,
                self.current_arm_pose
            )
            if len(main_points) > 0:
                all_points_list.append(main_points)
                valid_radars.append("main")
        
        # 处理左雷达
        if (self.left_radar_data is not None and 
            (current_time - self.last_left_time) < self.max_data_age):
            left_points = self.laser_scan_to_pointcloud(
                self.left_radar_data, 
                self.left_radar_offset, 
                self.left_radar_rotation,
                self.current_arm_pose
            )
            if len(left_points) > 0:
                all_points_list.append(left_points)
                valid_radars.append("left")
        
        # 处理右雷达
        if (self.right_radar_data is not None and 
            (current_time - self.last_right_time) < self.max_data_age):
            right_points = self.laser_scan_to_pointcloud(
                self.right_radar_data, 
                self.right_radar_offset, 
                self.right_radar_rotation,
                self.current_arm_pose
            )
            if len(right_points) > 0:
                all_points_list.append(right_points)
                valid_radars.append("right")
        
        # 合并所有点云
        if all_points_list:
            all_points = np.vstack(all_points_list)
            self.publish_pointcloud(all_points, current_time, valid_radars)
        else:
            # 发布空点云
            empty_points = np.array([]).reshape(0, 3)
            self.publish_pointcloud(empty_points, current_time, [])
        
        self.last_publish_time = current_time
    
    def publish_pointcloud(self, points, stamp, valid_radars=None):
        """发布点云消息"""
        # 创建点云消息
        header = Header()
        header.stamp = stamp
        header.frame_id = "radar_link"  # 固定为底座坐标系
        
        # 创建点云数据
        if len(points) > 0:
            pointcloud_msg = pc2.create_cloud_xyz32(header, points)
        else:
            # 空点云
            pointcloud_msg = PointCloud2()
            pointcloud_msg.header = header
        
        # 发布点云
        self.pointcloud_pub.publish(pointcloud_msg)
        
        # 调试信息
        # if valid_radars:
        #     rospy.loginfo_throttle(2.0, f"发布避障点云，共{len(points)}个点，来自雷达: {valid_radars}")
    
    def update_radar_positions(self, left_offset=None, right_offset=None):
        """
        更新左右雷达位置参数
        坐标系：X轴前后(后为正)，Y轴左右(右为正)，Z轴上下(上为正)
        
        Args:
            left_offset: 左雷达新的偏移位置 [x, y, z]
            right_offset: 右雷达新的偏移位置 [x, y, z]
        """
        if left_offset is not None:
            self.left_radar_offset = np.array(left_offset)
            rospy.loginfo(f"左雷达位置更新为: 前后={self.left_radar_offset[0]:.3f}m, "
                         f"左右={self.left_radar_offset[1]:.3f}m, 上下={self.left_radar_offset[2]:.3f}m")
        
        if right_offset is not None:
            self.right_radar_offset = np.array(right_offset)
            rospy.loginfo(f"右雷达位置更新为: 前后={self.right_radar_offset[0]:.3f}m, "
                         f"左右={self.right_radar_offset[1]:.3f}m, 上下={self.right_radar_offset[2]:.3f}m")
    
    def run(self):
        """运行避障系统"""
        rospy.loginfo("雷达避障系统开始运行...")
        rospy.spin()

if __name__ == '__main__':
    try:
        obstacle_avoidance = RadarObstacleAvoidance()
        
        # 示例：动态调节雷达位置（工件确定后）
        # obstacle_avoidance.update_radar_positions(
        #     left_offset=[0.02, -0.05, 0.03],   # 后方2cm，左侧5cm，上方3cm
        #     right_offset=[0.02, 0.05, 0.03]    # 后方2cm，右侧5cm，上方3cm
        # )
        
        obstacle_avoidance.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("雷达避障系统关闭")