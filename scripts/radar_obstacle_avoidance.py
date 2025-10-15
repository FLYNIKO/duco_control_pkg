#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from re import X
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
# 导入自定义消息类型（请根据你的包名修改）
from duco_control_pkg.msg import ObstacleFlags

from config import *

class EndEffectorObstacleAvoidance:
    def __init__(self):
        rospy.init_node('end_effector_obstacle_avoidance', anonymous=True)
        
        # 参数配置
        self.setup_parameters()
        
        # 机械臂末端状态
        self.end_effector_pose = None
        self.pose_updated = False
        
        # 订阅话题
        self.pointcloud_sub = rospy.Subscriber(
            '/obstacle_avoidance/pointcloud', 
            PointCloud2, 
            self.pointcloud_callback
        )
        
        self.duco_state_sub = rospy.Subscriber(
            '/Duco_state',
            Float64MultiArray,
            self.duco_state_callback
        )
        
        # 发布避障标志话题（使用自定义消息）
        self.obstacle_flags_pub = rospy.Publisher(
            '/obstacle_flags', 
            ObstacleFlags,
            queue_size=1
        )
        
        # 统计信息
        self.frame_count = 0
        self.last_log_time = rospy.Time.now()
        
        rospy.loginfo("机械臂末端避障节点已启动")
    
    def setup_parameters(self):
        """设置参数"""
        # 体素大小 (5cm)
        self.voxel_size = rospy.get_param('~voxel_size', 0.05)
        
        # 区域划分阈值（相对于末端位置）
        self.threshold_L = OB_THRESHOLD_L # 左区阈值
        self.threshold_R = OB_THRESHOLD_R # 右区阈值
        self.threshold_M = OB_THRESHOLD_M # 中区阈值
        self.threshold_U = OB_THRESHOLD_U # 上区阈值（相对末端向上）
        self.threshold_D = OB_THRESHOLD_D # 下区阈值（相对末端向下）
        
        # 前后分界阈值（X坐标）
        self.threshold_front_mid = OB_THRESHOLD_FRONT_MID  # 前后分界线
        self.threshold_mid_rear = OB_THRESHOLD_MID_REAR
        
        # 前方区域检测范围（末端前方）
        self.forward_min = OB_FORWARD_MIN    # 末端前方最小距离
        self.forward_max = OB_FORWARD_MAX    # 末端前方最大距离
        
        # 安全距离
        self.safe_distance = OB_SAFE_DISTANCE  # 相对末端的安全距离
        
        # 处理频率控制
        self.max_process_rate = 10.0
        self.last_process_time = rospy.Time.now()
        
        # 调试模式
        self.debug_mode = False
        
        # self.log_parameters()
    
    def log_parameters(self):
        """记录参数信息"""
        rospy.loginfo("=== 机械臂末端避障参数配置 ===")
        rospy.loginfo(f"体素大小: {self.voxel_size * 100:.1f}cm")
        rospy.loginfo(f"安全距离: {self.safe_distance * 100:.1f}cm") 
        rospy.loginfo(f"区域阈值（相对末端）:")
        rospy.loginfo(f"  左区 (Y > {self.threshold_L:.2f}m)")
        rospy.loginfo(f"  右区 (Y < {self.threshold_R:.2f}m)")
        rospy.loginfo(f"  中区 (|Y| <= {self.threshold_M:.2f}m)")
        rospy.loginfo(f"  上区 (Z > {self.threshold_U:.2f}m)")
        rospy.loginfo(f"  下区 (Z < {self.threshold_D:.2f}m)")
        rospy.loginfo(f"  前后分界线 (X = {self.threshold_front_rear:.2f}m)")
        rospy.loginfo(f"  前方范围: {self.forward_min:.2f}m ~ {self.forward_max:.2f}m")
        rospy.loginfo("===============================")
    
    def duco_state_callback(self, msg):
        """机械臂状态回调函数"""
        try:
            if len(msg.data) < 10:
                rospy.logwarn("Duco_state数据长度不足")
                return
            
            # 提取末端位姿 [4][5][6][7][8][9] -> X,Y,Z,RX,RY,RZ
            self.end_effector_pose = {
                'position': np.array([msg.data[4], msg.data[5], msg.data[6]]),
                'orientation': np.array([msg.data[7], msg.data[8], msg.data[9]])  # 欧拉角
            }
            
            self.pose_updated = True
            
            # if self.debug_mode:
            #     pos = self.end_effector_pose['position']
            #     orient = self.end_effector_pose['orientation']
            #     rospy.loginfo_throttle(2.0, f"末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
            #                                f"姿态: [{orient[0]:.2f}, {orient[1]:.2f}, {orient[2]:.2f}]°")
        
        except Exception as e:
            rospy.logerr(f"解析机械臂状态失败: {e}")
    
    def pointcloud_callback(self, msg):
        """点云数据回调函数"""
        try:
            # 检查是否有末端位姿数据
            if not self.pose_updated or self.end_effector_pose is None:
                rospy.logwarn_throttle(5.0, "等待机械臂位姿数据...")
                return
            
            # 频率控制
            current_time = rospy.Time.now()
            time_diff = (current_time - self.last_process_time).to_sec()
            if time_diff < (1.0 / self.max_process_rate):
                return
            
            self.last_process_time = current_time
            self.frame_count += 1
            
            # 解析点云数据
            points = self.extract_points_from_pointcloud(msg)
            
            if len(points) == 0:
                self.publish_safe_flags()
                return
            
            # 转换到末端坐标系
            end_effector_points = self.transform_to_end_effector_frame(points)
            
            # 过滤相关区域的点
            filtered_points = self.filter_relevant_points(end_effector_points)
            
            # 体素化处理
            voxelized_points = self.voxelize_pointcloud(filtered_points)
            
            # 区域划分和障碍检测
            obstacle_results = self.detect_obstacles_by_region(voxelized_points)
            
            # 发布避障标志
            self.publish_obstacle_flags(obstacle_results)
            
            # 定期输出统计信息
            # self.log_statistics(current_time, len(points), len(voxelized_points))
            
        except Exception as e:
            rospy.logerr(f"处理点云数据时出错: {e}")
            self.publish_safe_flags()
    
    def extract_points_from_pointcloud(self, pointcloud_msg):
        """从PointCloud2消息中提取点坐标"""
        points = []
        
        try:
            for point in pc2.read_points(pointcloud_msg, 
                                       field_names=("x", "y", "z"), 
                                       skip_nans=True):
                points.append([point[0], point[1], point[2]])
        except Exception as e:
            rospy.logerr(f"解析点云数据失败: {e}")
            return np.array([])
        
        return np.array(points)
    
    def get_end_effector_transform_matrix(self):
        """获取末端执行器的变换矩阵"""
        position = self.end_effector_pose['position']
        orientation = self.end_effector_pose['orientation']  # 欧拉角(度)
        
        # 将欧拉角转换为弧度
        roll, pitch, yaw = np.radians(orientation)
        
        # 创建旋转矩阵（ZYX欧拉角顺序）
        rotation_matrix = tf_trans.euler_matrix(roll, pitch, yaw)[:3, :3]
        
        # 创建齐次变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position
        
        return transform_matrix
    
    def transform_to_end_effector_frame(self, points):
        """将点云从基座坐标系转换到末端执行器坐标系"""
        if len(points) == 0:
            return points
        
        # 获取末端执行器变换矩阵
        T_base_to_end = self.get_end_effector_transform_matrix()
        
        # 计算逆变换矩阵（从末端到基座的逆变换）
        T_end_to_base = np.linalg.inv(T_base_to_end)
        
        # 转换点云
        # 添加齐次坐标
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        
        # 应用变换
        transformed_points = (T_end_to_base @ points_homogeneous.T).T
        
        # 返回3D坐标
        return transformed_points[:, :3]
    
    def filter_relevant_points(self, end_effector_points):
        """过滤与末端相关的点"""
        if len(end_effector_points) == 0:
            return end_effector_points
        
        # 在末端坐标系中，过滤距离过远的点
        distances = np.linalg.norm(end_effector_points, axis=1)
        max_distance = 1.0  # 1米范围内
        
        # 过滤距离
        distance_mask = distances < max_distance
        
        # 过滤高度异常的点（相对末端）
        z_values = end_effector_points[:, 2]
        height_mask = (z_values > -0.5) & (z_values < 0.5)  # 末端上下50cm
        
        # 综合掩码
        valid_mask = distance_mask & height_mask
        
        filtered_points = end_effector_points[valid_mask]
        
        # if self.debug_mode and len(filtered_points) != len(end_effector_points):
        #     rospy.loginfo_throttle(5.0, f"过滤点云: {len(end_effector_points)} -> {len(filtered_points)}")
        
        return filtered_points
    
    def voxelize_pointcloud(self, points):
        """点云体素化处理"""
        if len(points) == 0:
            return np.array([])
        
        # 将点坐标转换为体素索引
        voxel_indices = np.floor(points / self.voxel_size).astype(int)
        
        # 使用集合去除重复体素
        unique_voxels = set()
        for voxel_idx in voxel_indices:
            unique_voxels.add(tuple(voxel_idx))
        
        # 转换回体素中心坐标
        voxelized_points = []
        for voxel_key in unique_voxels:
            voxel_center = (np.array(voxel_key) + 0.5) * self.voxel_size
            voxelized_points.append(voxel_center)
        
        return np.array(voxelized_points)
    
    def detect_obstacles_by_region(self, voxelized_points):
        """根据区域划分检测障碍（相对于末端执行器）"""
        results = {
            'left_front': False,
            'left_mid': False,
            'left_rear': False,
            'right_front': False,
            'right_mid': False,
            'right_rear': False,
            'center': False,
            'up': False,
            'down': False,
            'min_distances': {},
            'point_counts': {}
        }
        
        if len(voxelized_points) == 0:
            return results
        
        # 在末端坐标系中，计算距离
        distances = np.linalg.norm(voxelized_points, axis=1)
        
        # 提取坐标（相对于末端）
        x, y, z = voxelized_points[:, 0], voxelized_points[:, 1], voxelized_points[:, 2]
        
        # 区域划分（在末端坐标系中）
        regions = {
            'left_front': (y < self.threshold_L) & (x < self.threshold_front_mid),  # 左前区
            'left_mid': (y < self.threshold_L) & (x >= self.threshold_front_mid) & (x < self.threshold_mid_rear),  # 左中区
            'left_rear': (y < self.threshold_L) & (x >= self.threshold_mid_rear),   # 左后区

            'right_front': (y > self.threshold_R) & (x < self.threshold_front_mid),  # 右前区
            'right_mid': (y > self.threshold_R) & (x >= self.threshold_front_mid) & (x < self.threshold_mid_rear),  # 右前区
            'right_rear': (y > self.threshold_R) & (x >= self.threshold_mid_rear),  # 右后区
            'center': (np.abs(y) <= self.threshold_M) & (x < self.forward_min) & (x > self.forward_max) & (z <= self.threshold_U) & (z >= self.threshold_D),  # 末端前方中央
            'up': z > self.threshold_U,  # 末端上方
            'down': z < self.threshold_D  # 末端下方
        }
        
        for region_name, mask in regions.items():
            if np.any(mask):
                region_distances = distances[mask]
                min_distance = np.min(region_distances)
                point_count = np.sum(mask)
                
                results['min_distances'][region_name] = min_distance
                results['point_counts'][region_name] = point_count
                if region_name == 'center':
                    self.safe_distance = OB_SAFE_DISTANCE_F
                elif region_name == 'up' or region_name == 'down':
                    self.safe_distance = OB_SAFE_DISTANCE_F
                elif region_name == 'left_mid' or region_name == 'right_mid':
                    self.safe_distance = OB_SAFE_DISTANCE
                elif region_name == 'left_rear' or region_name == 'right_rear':
                    self.safe_distance = OB_SAFE_DISTANCE + 0.07
                else:
                    self.safe_distance = OB_SAFE_DISTANCE

                if min_distance < self.safe_distance:
                    results[region_name] = True
        
        return results
    
    def publish_obstacle_flags(self, results):
        """发布避障标志（使用自定义消息）"""
        # 创建ObstacleFlags消息
        msg = ObstacleFlags()
        
        # 设置各区域标志
        msg.left_front = results['left_front']
        msg.left_mid = results['left_mid']
        msg.left_rear = results['left_rear']
        msg.right_front = results['right_front']
        msg.right_mid = results['right_mid']
        msg.right_rear = results['right_rear']
        msg.center = results['center']
        msg.up = results['up']
        msg.down = results['down']
        
        # 设置时间戳
        msg.stamp = rospy.Time.now()
        
        # 设置安全距离
        msg.safe_distance = self.safe_distance
        
        # 发布消息
        self.obstacle_flags_pub.publish(msg)
        
        # 调试信息
        if self.debug_mode:
            active_regions = []
            region_names = {
                'left_front': '左前', 'left_mid': '左中前', 'left_rear': '左中后',
                'right_front': '右前', 'right_mid': '右中前', 'right_rear': '右中后',
                'center': '中', 'up': '上', 'down': '下'
            }
            
            for region, chinese_name in region_names.items():
                if results[region]:
                    active_regions.append(chinese_name)
            
            if active_regions and self.debug_mode:
                rospy.logwarn(f"末端附近检测到障碍: {', '.join(active_regions)}")
                for region, chinese_name in region_names.items():
                    if results[region] and region in results['min_distances']:
                        distance = results['min_distances'][region]
                        count = results['point_counts'][region]
                        rospy.logwarn(f"  {chinese_name}区: 最近{distance:.3f}m, {count}个障碍点")
            else:
                rospy.loginfo_throttle(10.0, "末端周围安全")
    
    def publish_safe_flags(self):
        """发布全部安全标志"""
        # 创建全部安全的ObstacleFlags消息
        msg = ObstacleFlags()
        msg.left_front = False
        msg.left_mid = False
        msg.left_rear = False
        msg.right_front = False
        msg.right_mid = False
        msg.right_rear = False
        msg.center = False
        msg.up = False
        msg.down = False
        msg.stamp = rospy.Time.now()
        msg.safe_distance = self.safe_distance
        
        self.obstacle_flags_pub.publish(msg)
    
    def log_statistics(self, current_time, original_points, voxel_points):
        """记录统计信息"""
        time_since_last_log = (current_time - self.last_log_time).to_sec()
        if time_since_last_log >= 10.0:  # 每10秒输出一次
            processing_rate = self.frame_count / time_since_last_log
            
            if self.end_effector_pose is not None:
                pos = self.end_effector_pose['position']
                rospy.loginfo(f"末端避障统计 - 频率: {processing_rate:.1f}Hz, "
                             f"末端位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                             f"处理点云: 原始{original_points} -> 体素{voxel_points}")
            
            self.frame_count = 0
            self.last_log_time = current_time
    
    def run(self):
        """运行节点"""
        rospy.loginfo("机械臂末端避障节点开始运行...")
        rospy.spin()

def main():
    try:
        obstacle_avoidance = EndEffectorObstacleAvoidance()
        obstacle_avoidance.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("机械臂末端避障节点正常关闭")
    except Exception as e:
        rospy.logerr(f"节点启动失败: {e}")

if __name__ == '__main__':
    main()