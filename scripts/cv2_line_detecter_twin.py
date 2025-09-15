#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import matplotlib.pyplot as plt
from collections import deque
from config import *
import threading

from duco_control_pkg.msg import LineInfo, LineDetectionArray 

class SeparateRadarLineDetector:
    def __init__(self):
        rospy.init_node('separate_radar_line_detector')
        self.debug_mode = DEBUG_MODE
        # Parameters for image conversion
        self.image_size = 800  # 适中的图像尺寸
        self.max_range = 1.0      # 适中的范围
        self.resolution = self.max_range / (self.image_size / 2)
        
        # 圆形处理范围参数
        self.processing_radius_meters = 1.0  # 处理半径（米），只处理此范围内的数据
        self.processing_radius_pixels = int(self.processing_radius_meters / self.resolution)  # 转换为像素
        
        # Enhanced Hough line detection parameters (保持原有准确的参数)
        self.hough_threshold = 25
        self.min_line_length = 40
        self.max_line_gap = 40
        
        # Edge detection parameters
        self.canny_low = 10
        self.canny_high = 90
        self.gaussian_kernel = 3
        
        # Stability parameters
        self.temporal_buffer_size = 3  # 减少从5到3
        self.line_id_counter = 0
        
        # Line tracking and stability parameters
        self.position_threshold = 0.5
        self.angle_threshold = 20
        self.stability_requirement = 2  # 减少从3到2，更快达到稳定
        self.max_line_age = 5
        
        # Advanced filtering parameters
        self.min_line_length_meters = 0.15
        self.max_line_length_meters = 8.0
        self.angle_tolerance_deg = 75
        self.density_threshold = 0.6
        
        # 雷达间距参数（可调节）
        self.radar_separation = abs(LEFT_RADAR_OFFSET[0]) + abs(RIGHT_RADAR_OFFSET[0])  # 两雷达间距离，默认1.0米
        
        # 双雷达独立配置（相对于中央点的位置）
        self.radar_configs = {
            'left': {
                'topic': '/left_radar/filtered_scan',
                'frame_id': 'center_point',  # 统一使用中央点坐标系
                'position': {'x': 0.0, 'y': -self.radar_separation/2, 'angle': 0.0},  # 左雷达位置
                'data': None,
                'timestamp': None,
                'line_history': deque(maxlen=self.temporal_buffer_size),
                'stable_lines': [],
                'line_id_counter': 0
            },
            'right': {
                'topic': '/right_radar/filtered_scan',
                'frame_id': 'center_point',  # 统一使用中央点坐标系
                'position': {'x': 0.0, 'y': self.radar_separation/2, 'angle': 0.0},   # 右雷达位置
                'data': None,
                'timestamp': None,
                'line_history': deque(maxlen=self.temporal_buffer_size),
                'stable_lines': [],
                'line_id_counter': 0
            }
        }
        
        # 数据同步参数
        self.data_lock = threading.Lock()
        
        # Subscribers and publishers for each radar
        self.left_scan_sub = rospy.Subscriber(
            self.radar_configs['left']['topic'], 
            LaserScan, 
            lambda msg: self.radar_callback(msg, 'left')
        )
        self.right_scan_sub = rospy.Subscriber(
            self.radar_configs['right']['topic'], 
            LaserScan, 
            lambda msg: self.radar_callback(msg, 'right')
        )
        
        # 为每个雷达创建独立的发布器
        self.left_marker_pub = rospy.Publisher('/left_radar/detected_lines', MarkerArray, queue_size=10)
        self.left_scan_points_pub = rospy.Publisher('/left_radar/scan_points', Marker, queue_size=10)
        self.left_debug_pub = rospy.Publisher('/left_radar/debug_lines', MarkerArray, queue_size=10)
        
        self.right_marker_pub = rospy.Publisher('/right_radar/detected_lines', MarkerArray, queue_size=10)
        self.right_scan_points_pub = rospy.Publisher('/right_radar/scan_points', Marker, queue_size=10)
        self.right_debug_pub = rospy.Publisher('/right_radar/debug_lines', MarkerArray, queue_size=10)
        
        # 为两线连接功能添加发布器
        self.connection_line_pub = rospy.Publisher('/dual_radar/connection_line', MarkerArray, queue_size=10)
        
        self.line_info_pub = rospy.Publisher('/twin_radar/line_detection_info', LineDetectionArray, queue_size=10)
        
        # 打印初始化信息
        rospy.loginfo(f"Twin Radar Line Detector initialized")

    def radar_callback(self, scan_msg, radar_name):
        """雷达数据回调函数"""
        with self.data_lock:
            self.radar_configs[radar_name]['data'] = scan_msg
            self.radar_configs[radar_name]['timestamp'] = rospy.Time.now()
            
        # 独立处理该雷达数据
        self.process_single_radar(scan_msg, radar_name)

    def transform_to_center_coordinates(self, x_local, y_local, radar_name):
        """将雷达本地坐标转换为中央点坐标系"""
        radar_pos = self.radar_configs[radar_name]['position']
        
        # 应用位置偏移变换
        x_center = x_local + radar_pos['x']
        y_center = y_local + radar_pos['y']
        
        return x_center, y_center

    def polar_to_cartesian(self, ranges, angles):
        """Convert polar coordinates to cartesian coordinates"""
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return x, y

    def cartesian_to_image_coords(self, x, y):
        """Convert cartesian coordinates to image pixel coordinates"""
        center = self.image_size // 2
        img_x = center + (x / self.resolution).astype(int)
        img_y = center - (y / self.resolution).astype(int)
        img_x = np.clip(img_x, 0, self.image_size - 1)
        img_y = np.clip(img_y, 0, self.image_size - 1)
        return img_x, img_y

    def image_coords_to_cartesian(self, img_x, img_y):
        """Convert image pixel coordinates back to cartesian coordinates"""
        center = self.image_size // 2
        x = (img_x - center) * self.resolution
        y = (center - img_y) * self.resolution
        return x, y

    def create_enhanced_occupancy_image(self, scan_points):
        """从单个雷达的点云数据创建占用栅格图像"""
        if len(scan_points[0]) == 0:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8), ([], [])
        
        x_coords = scan_points[0]
        y_coords = scan_points[1]
        
        img_x, img_y = self.cartesian_to_image_coords(x_coords, y_coords)
        
        # 创建基础占用图像
        occupancy_img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # 只处理圆形范围内的点
        center = self.image_size // 2
        for i in range(len(img_x)):
            # 计算点到中心的距离（像素）
            dist_from_center = math.sqrt((img_x[i] - center)**2 + (img_y[i] - center)**2)
            
            # 只保留在指定半径内的点
            if dist_from_center <= self.processing_radius_pixels:
                occupancy_img[img_y[i], img_x[i]] = 255
        
        # 多尺度形态学操作提高线条连续性
        # 小核处理细节
        kernel_small = np.ones((2, 2), np.uint8)
        occupancy_img = cv2.morphologyEx(occupancy_img, cv2.MORPH_CLOSE, kernel_small)
        
        # 中等核连接临近点
        kernel_medium = np.ones((3, 3), np.uint8)
        occupancy_img = cv2.dilate(occupancy_img, kernel_medium, iterations=1)
        
        # 大核处理主要结构连续性
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occupancy_img = cv2.morphologyEx(occupancy_img, cv2.MORPH_CLOSE, kernel_large)
        
        return occupancy_img, (x_coords, y_coords)

    def detect_lines_multi_scale(self, image):
        """Multi-scale line detection for better stability"""
        # Apply different scales of Gaussian blur
        blurred_fine = cv2.GaussianBlur(image, (3, 3), 0)
        blurred_coarse = cv2.GaussianBlur(image, (7, 7), 0)
        
        # Edge detection at multiple scales
        edges_fine = cv2.Canny(blurred_fine, self.canny_low, self.canny_high)
        edges_coarse = cv2.Canny(blurred_coarse, self.canny_low // 2, self.canny_high // 2)
        
        # Combine edges
        edges_combined = cv2.bitwise_or(edges_fine, edges_coarse)
        
        # Multiple Hough transforms with different parameters
        lines_strict = cv2.HoughLinesP(edges_combined, 
                                     rho=1, 
                                     theta=np.pi/180, 
                                     threshold=self.hough_threshold + 10,
                                     minLineLength=self.min_line_length + 10,
                                     maxLineGap=self.max_line_gap - 5)
        
        lines_loose = cv2.HoughLinesP(edges_combined, 
                                    rho=1, 
                                    theta=np.pi/180, 
                                    threshold=self.hough_threshold,
                                    minLineLength=self.min_line_length,
                                    maxLineGap=self.max_line_gap)
        
        # Combine results
        all_lines = []
        if lines_strict is not None:
            all_lines.extend(lines_strict)
        if lines_loose is not None:
            all_lines.extend(lines_loose)
        
        return all_lines, edges_combined

    def calculate_line_properties(self, x1, y1, x2, y2, radar_name):
        """Calculate comprehensive line properties in center coordinate system"""
        # 转换图像坐标到本地笛卡尔坐标
        start_x_local, start_y_local = self.image_coords_to_cartesian(x1, y1)
        end_x_local, end_y_local = self.image_coords_to_cartesian(x2, y2)
        
        # 转换到中央点坐标系
        start_x, start_y = self.transform_to_center_coordinates(start_x_local, start_y_local, radar_name)
        end_x, end_y = self.transform_to_center_coordinates(end_x_local, end_y_local, radar_name)
        
        length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        angle = math.atan2(end_y - start_y, end_x - start_x)
        
        # Normalize angle to [0, pi] for consistency
        angle_normalized = angle % math.pi
        
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        distance = math.sqrt(mid_x**2 + mid_y**2)
        
        return {
            'start_point': (start_x, start_y),
            'end_point': (end_x, end_y),
            'midpoint': (mid_x, mid_y),
            'length': length,
            'angle_rad': angle_normalized,
            'angle_deg': math.degrees(angle_normalized),
            'distance_from_origin': distance,
            'radar_source': radar_name
        }

    def validate_line_with_scan_data(self, line_props, scan_points):
        """Validate line by checking point density along the line"""
        if len(scan_points[0]) == 0:
            return False
        
        start = np.array(line_props['start_point'])
        end = np.array(line_props['end_point'])
        line_vec = end - start
        line_length = np.linalg.norm(line_vec)
        
        if line_length < self.min_line_length_meters:
            return False
        
        # Check how many scan points are close to the line
        scan_points_array = np.column_stack(scan_points)
        
        # Calculate distance from each scan point to the line
        points_on_line = 0
        tolerance = 0.1  # 10cm tolerance
        
        for point in scan_points_array:
            point_vec = point - start
            projection_length = np.dot(point_vec, line_vec) / line_length
            
            # Check if projection is within line segment
            if 0 <= projection_length <= line_length:
                projection_point = start + (projection_length / line_length) * line_vec
                distance_to_line = np.linalg.norm(point - projection_point)
                
                if distance_to_line <= tolerance:
                    points_on_line += 1
        
        # Calculate density
        expected_points = max(1, int(line_length / 0.05))  # Expected points every 5cm
        density = points_on_line / expected_points
        
        return density >= self.density_threshold

    def cluster_similar_lines(self, lines):
        """Advanced clustering of similar lines"""
        if not lines:
            return []
        
        clusters = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            
            cluster = [line1]
            used[i] = True
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if used[j]:
                    continue
                
                if self.are_lines_similar(line1['properties'], line2['properties']):
                    cluster.append(line2)
                    used[j] = True
            
            clusters.append(cluster)
        
        return clusters

    def are_lines_similar(self, props1, props2):
        """Check if two lines are similar enough to be merged"""
        # Angle similarity (considering 0/180 degree wrap-around)
        angle_diff = abs(props1['angle_deg'] - props2['angle_deg'])
        angle_diff = min(angle_diff, 180 - angle_diff)
        
        # Position similarity
        pos_diff = math.sqrt((props1['midpoint'][0] - props2['midpoint'][0])**2 + 
                           (props1['midpoint'][1] - props2['midpoint'][1])**2)
        
        # Parallel lines check (similar angle, different position)
        parallel_check = angle_diff < self.angle_threshold and pos_diff < self.position_threshold * 2
        
        # Collinear lines check (similar position and angle)
        collinear_check = angle_diff < self.angle_threshold and pos_diff < self.position_threshold
        
        return parallel_check or collinear_check

    def merge_line_cluster(self, cluster):
        """Merge a cluster of similar lines using the longest line as base"""
        if len(cluster) == 1:
            return cluster[0]
        
        # Find the longest line in the cluster as the base
        longest_line = max(cluster, key=lambda x: x['properties']['length'])
        
        if len(cluster) == 2:
            # For two lines, extend the longest one to cover both
            other_line = cluster[0] if cluster[1] == longest_line else cluster[1]
            return self.extend_line_to_cover(longest_line, other_line)
        
        # For more than 2 lines, collect all endpoints and fit
        all_endpoints = []
        for line in cluster:
            all_endpoints.extend([line['properties']['start_point'], 
                                line['properties']['end_point']])
        
        # Remove duplicates and find the extreme points along the main direction
        points = np.array(all_endpoints)
        
        # Use the longest line's direction as reference
        base_start = np.array(longest_line['properties']['start_point'])
        base_end = np.array(longest_line['properties']['end_point'])
        base_direction = base_end - base_start
        base_direction = base_direction / np.linalg.norm(base_direction)
        
        # Project all points onto the base line direction
        projections = []
        for point in points:
            point_vec = point - base_start
            projection_scalar = np.dot(point_vec, base_direction)
            projections.append(projection_scalar)
        
        # Find the extreme projections
        min_proj = min(projections)
        max_proj = max(projections)
        
        # Calculate new endpoints
        new_start = base_start + min_proj * base_direction
        new_end = base_start + max_proj * base_direction
        
        # Convert to image coordinates
        img_coords = self.cartesian_to_image_coords(
            np.array([new_start[0], new_end[0]]), 
            np.array([new_start[1], new_end[1]])
        )
        
        # Calculate properties
        props = self.calculate_line_properties(img_coords[0][0], img_coords[1][0], 
                                             img_coords[0][1], img_coords[1][1], 'unknown')
        
        return {
            'image_coords': (img_coords[0][0], img_coords[1][0], img_coords[0][1], img_coords[1][1]),
            'properties': props
        }
    
    def extend_line_to_cover(self, base_line, other_line):
        """Extend base line to cover the other line"""
        # Get all four endpoints
        points = np.array([
            base_line['properties']['start_point'],
            base_line['properties']['end_point'],
            other_line['properties']['start_point'],
            other_line['properties']['end_point']
        ])
        
        # Use base line direction
        base_start = np.array(base_line['properties']['start_point'])
        base_end = np.array(base_line['properties']['end_point'])
        direction = base_end - base_start
        direction = direction / np.linalg.norm(direction)
        
        # Project all points onto the line
        projections = []
        for point in points:
            point_vec = point - base_start
            projection_scalar = np.dot(point_vec, direction)
            projections.append(projection_scalar)
        
        # Find extreme points
        min_proj = min(projections)
        max_proj = max(projections)
        
        new_start = base_start + min_proj * direction
        new_end = base_start + max_proj * direction
        
        # Convert to image coordinates
        img_coords = self.cartesian_to_image_coords(
            np.array([new_start[0], new_end[0]]), 
            np.array([new_start[1], new_end[1]])
        )
        
        props = self.calculate_line_properties(img_coords[0][0], img_coords[1][0], 
                                             img_coords[0][1], img_coords[1][1], 'unknown')
        
        return {
            'image_coords': (img_coords[0][0], img_coords[1][0], img_coords[0][1], img_coords[1][1]),
            'properties': props
        }

    def track_lines_temporally(self, current_lines, radar_name):
        """Track lines across multiple frames for stability"""
        # Add current detection to history
        self.radar_configs[radar_name]['line_history'].append(current_lines)
        
        if len(self.radar_configs[radar_name]['line_history']) < self.stability_requirement:
            rospy.loginfo(f"{radar_name} radar: Not enough history yet ({len(self.radar_configs[radar_name]['line_history'])}/{self.stability_requirement})")
            return []  # Not enough history yet
        
        # Find consistently detected lines
        stable_candidates = []
        
        for current_line in current_lines:
            consistency_count = 1  # Current frame
            
            # Check consistency across history
            for historical_frame in list(self.radar_configs[radar_name]['line_history'])[:-1]:
                for historical_line in historical_frame:
                    if self.are_lines_similar(current_line['properties'], 
                                            historical_line['properties']):
                        consistency_count += 1
                        break
            
            # If line appears consistently, it's stable
            if consistency_count >= self.stability_requirement:
                current_line['stability_score'] = consistency_count
                stable_candidates.append(current_line)
        
        return stable_candidates

    def update_stable_lines(self, new_stable_lines, radar_name):
        """Update the stable line list with temporal filtering"""
        stable_lines = self.radar_configs[radar_name]['stable_lines']
        
        # Age existing stable lines
        for line in stable_lines:
            line['age'] = line.get('age', 0) + 1
        
        # Remove old lines
        stable_lines[:] = [line for line in stable_lines 
                          if line['age'] < self.max_line_age]
        
        # Update or add new stable lines
        for new_line in new_stable_lines:
            # Find matching existing line
            matched = False
            for existing_line in stable_lines:
                if self.are_lines_similar(new_line['properties'], 
                                        existing_line['properties']):
                    # Update existing line with new detection
                    existing_line.update(new_line)
                    existing_line['age'] = 0
                    matched = True
                    break
            
            if not matched:
                # Add new stable line
                new_line['age'] = 0
                new_line['id'] = self.radar_configs[radar_name]['line_id_counter']
                self.radar_configs[radar_name]['line_id_counter'] += 1
                stable_lines.append(new_line)

    def publish_scan_points(self, scan_points, radar_name):
        """发布单个雷达的扫描点"""
        pub = (self.left_scan_points_pub if radar_name == 'left' 
               else self.right_scan_points_pub)
        
        marker = Marker()
        marker.header.frame_id = self.radar_configs[radar_name]['frame_id']
        marker.header.stamp = rospy.Time.now()
        marker.ns = f"{radar_name}_scan_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        
        # 设置不同颜色：左雷达绿色，右雷达蓝色
        if radar_name == 'left':
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        marker.color.a = 0.6
        
        # 添加点
        if len(scan_points[0]) > 0:
            for x, y in zip(scan_points[0], scan_points[1]):
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.0
                marker.points.append(point)
        
        pub.publish(marker)

    def publish_current_lines(self, current_lines, radar_name):
        """发布当前检测到的线（即使不稳定）"""
        pub = (self.left_marker_pub if radar_name == 'left' 
               else self.right_marker_pub)
        
        marker_array = MarkerArray()
        
        # Clear previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        for i, line in enumerate(current_lines):
            # Main line marker
            line_marker = Marker()
            line_marker.header.frame_id = self.radar_configs[radar_name]['frame_id']
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = f"{radar_name}_current_lines"
            line_marker.id = i
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            
            line_marker.scale.x = 0.06  # 稍细的线条表示不稳定
            
            # 使用不同颜色表示不稳定状态
            if radar_name == 'left':
                line_marker.color.r = 1.0
                line_marker.color.g = 0.5
                line_marker.color.b = 0.0  # 橙色
            else:
                line_marker.color.r = 0.5
                line_marker.color.g = 0.0
                line_marker.color.b = 1.0  # 紫色
            
            line_marker.color.a = 0.6  # 半透明
            
            # Add points
            start_point = Point()
            start_point.x = line['properties']['start_point'][0]
            start_point.y = line['properties']['start_point'][1]
            start_point.z = 0.0
            
            end_point = Point()
            end_point.x = line['properties']['end_point'][0]
            end_point.y = line['properties']['end_point'][1]
            end_point.z = 0.0
            
            line_marker.points = [start_point, end_point]
            marker_array.markers.append(line_marker)
        
        pub.publish(marker_array)

    def publish_stable_lines(self, radar_name):
        """Publish stable lines for a single radar"""
        stable_lines = self.radar_configs[radar_name]['stable_lines']
        pub = (self.left_marker_pub if radar_name == 'left' 
               else self.right_marker_pub)
        
        marker_array = MarkerArray()
        
        # Clear previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        for i, line in enumerate(stable_lines):
            # Main line marker with stability-based coloring
            line_marker = Marker()
            line_marker.header.frame_id = self.radar_configs[radar_name]['frame_id']
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = f"{radar_name}_stable_lines"
            line_marker.id = line.get('id', i)
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            
            # Color based on stability score and radar
            stability_score = line.get('stability_score', 1)
            max_stability = self.temporal_buffer_size
            
            line_marker.scale.x = 0.08
            
            # 根据雷达设置基础颜色
            if radar_name == 'left':
                line_marker.color.r = 1.0  # 左雷达红色系
                line_marker.color.g = min(1.0, stability_score / max_stability)
                line_marker.color.b = 0.0
            else:
                line_marker.color.r = 0.0
                line_marker.color.g = min(1.0, stability_score / max_stability)
                line_marker.color.b = 1.0  # 右雷达蓝色系
            
            line_marker.color.a = 0.8 + 0.2 * (stability_score / max_stability)
            
            # Add points
            start_point = Point()
            start_point.x = line['properties']['start_point'][0]
            start_point.y = line['properties']['start_point'][1]
            start_point.z = 0.0
            
            end_point = Point()
            end_point.x = line['properties']['end_point'][0]
            end_point.y = line['properties']['end_point'][1]
            end_point.z = 0.0
            
            line_marker.points = [start_point, end_point]
            marker_array.markers.append(line_marker)
            
            # Text marker with enhanced information
            text_marker = Marker()
            text_marker.header = line_marker.header
            text_marker.ns = f"{radar_name}_line_info"
            text_marker.id = line.get('id', i)
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = line['properties']['midpoint'][0]
            text_marker.pose.position.y = line['properties']['midpoint'][1]
            text_marker.pose.position.z = 0.3
            
            text_marker.scale.z = 0.15
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            # Enhanced text with stability info
            text_marker.text = (f"{radar_name[0].upper()}{line.get('id', i)}: "
                              f"{line['properties']['length']:.2f}m, "
                              f"{line['properties']['angle_deg']:.1f}°, "
                              f"S:{stability_score}")
            marker_array.markers.append(text_marker)
        
        pub.publish(marker_array)

    def process_single_radar(self, scan_msg, radar_name):
        """处理单个雷达的数据"""
        try:
            # 提取和预处理扫描数据
            ranges = np.array(scan_msg.ranges)
            angles = np.arange(scan_msg.angle_min, 
                              scan_msg.angle_max + scan_msg.angle_increment, 
                              scan_msg.angle_increment)
            
            # 确保ranges和angles长度一致
            min_len = min(len(ranges), len(angles))
            ranges = ranges[:min_len]
            angles = angles[:min_len]
            
            # 过滤有效数据
            valid_mask = ((ranges >= scan_msg.range_min) & 
                         (ranges <= scan_msg.range_max) & 
                         np.isfinite(ranges))
            ranges_valid = ranges[valid_mask]
            angles_valid = angles[valid_mask]
            
            if len(ranges_valid) == 0:
                print(f"No valid points for {radar_name} radar")
                return
            
            # 转换为笛卡尔坐标（雷达本地坐标系）
            x_coords, y_coords = self.polar_to_cartesian(ranges_valid, angles_valid)
            
            # 转换到中央点坐标系用于发布
            x_center, y_center = self.transform_to_center_coordinates(x_coords, y_coords, radar_name)
            scan_points = (x_coords, y_coords)  # 用于图像处理的本地坐标
            scan_points_center = (x_center, y_center)  # 用于发布的中央坐标
            
            
            # 创建占用栅格图像
            occupancy_img, _ = self.create_enhanced_occupancy_image(scan_points)
            
            # 多尺度线检测
            lines, edges = self.detect_lines_multi_scale(occupancy_img)
            
            if lines:
                # 计算所有检测到的线的属性
                current_lines = []
                for line in lines:
                    # 处理不同格式的线数据
                    if isinstance(line, np.ndarray) and line.ndim > 1:
                        x1, y1, x2, y2 = line[0]
                    else:
                        x1, y1, x2, y2 = line
                    
                    # 确保坐标是整数
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 跳过无效线
                    if x1 == x2 and y1 == y2:
                        continue
                        
                    props = self.calculate_line_properties(x1, y1, x2, y2, radar_name)
                    
                    # 基本长度检查
                    if (props['length'] >= self.min_line_length_meters and 
                        props['length'] <= self.max_line_length_meters):
                        
                        # 角度检查：过滤掉90度左右的线（垂直线）
                        angle = props['angle_deg']
                        if not (angle >= (90 - self.angle_tolerance_deg) and
                                angle <= (90 + self.angle_tolerance_deg)):
                            current_lines.append({
                                'image_coords': (x1, y1, x2, y2),
                                'properties': props
                            })
                
                
                # 聚类相似线条
                clusters = self.cluster_similar_lines(current_lines)
                merged_lines = []
                
                for cluster in clusters:
                    try:
                        merged_line = self.merge_line_cluster(cluster)
                        if merged_line is not None:
                            merged_lines.append(merged_line)
                    except Exception as e:
                        # print(f"Error merging cluster in {radar_name}: {e}")
                        # 回退到簇中最长的线
                        longest = max(cluster, key=lambda x: x['properties']['length'])
                        merged_lines.append(longest)
                
                # 时间跟踪以获得稳定性
                stable_lines = self.track_lines_temporally(merged_lines, radar_name)
                self.update_stable_lines(stable_lines, radar_name)
                
                # 调试输出
                rospy.loginfo(f"{radar_name} radar: {len(merged_lines)} lines detected, {len(stable_lines)} stable")
                
                # 发布结果（使用中央坐标系的点）
                self.publish_scan_points(scan_points_center, radar_name)
                self.publish_stable_lines(radar_name)
                
                # 临时发布当前检测到的线（即使不稳定）
                if len(merged_lines) > 0:
                    self.publish_current_lines(merged_lines, radar_name)
                
                if self.debug_mode: 
                    self.save_debug_images(occupancy_img, edges, merged_lines, radar_name)
                
                # 检查两线连接功能
                self.check_and_publish_connection_line()
                
                # 调试输出
                stable_count = len(self.radar_configs[radar_name]['stable_lines'])
                for i, line in enumerate(self.radar_configs[radar_name]['stable_lines']):
                    props = line['properties']
                    start = props['start_point']
                    end = props['end_point']

            else:
                if self.debug_mode:
                    print(f"No lines detected in {radar_name} radar data")
                self.publish_scan_points(scan_points_center, radar_name)
                # 仍然检查连接线（以防另一个雷达有检测结果）
                self.check_and_publish_connection_line()
            
        except Exception as e:
            rospy.logerr(f"Error in {radar_name} radar processing: {e}")
            import traceback
            traceback.print_exc()

    def check_and_publish_connection_line(self):
        """检查两个雷达是否都只检测到一条线，如果是则连接它们"""
        left_lines = self.radar_configs['left']['stable_lines']
        right_lines = self.radar_configs['right']['stable_lines']

        # 检查条件：两个雷达都只有一条稳定线
        if len(left_lines) == 1 and len(right_lines) == 1:
            left_line = left_lines[0]
            right_line = right_lines[0]
            
            # 获取两条线的中点
            left_midpoint = left_line['properties']['midpoint']
            right_midpoint = right_line['properties']['midpoint']

            # 计算连接线的中点和角度
            connection_midpoint = (
                (left_midpoint[0] + right_midpoint[0]) / 2,
                (left_midpoint[1] + right_midpoint[1]) / 2
            )
            
            # 计算连接线的角度
            dx = right_midpoint[0] - left_midpoint[0]
            dy = right_midpoint[1] - left_midpoint[1]
            connection_angle_rad = math.atan2(dy, dx)
            connection_angle_deg = math.degrees(connection_angle_rad)
            
            # 计算连接线长度
            connection_length = math.sqrt(dx**2 + dy**2)
            
            # 发布连接线可视化
            self.publish_connection_line(left_midpoint, right_midpoint, connection_midpoint, 
                                       connection_angle_deg, connection_length)
            
            connection_line_info = {
                'id': 999,  # 给连接线一个独特的ID
                'properties': {
                    'start_point': left_midpoint,  # 连接线的起点可以是左雷达的中心点
                    'end_point': right_midpoint,   # 连接线的终点可以是右雷达的中心点
                    'midpoint': connection_midpoint,
                    'length': connection_length,
                    'angle_rad': connection_angle_rad,
                    'angle_deg': connection_angle_deg,
                    'distance_from_origin': math.sqrt(connection_midpoint[0]**2 + connection_midpoint[1]**2),
                    'radar_source': 'connection' # 标识来源是连接线
                }
            }
            all_lines_to_publish = []
            for line in left_lines:
                # 复制一份以避免修改原始数据
                temp_line = line.copy()
                if 'id' not in temp_line:
                    temp_line['id'] = self.radar_configs['left']['line_id_counter']
                    self.radar_configs['left']['line_id_counter'] += 1
                all_lines_to_publish.append(temp_line)
                if self.debug_mode:
                    rospy.loginfo("===yes in left")
            for line in right_lines:
                # 复制一份以避免修改原始数据
                temp_line = line.copy()
                if 'id' not in temp_line:
                    temp_line['id'] = self.radar_configs['right']['line_id_counter']
                    self.radar_configs['right']['line_id_counter'] += 1
                all_lines_to_publish.append(temp_line)
                if self.debug_mode:
                    rospy.loginfo("===yes in right")

                
            all_lines_to_publish.append(connection_line_info) # 添加连接线
            
            self.publish_debug_line_info(all_lines_to_publish)
            
        else:
            # 清除连接线可视化
            self.clear_connection_line()

    def publish_connection_line(self, left_point, right_point, midpoint, angle_deg, length):
        """发布连接线的可视化标记"""
        marker_array = MarkerArray()
        
        # 连接线标记
        line_marker = Marker()
        line_marker.header.frame_id = "center_point"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "connection_line"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        line_marker.scale.x = 0.12  # 较粗的线条
        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0  # 黄色
        line_marker.color.a = 1.0
        
        # 添加连接线的两个端点
        start_point = Point()
        start_point.x = left_point[0]
        start_point.y = left_point[1]
        start_point.z = 0.0
        
        end_point = Point()
        end_point.x = right_point[0]
        end_point.y = right_point[1]
        end_point.z = 0.0
        
        line_marker.points = [start_point, end_point]
        marker_array.markers.append(line_marker)
        
        # 中点标记
        center_marker = Marker()
        center_marker.header = line_marker.header
        center_marker.ns = "connection_center"
        center_marker.id = 1
        center_marker.type = Marker.SPHERE
        center_marker.action = Marker.ADD
        
        center_marker.pose.position.x = midpoint[0]
        center_marker.pose.position.y = midpoint[1]
        center_marker.pose.position.z = 0.0
        
        center_marker.scale.x = 0.2
        center_marker.scale.y = 0.2
        center_marker.scale.z = 0.2
        
        center_marker.color.r = 1.0
        center_marker.color.g = 0.5
        center_marker.color.b = 0.0  # 橙色
        center_marker.color.a = 1.0
        
        marker_array.markers.append(center_marker)
        
        # 信息文本标记
        text_marker = Marker()
        text_marker.header = line_marker.header
        text_marker.ns = "connection_info"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose.position.x = midpoint[0]
        text_marker.pose.position.y = midpoint[1]
        text_marker.pose.position.z = 0.5
        
        text_marker.scale.z = 0.3
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        
        text_marker.text = f"Connection: {length:.2f}m, {angle_deg:.1f}°"
        marker_array.markers.append(text_marker)
        
        # 端点标记
        for i, point in enumerate([left_point, right_point]):
            point_marker = Marker()
            point_marker.header = line_marker.header
            point_marker.ns = "connection_endpoints"
            point_marker.id = 3 + i
            point_marker.type = Marker.CYLINDER
            point_marker.action = Marker.ADD
            
            point_marker.pose.position.x = point[0]
            point_marker.pose.position.y = point[1]
            point_marker.pose.position.z = 0.0
            
            point_marker.scale.x = 0.1
            point_marker.scale.y = 0.1
            point_marker.scale.z = 0.2
            
            if i == 0:  # 左端点
                point_marker.color.r = 0.0
                point_marker.color.g = 1.0
                point_marker.color.b = 0.0
            else:  # 右端点
                point_marker.color.r = 0.0
                point_marker.color.g = 0.0
                point_marker.color.b = 1.0
            point_marker.color.a = 0.8
            
            marker_array.markers.append(point_marker)
        
        self.connection_line_pub.publish(marker_array)

    def clear_connection_line(self):
        """清除连接线可视化"""
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.connection_line_pub.publish(marker_array)

    def save_debug_images(self, occupancy_img, edges, lines, radar_name):
        """保存调试图像显示检测过程"""
        # 创建可视化图像
        vis_img = cv2.cvtColor(occupancy_img, cv2.COLOR_GRAY2BGR)
        
        # 绘制检测到的线条
        for line in lines:
            x1, y1, x2, y2 = line['image_coords']
            cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # 绘制起点和终点
            cv2.circle(vis_img, (int(x1), int(y1)), 3, (0, 255, 0), -1)
            cv2.circle(vis_img, (int(x2), int(y2)), 3, (255, 0, 0), -1)
        
        # 添加雷达位置标记（在图像坐标系中显示）
        center = self.image_size // 2
        
        # 雷达原点（相对于雷达本身的位置）
        cv2.circle(vis_img, (center, center), 8, (255, 255, 255), -1)
        cv2.putText(vis_img, radar_name[0].upper(), (center-8, center+8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 显示中央点的相对位置
        radar_pos = self.radar_configs[radar_name]['position']
        center_img_x, center_img_y = self.cartesian_to_image_coords(
            np.array([-radar_pos['x']]), np.array([-radar_pos['y']])
        )
        cv2.circle(vis_img, (int(center_img_x[0]), int(center_img_y[0])), 6, (255, 255, 0), -1)
        cv2.putText(vis_img, 'C', (int(center_img_x[0])-5, int(center_img_y[0])+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 添加坐标轴
        # X轴（红色）
        cv2.arrowedLine(vis_img, (center, center), (center + 50, center), (0, 0, 255), 2)
        cv2.putText(vis_img, 'X', (center + 55, center + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Y轴（绿色）
        cv2.arrowedLine(vis_img, (center, center), (center, center - 50), (0, 255, 0), 2)
        cv2.putText(vis_img, 'Y', (center + 5, center - 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加标题和信息
        title = f"{radar_name.capitalize()} Radar - Lines: {len(lines)} (Separation: {self.radar_separation:.1f}m)"
        cv2.putText(vis_img, title, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示稳定线数量
        stable_count = len(self.radar_configs[radar_name]['stable_lines'])
        info_text = f"Stable: {stable_count}"
        cv2.putText(vis_img, info_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示坐标系信息
        coord_info = f"Radar pos: ({self.radar_configs[radar_name]['position']['x']:.1f}, {self.radar_configs[radar_name]['position']['y']:.1f})"
        cv2.putText(vis_img, coord_info, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 显示处理范围信息
        range_info = f"Processing radius: {self.processing_radius_meters:.1f}m ({self.processing_radius_pixels}px)"
        cv2.putText(vis_img, range_info, (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 在图像上绘制处理范围边界
        cv2.circle(vis_img, (center, center), self.processing_radius_pixels, (0, 255, 0), 2)
        
        # 可选：保存图像（如果不需要可以注释掉）
        # cv2.imwrite(f'/tmp/{radar_name}_radar_occupancy.png', occupancy_img)
        # cv2.imwrite(f'/tmp/{radar_name}_radar_edges.png', edges)
        # cv2.imwrite(f'/tmp/{radar_name}_radar_lines.png', vis_img)
        
        # 显示图像
        window_name = f"{radar_name.capitalize()} Radar Detected Lines"
        cv2.imshow(window_name, vis_img)
        
        # 创建占用图像的可视化版本，显示处理范围
       # occupancy_vis = cv2.cvtColor(occupancy_img, cv2.COLOR_GRAY2BGR)
       # cv2.circle(occupancy_vis, (center, center), self.processing_radius_pixels, (0, 255, 0), 2)
       # cv2.putText(occupancy_vis, f"Processing Range: {self.processing_radius_meters:.1f}m", 
       #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       # 
       # occupancy_window = f"{radar_name.capitalize()} Radar Occupancy (with range)"
       # cv2.imshow(occupancy_window, occupancy_vis)
        cv2.waitKey(1)

    def get_radar_status(self, radar_name):
        """获取单个雷达状态信息"""
        with self.data_lock:
            data_status = "OK" if self.radar_configs[radar_name]['data'] is not None else "NO_DATA"
            timestamp = self.radar_configs[radar_name]['timestamp']
            stable_lines = len(self.radar_configs[radar_name]['stable_lines'])
            
            age_info = "FRESH"
            if timestamp is not None:
                age = (rospy.Time.now() - timestamp).to_sec()
                if age > 1.0:
                    age_info = f"OLD({age:.1f}s)"
            else:
                age_info = "NO_TIME"
            
            return {
                'data': data_status,
                'age': age_info,
                'stable_lines': stable_lines
            }

    def get_all_radar_status(self):
        """获取所有雷达状态信息"""
        return {
            'left': self.get_radar_status('left'),
            'right': self.get_radar_status('right')
        }

    def publish_debug_line_info(self,st_line):
        """/twin_radar/line_detection_info"""
        line_detection_array_msg = LineDetectionArray()
        line_detection_array_msg.header.stamp = rospy.Time.now()
        line_detection_array_msg.header.frame_id = "radar_platform/twin_radar" # 或者你希望的 frame_id

        for i, line in enumerate(st_line):
            props = line['properties']
            start = props['start_point']
            end = props['end_point']

            # 创建 LineInfo 消息
            line_info = LineInfo()
            line_info.id = line.get('id', i)

            # 填充起点
            line_info.start_point.x = start[0]
            line_info.start_point.y = start[1]
            line_info.start_point.z = 0.0 # 假设是2D雷达，Z为0

            # 填充终点
            line_info.end_point.x = end[0]
            line_info.end_point.y = end[1]
            line_info.end_point.z = 0.0

            # 填充长度和角度
            line_info.length = props['length']
            line_info.angle_deg = props['angle_deg']
            line_info.distance = props['distance_from_origin']

            line_detection_array_msg.lines.append(line_info)

        
        self.line_info_pub.publish(line_detection_array_msg)

    def run(self):
        """主运行循环"""
        rospy.loginfo("Separate Radar Line Detector started")
        rospy.loginfo("Processing left and right radars independently...")
        
        # 定期状态检查
        rate = rospy.Rate(0.5)  # 0.5Hz，每2秒检查一次
        status_counter = 0
        while not rospy.is_shutdown():
            status_counter += 1
            rate.sleep()
            
if __name__ == '__main__':
    try:
        detector = SeparateRadarLineDetector()
        detector.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass