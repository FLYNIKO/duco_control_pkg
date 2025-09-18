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
from duco_control_pkg.msg import LineInfo, LineDetectionArray 

class StableRadarLineDetector:
    def __init__(self):
        rospy.init_node('stable_radar_line_detector')
        self.debug_mode = True
        # Parameters for image conversion
        self.image_size = 800
        self.max_range = 1.0
        self.resolution = self.max_range / (self.image_size / 2)
        
         # 圆形处理范围参数
        self.processing_radius_meters = 0.7  # 处理半径（米），只处理此范围内的数据
        self.processing_radius_pixels = int(self.processing_radius_meters / self.resolution)  # 转换为像素
        
        # Enhanced Hough line detection parameters - 平衡检测能力和精度
        self.hough_threshold = 25     # 适当提高阈值以减少误检
        self.min_line_length = 40      # 稍微提高最小线长要求
        self.max_line_gap = 50         # 适当减少间隙容忍度
        
        # Separate parameters for short lines (flanges)
        self.flange_hough_threshold = 14    # 适当提高阈值
        self.flange_min_line_length = 12    # 提高最小线长要求
        self.flange_max_line_gap = 20      # 减少间隙容忍度
        
        # Edge detection parameters
        self.canny_low = 8             # 适当提高低阈值
        self.canny_high = 90           # 适当提高高阈值
        self.gaussian_kernel = 3       # 保持小核以减少平滑
        
        # Stability parameters
        self.temporal_buffer_size = 5  # Number of frames to track
        self.line_history = deque(maxlen=self.temporal_buffer_size)
        self.stable_lines = []
        self.line_id_counter = 0
        
        # Line tracking and stability parameters
        self.position_threshold = 0.50   # 放宽位置阈值（暴力拟合
        self.angle_threshold = 30       # 放宽角度阈值（暴力拟合
        self.stability_requirement = 2  # 降低稳定性要求，只需要2帧
        self.max_line_age = 6          # 增加最大线龄
        
        # Advanced filtering parameters
        self.min_line_length_meters = 0.25  # 适当提高最小线长要求
        self.min_line_deg = 90
        self.min_flange_length_meters = 0.06 # 适当提高最小法兰长度
        self.max_line_length_meters = 2.2   # 适当减少最大线长
        self.density_threshold = 0.5    # 适当提高密度阈值

        # 如果还是有太多小线段，可以增加最小线长要求
        self.min_line_length_pixels_final = 25  # 最终过滤的最小像素长度
        
        # Subscriber and publishers
        self.scan_sub = rospy.Subscriber('/left_radar/filtered_scan', LaserScan, self.scan_callback)
        self.marker_pub = rospy.Publisher('/detected_Hs', MarkerArray, queue_size=10)
        self.scan_points_pub = rospy.Publisher('/radar_scan_Hs', Marker, queue_size=10)
        self.debug_pub = rospy.Publisher('/debug_Hs', MarkerArray, queue_size=10)
        self.line_info_pub = rospy.Publisher('/left_radar/H_detection_info', LineDetectionArray, queue_size=10)

        print("Stable Radar Line Detector initialized")
        print(f"Temporal buffer size: {self.temporal_buffer_size}")
        print(f"Stability requirement: {self.stability_requirement} frames")

    def polar_to_cartesian(self, ranges, angles):
        """Convert polar coordinates to cartesian coordinates"""
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return x, y

    def cartesian_to_image_coords(self, x, y):
        """Convert cartesian coordinates to image pixel coordinates"""
        center = self.image_size // 2
        img_x = center + (x / self.resolution).astype(int)
        img_y = center - (y / self.resolution).astype(int)  # 修复镜像问题：从 center - y 改为 center + y
        img_x = np.clip(img_x, 0, self.image_size - 1)
        img_y = np.clip(img_y, 0, self.image_size - 1)
        return img_x, img_y

    def image_coords_to_cartesian(self, img_x, img_y):
        """Convert image pixel coordinates back to cartesian coordinates"""
        center = self.image_size // 2
        x = (img_x - center) * self.resolution
        y = (center - img_y) * self.resolution
        return x, y

    def create_enhanced_occupancy_image(self, scan_msg):
        """Create enhanced occupancy image with multi-scale processing"""
        ranges = np.array(scan_msg.ranges)
        angles = np.arange(scan_msg.angle_min, 
                          scan_msg.angle_max + scan_msg.angle_increment, 
                          scan_msg.angle_increment)
        
        min_len = min(len(ranges), len(angles))
        ranges = ranges[:min_len]
        angles = angles[:min_len]
        
        # Filter valid ranges
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max) & np.isfinite(ranges)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]
        
        # Convert to cartesian
        x, y = self.polar_to_cartesian(ranges, angles)
        img_x, img_y = self.cartesian_to_image_coords(x, y)
        
        # Create base occupancy image
        occupancy_img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # 只处理圆形范围内的点
        center = self.image_size // 2
        for i in range(len(img_x)):
            # 计算点到中心的距离（像素）
            dist_from_center = math.sqrt((img_x[i] - center)**2 + (img_y[i] - center)**2)
            
            # 只保留在指定半径内的点
            if dist_from_center <= self.processing_radius_pixels:
                occupancy_img[img_y[i], img_x[i]] = 255
        # Multi-scale morphological operations for better line continuity
        # Small kernel for fine details
        kernel_small = np.ones((2, 2), np.uint8)
        occupancy_img = cv2.morphologyEx(occupancy_img, cv2.MORPH_CLOSE, kernel_small)
        
        # Medium kernel for connecting nearby points
        kernel_medium = np.ones((3, 3), np.uint8)
        occupancy_img = cv2.dilate(occupancy_img, kernel_medium, iterations=2)  # 增加迭代次数
        
        # Large kernel for major structure continuity
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 增加核大小
        occupancy_img = cv2.morphologyEx(occupancy_img, cv2.MORPH_CLOSE, kernel_large)
        
        # 额外的形态学操作以更好地连接U型结构
        kernel_connect = np.ones((4, 4), np.uint8)
        occupancy_img = cv2.morphologyEx(occupancy_img, cv2.MORPH_CLOSE, kernel_connect)
        
        # 添加噪声清理：去除小的孤立点
        kernel_clean = np.ones((2, 2), np.uint8)
        occupancy_img = cv2.morphologyEx(occupancy_img, cv2.MORPH_OPEN, kernel_clean)
        
        return occupancy_img, (x, y)

    def detect_lines_multi_scale(self, image):
        """Multi-scale line detection optimized for H-beam geometry"""
        # Apply different scales of Gaussian blur
        blurred_fine = cv2.GaussianBlur(image, (3, 3), 0)
        blurred_coarse = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Edge detection at multiple scales
        edges_fine = cv2.Canny(blurred_fine, self.canny_low, self.canny_high)
        edges_coarse = cv2.Canny(blurred_coarse, self.canny_low // 2, self.canny_high // 2)
        
        # Combine edges
        edges_combined = cv2.bitwise_or(edges_fine, edges_coarse)
        
        # Standard Hough transform for longer lines (web)
        lines_standard = cv2.HoughLinesP(edges_combined, 
                                       rho=1, 
                                       theta=np.pi/180, 
                                       threshold=self.hough_threshold,
                                       minLineLength=self.min_line_length,
                                       maxLineGap=self.max_line_gap)
        
        # More sensitive Hough transform for shorter lines (flanges)
        lines_sensitive = cv2.HoughLinesP(edges_combined, 
                                        rho=1, 
                                        theta=np.pi/180, 
                                        threshold=self.flange_hough_threshold,
                                        minLineLength=self.flange_min_line_length,
                                        maxLineGap=self.flange_max_line_gap)
        
        # Alternative approach: Detect very short lines with different theta resolution
        lines_fine_angle = cv2.HoughLinesP(edges_combined, 
                                         rho=1, 
                                         theta=np.pi/360,  # Higher angular resolution
                                         threshold=self.flange_hough_threshold,
                                         minLineLength=self.flange_min_line_length,
                                         maxLineGap=self.flange_max_line_gap)
        
        # 额外的检测策略：使用平衡的参数检测U型结构的曲线部分
        lines_curved = cv2.HoughLinesP(edges_combined, 
                                     rho=2,  # 增加rho步长
                                     theta=np.pi/180, 
                                     threshold=12,  # 平衡的阈值
                                     minLineLength=25,  # 适当的最小线长
                                     maxLineGap=30)  # 平衡的间隙容忍度
        
        # Combine all results
        all_lines = []
        if lines_standard is not None:
            all_lines.extend(lines_standard)
        if lines_sensitive is not None:
            all_lines.extend(lines_sensitive)
        if lines_fine_angle is not None:
            all_lines.extend(lines_fine_angle)
        if lines_curved is not None:
            all_lines.extend(lines_curved)
        
        return all_lines, edges_combined

    def calculate_line_properties(self, x1, y1, x2, y2):
        """Calculate comprehensive line properties"""
        start_x, start_y = self.image_coords_to_cartesian(x1, y1)
        end_x, end_y = self.image_coords_to_cartesian(x2, y2)
        
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
            'distance_from_origin': distance
        }

    def classify_line_type(self, line_props):
        """Classify line as web or flange based on length and characteristics"""
        length = line_props['length']
        deg = line_props['angle_deg']
        
        # 调试输出
        print(f"线段分类: 长度={length:.3f}m, 角度={deg:.1f}°")
        
        if (length >= self.min_line_length_meters) or (self.min_line_deg-5 < deg < self.min_line_deg+5):
            print(f"  -> 分类为web (长度条件: {length >= self.min_line_length_meters}, 角度条件: {self.min_line_deg-5 < deg < self.min_line_deg+5})")
            return 'web'  # Longer lines are likely webs
        elif length >= self.min_flange_length_meters:
            print(f"  -> 分类为flange (长度条件: {length >= self.min_flange_length_meters})")
            return 'flange'  # Shorter lines are likely flanges
        else:
            print(f"  -> 分类为noise (长度太短: {length:.3f} < {self.min_flange_length_meters})")
            return 'noise'  # Too short to be useful
    
    def detect_h_beam_structures(self, lines):
        """Detect H-beam structures by finding perpendicular line relationships"""
        if not lines:
            return []
        
        # Classify lines
        webs = []
        flanges = []
        
        for line in lines:
            line_type = self.classify_line_type(line['properties'])
            line['type'] = line_type
            
            if line_type == 'web':
                webs.append(line)
            elif line_type == 'flange':
                flanges.append(line)
        
        print(f"分类结果: {len(webs)} webs, {len(flanges)} flanges")
        
        if len(webs) > 1:
            web_clusters = self.cluster_similar_lines(webs)
            merged_webs = []
            for cluster in web_clusters:
                merged = self.merge_line_cluster(cluster)
                if merged is not None:
                    merged['type'] = 'web'
                    merged_webs.append(merged)
            webs = merged_webs
            # print(f"Webs merged to {len(webs)} after clustering")
        
        # Find H-beam structures (webs with perpendicular flanges)
        h_beams = []
        used_web_ids = set()

        for web in webs:
            web_angle = web['properties']['angle_deg']
            web_midpoint = np.array(web['properties']['midpoint'])
            
            print(f"检查web: 角度={web_angle:.1f}°, 中点=({web_midpoint[0]:.3f}, {web_midpoint[1]:.3f})")
            
            # Find flanges that are approximately perpendicular to this web
            associated_webs = []
            associated_flanges = []
            
            for flange in flanges:
                flange_angle = flange['properties']['angle_deg']
                flange_midpoint = np.array(flange['properties']['midpoint'])
                
                # Check if angles are approximately perpendicular (90 degrees apart)
                angle_diff = abs(web_angle - flange_angle)
                angle_diff = min(angle_diff, 180 - angle_diff, abs(angle_diff - 90), abs(angle_diff - 270))
                
                # Check if flange is close to the web
                distance_to_web = self.point_to_line_distance(flange_midpoint, web['properties'])
                
                print(f"  检查flange: 角度={flange_angle:.1f}°, 角度差={angle_diff:.1f}°, 距离={distance_to_web:.3f}m")
                
                if angle_diff < 25 and distance_to_web < 0.8:  # 放宽角度和距离容忍度
                    print(f"    -> flange匹配!")
                    associated_flanges.append(flange)
                else:
                    print(f"    -> flange不匹配 (角度差={angle_diff:.1f}° >= 25° 或 距离={distance_to_web:.3f}m >= 0.8m)")
            
            print(f"  web找到 {len(associated_flanges)} 个匹配的flange")
            
            # Create H-beam structure
            if len(associated_flanges) >= 2:
                print(f"  -> 创建H-beam结构!")
                h_beam = {
                    'web': web,
                    'flanges': associated_flanges,
                    'type': 'h_beam',
                    'total_lines': 1 + len(associated_flanges)
                }
                h_beams.append(h_beam)
                used_web_ids.add(id(web))
            else:
                print(f"  -> 不构成H-beam (需要≥2个flange，实际{len(associated_flanges)}个)")

        # Also include standalone flanges that weren't associated with webs
        standalone_flanges = []
        associated_flange_ids = set()
        standalone_webs = [web for web in webs if id(web) not in used_web_ids]
    
        for h_beam in h_beams:
            for flange in h_beam['flanges']:
                associated_flange_ids.add(id(flange))
        
        for flange in flanges:
            if id(flange) not in associated_flange_ids:
                standalone_flanges.append(flange)
        
        # print(f"Found {len(h_beams)} H-beam structures, {len(standalone_flanges)} standalone flanges")
        
        return h_beams, standalone_flanges, standalone_webs

    def simple_merge_similar_lines(self, lines):
        """简单暴力的线段合并方法"""
        if not lines:
            return []
        
        merged_lines = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            
            # 收集所有与line1相似的线段
            similar_lines = [line1]
            used[i] = True
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if used[j]:
                    continue
                
                props1 = line1['properties']
                props2 = line2['properties']
                
                # 简单判断条件：中点距离 + 角度差异
                midpoint_dist = math.sqrt(
                    (props1['midpoint'][0] - props2['midpoint'][0])**2 + 
                    (props1['midpoint'][1] - props2['midpoint'][1])**2
                )
                
                angle_diff = abs(props1['angle_deg'] - props2['angle_deg'])
                angle_diff = min(angle_diff, 180 - angle_diff)  # 处理0/180度环绕
                
                # 暴力判断：距离和角度都满足阈值就合并
                if midpoint_dist < self.position_threshold and angle_diff < self.angle_threshold:
                    similar_lines.append(line2)
                    used[j] = True
            
            # 将相似的线段合并成一条
            if len(similar_lines) == 1:
                merged_lines.append(similar_lines[0])
            else:
                merged_line = self.simple_fit_single_line(similar_lines)
                if merged_line:
                    merged_lines.append(merged_line)
        
        return merged_lines

    def simple_fit_single_line(self, lines):
        """将多条线段简单拟合成一条线"""
        if not lines:
            return None
        
        # 收集所有端点
        all_points = []
        for line in lines:
            all_points.append(line['properties']['start_point'])
            all_points.append(line['properties']['end_point'])
        
        # 转换为numpy数组
        points = np.array(all_points)
        
        # 找到距离原点最远和最近的两个点作为新线段的端点
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # 或者用更简单的方法：找到x+y最小和最大的点
        sums = points[:, 0] + points[:, 1]
        min_idx = np.argmin(sums)
        max_idx = np.argmax(sums)
        
        start_point = points[min_idx]
        end_point = points[max_idx]
        
        # 如果两点太近，用最长的原线段
        if np.linalg.norm(end_point - start_point) < 0.05:
            longest_line = max(lines, key=lambda x: x['properties']['length'])
            return longest_line
        
        # 转换回图像坐标
        img_coords = self.cartesian_to_image_coords(
            np.array([start_point[0], end_point[0]]),
            np.array([start_point[1], end_point[1]])
        )
        
        # 计算新线段的属性
        props = self.calculate_line_properties(
            int(img_coords[0][0]), int(img_coords[1][0]),
            int(img_coords[0][1]), int(img_coords[1][1])
        )
        
        # 继承最长线段的类型
        longest_line = max(lines, key=lambda x: x['properties']['length'])
        line_type = longest_line.get('type', 'unknown')
        
        return {
            'image_coords': (int(img_coords[0][0]), int(img_coords[1][0]), 
                            int(img_coords[0][1]), int(img_coords[1][1])),
            'properties': props,
            'type': line_type
        }
    def point_to_line_distance(self, point, line_props):
        """Calculate perpendicular distance from point to line"""
        start = np.array(line_props['start_point'])
        end = np.array(line_props['end_point'])
        
        # Line vector
        line_vec = end - start
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-6:
            return np.linalg.norm(point - start)
        
        # Unit vector along line
        line_unit = line_vec / line_length
        
        # Vector from start to point
        point_vec = point - start
        
        # Project point onto line
        projection_length = np.dot(point_vec, line_unit)
        
        # Find closest point on line segment
        if projection_length <= 0:
            closest_point = start
        elif projection_length >= line_length:
            closest_point = end
        else:
            closest_point = start + projection_length * line_unit
        
        return np.linalg.norm(point - closest_point)

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
                                             img_coords[0][1], img_coords[1][1])
        
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
                                             img_coords[0][1], img_coords[1][1])
        
        return {
            'image_coords': (img_coords[0][0], img_coords[1][0], img_coords[0][1], img_coords[1][1]),
            'properties': props
        }

    def fit_robust_line(self, points):
        """Fit a robust line through points using iterative approach"""
        if len(points) < 2:
            return None
        
        # Initial fit using least squares
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Handle vertical lines
        if np.std(x_coords) < 0.01:
            # Vertical line
            x_avg = np.mean(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            start_point = (x_avg, y_min)
            end_point = (x_avg, y_max)
        else:
            # Regular line fitting
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            slope, intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min = slope * x_min + intercept
            y_max = slope * x_max + intercept
            
            start_point = (x_min, y_min)
            end_point = (x_max, y_max)
        
        # Convert back to image coordinates for consistency
        img_x = self.cartesian_to_image_coords(
            np.array([start_point[0], end_point[0]]), 
            np.array([start_point[1], end_point[1]])
        )
        
        props = self.calculate_line_properties(img_x[0][0], img_x[1][0], img_x[0][1], img_x[1][1])
        
        return {
            'image_coords': (img_x[0][0], img_x[1][0], img_x[0][1], img_x[1][1]),
            'properties': props
        }

    def track_lines_temporally(self, current_lines):
        """Track lines across multiple frames for stability"""
        # Add current detection to history
        self.line_history.append(current_lines)
        
        if len(self.line_history) < self.stability_requirement:
            return []  # Not enough history yet
        
        # Find consistently detected lines
        stable_candidates = []
        
        for current_line in current_lines:
            consistency_count = 1  # Current frame
            
            # Check consistency across history
            for historical_frame in list(self.line_history)[:-1]:
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

    def update_stable_lines(self, new_stable_lines):
        """Update the stable line list with temporal filtering"""
        # Age existing stable lines
        for line in self.stable_lines:
            line['age'] = line.get('age', 0) + 1
        
        # Remove old lines
        self.stable_lines = [line for line in self.stable_lines 
                           if line['age'] < self.max_line_age]
        
        # Update or add new stable lines
        for new_line in new_stable_lines:
            # Find matching existing line
            matched = False
            for existing_line in self.stable_lines:
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
                new_line['id'] = self.line_id_counter
                self.line_id_counter += 1
                self.stable_lines.append(new_line)

    def publish_scan_points(self, scan_msg):
        """Publish scan points as RViz markers for visualization"""
        marker = Marker()
        marker.header.frame_id = "radar_platform/left_radar"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "scan_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        
        ranges = np.array(scan_msg.ranges)
        angles = np.arange(scan_msg.angle_min, 
                          scan_msg.angle_max + scan_msg.angle_increment, 
                          scan_msg.angle_increment)
        
        min_len = min(len(ranges), len(angles))
        ranges = ranges[:min_len]
        angles = angles[:min_len]
        
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max) & np.isfinite(ranges)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]
        
        x, y = self.polar_to_cartesian(ranges, angles)
        
        for xi, yi in zip(x, y):
            point = Point()
            point.x = xi
            point.y = yi
            point.z = 0.0
            marker.points.append(point)
        
        self.scan_points_pub.publish(marker)

    def publish_h_beam_markers(self, h_beams, standalone_flanges):
        """Publish H-beam structures with enhanced visualization"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        marker_id = 0
        
        # Publish H-beam structures
        for i, h_beam in enumerate(h_beams):
            web = h_beam['web']
            flanges = h_beam['flanges']
            
            # Web marker (red, thicker)
            web_marker = Marker()
            web_marker.header.frame_id = "radar_platform/left_radar"
            web_marker.header.stamp = rospy.Time.now()
            web_marker.ns = "h_beam_webs"
            web_marker.id = marker_id
            marker_id += 1
            web_marker.type = Marker.LINE_STRIP
            web_marker.action = Marker.ADD
            
            web_marker.scale.x = 0.08  # Thick line for web
            web_marker.color.r = 1.0
            web_marker.color.g = 0.0
            web_marker.color.b = 0.0
            web_marker.color.a = 1.0
            
            start_point = Point()
            start_point.x = web['properties']['start_point'][0]
            start_point.y = web['properties']['start_point'][1]
            start_point.z = 0.0
            
            end_point = Point()
            end_point.x = web['properties']['end_point'][0]
            end_point.y = web['properties']['end_point'][1]
            end_point.z = 0.0
            
            web_marker.points = [start_point, end_point]
            marker_array.markers.append(web_marker)
            
            # Flange markers (blue, thinner)
            for j, flange in enumerate(flanges):
                flange_marker = Marker()
                flange_marker.header = web_marker.header
                flange_marker.ns = "h_beam_flanges"
                flange_marker.id = marker_id
                marker_id += 1
                flange_marker.type = Marker.LINE_STRIP
                flange_marker.action = Marker.ADD
                
                flange_marker.scale.x = 0.05  # Thinner line for flanges
                flange_marker.color.r = 0.0
                flange_marker.color.g = 0.0
                flange_marker.color.b = 1.0
                flange_marker.color.a = 0.8
                
                f_start = Point()
                f_start.x = flange['properties']['start_point'][0]
                f_start.y = flange['properties']['start_point'][1]
                f_start.z = 0.0
                
                f_end = Point()
                f_end.x = flange['properties']['end_point'][0]
                f_end.y = flange['properties']['end_point'][1]
                f_end.z = 0.0
                
                flange_marker.points = [f_start, f_end]
                marker_array.markers.append(flange_marker)
            
            # H-beam label
            text_marker = Marker()
            text_marker.header = web_marker.header
            text_marker.ns = "h_beam_labels"
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = web['properties']['midpoint'][0]
            text_marker.pose.position.y = web['properties']['midpoint'][1]
            text_marker.pose.position.z = 0.4
            
            text_marker.scale.z = 0.25
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 0.0
            text_marker.color.a = 1.0
            
            text_marker.text = f"H{i}: W={web['properties']['length']:.2f}m, F={len(flanges)}"
            marker_array.markers.append(text_marker)
        
        # Publish standalone flanges (green)
        for i, flange in enumerate(standalone_flanges):
            flange_marker = Marker()
            flange_marker.header.frame_id = "radar_platform/left_radar"
            flange_marker.header.stamp = rospy.Time.now()
            flange_marker.ns = "standalone_flanges"
            flange_marker.id = marker_id
            marker_id += 1
            flange_marker.type = Marker.LINE_STRIP
            flange_marker.action = Marker.ADD
            
            flange_marker.scale.x = 0.04
            flange_marker.color.r = 0.0
            flange_marker.color.g = 1.0
            flange_marker.color.b = 0.0
            flange_marker.color.a = 0.7
            
            f_start = Point()
            f_start.x = flange['properties']['start_point'][0]
            f_start.y = flange['properties']['start_point'][1]
            f_start.z = 0.0
            
            f_end = Point()
            f_end.x = flange['properties']['end_point'][0]
            f_end.y = flange['properties']['end_point'][1]
            f_end.z = 0.0
            
            flange_marker.points = [f_start, f_end]
            marker_array.markers.append(flange_marker)
        
        self.marker_pub.publish(marker_array)

    def publish_debug_line_info(self, st_line):
        """/main_radar/line_detection_info"""
        line_detection_array_msg = LineDetectionArray()
        line_detection_array_msg.header.stamp = rospy.Time.now()
        line_detection_array_msg.header.frame_id = "radar_platform/left_radar"

        line_id = 0
        for h_beam in st_line:
            # 添加 web
            web = h_beam['web']
            props = web['properties']
            start = props['start_point']
            end = props['end_point']

            line_info = LineInfo()
            line_info.id = web.get('id', line_id)
            line_id += 1
            line_info.start_point.x = start[0]
            line_info.start_point.y = 0.0
            line_info.start_point.z = start[1]
            line_info.end_point.x = end[0]
            line_info.end_point.y = 0.0
            line_info.end_point.z = end[1]
            line_info.length = props['length']
            line_info.angle_deg = props['angle_deg']
            line_info.distance = props['distance_from_origin']
            line_info.type = 0
            line_detection_array_msg.lines.append(line_info)

            # 添加 flanges
            for flange in h_beam['flanges']:
                props = flange['properties']
                start = props['start_point']
                end = props['end_point']

                line_info = LineInfo()
                line_info.id = flange.get('id', line_id)
                line_id += 1
                line_info.start_point.x = start[0]
                line_info.start_point.y = 0.0
                line_info.start_point.z = start[1]
                line_info.end_point.x = end[0]
                line_info.end_point.y = 0.0
                line_info.end_point.z = end[1]
                line_info.length = props['length']
                line_info.angle_deg = props['angle_deg']
                line_info.distance = props['distance_from_origin']
                line_info.type = 1
                line_detection_array_msg.lines.append(line_info)

        self.line_info_pub.publish(line_detection_array_msg)

    def scan_callback(self, scan_msg):
        """Enhanced main callback with stability processing"""
        try:
            # Create enhanced occupancy image
            occupancy_img, scan_points = self.create_enhanced_occupancy_image(scan_msg)
            
            # Multi-scale line detection
            lines, edges = self.detect_lines_multi_scale(occupancy_img)
            
            if lines:
                # Calculate properties for all detected lines
                current_lines = []
                for line in lines:
                    # Handle different line formats from HoughLinesP
                    if isinstance(line, np.ndarray) and line.ndim > 1:
                        x1, y1, x2, y2 = line[0]
                    else:
                        x1, y1, x2, y2 = line
                    
                    # Ensure coordinates are integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Skip invalid lines
                    if x1 == x2 and y1 == y2:
                        continue
                        
                    props = self.calculate_line_properties(x1, y1, x2, y2)
                    
                    # 更严格的长度和角度检查以减少误检
                    if (props['length'] >= self.min_flange_length_meters and 
                        props['length'] <= self.max_line_length_meters):
                        
                        # 额外的质量检查：确保线段有足够的点密度
                        line_length_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if line_length_pixels >= self.min_line_length:  # 像素级别的长度检查
                            
                            current_lines.append({
                                'image_coords': (x1, y1, x2, y2),
                                'properties': props
                            })
                
                '''                
                print(f"Detected {len(current_lines)} valid lines")
                merged_lines = self.merge_overlapping_lines_advanced(current_lines)
                print(f"After advanced merging: {len(merged_lines)}")
                # Detect H-beam structures
                h_beams, standalone_flanges, standalone_webs = self.detect_h_beam_structures(current_lines)
                
                # Combine all lines for temporal tracking
                all_structural_lines = []
                for h_beam in h_beams:
                    all_structural_lines.append(h_beam['web'])
                    all_structural_lines.extend(h_beam['flanges'])
                all_structural_lines.extend(standalone_flanges)
                
                # Apply clustering only to similar line types
                web_lines = [line for line in all_structural_lines if line.get('type') == 'web']
                flange_lines = [line for line in all_structural_lines if line.get('type') == 'flange']
                
                # Cluster webs and flanges separately
                web_clusters = self.cluster_similar_lines(web_lines)
                flange_clusters = self.cluster_similar_lines(flange_lines)
                
                merged_lines = []
                for cluster in web_clusters + flange_clusters:
                    try:
                        merged_line = self.merge_line_cluster(cluster)
                        if merged_line is not None:
                            # Preserve line type
                            merged_line['type'] = cluster[0].get('type', 'unknown')
                            merged_lines.append(merged_line)
                    except Exception as e:
                        print(f"Error merging cluster: {e}")
                        longest = max(cluster, key=lambda x: x['properties']['length'])
                        merged_lines.append(longest)
                '''
                # print(f"Detected {len(current_lines)} valid lines")
                
                # 简单暴力合并相似线段
                merged_lines = self.simple_merge_similar_lines(current_lines)
                print(f"After simple merging: {len(merged_lines)} lines")
                
                # Detect H-beam structures using merged lines
                h_beams, standalone_flanges, standalone_webs = self.detect_h_beam_structures(merged_lines)
                print(f"当前帧H-beam结构数量: {len(h_beams)}")
                print(f"当前帧独立flange数量: {len(standalone_flanges)}")
                
                # Temporal tracking for stability
                stable_lines = self.track_lines_temporally(merged_lines)
                self.update_stable_lines(stable_lines)
                
                # Re-analyze stable lines for H-beam structures
                stable_h_beams, stable_standalone, standalone_webs = self.detect_h_beam_structures(self.stable_lines)
                
                # 调试输出
                print(f"稳定线段数量: {len(stable_lines)}")
                print(f"稳定H-beam结构数量: {len(stable_h_beams)}")
                print(f"独立flange数量: {len(stable_standalone)}")
                
                # 如果稳定检测没有结果，使用当前帧的检测结果
                if len(stable_h_beams) == 0 and len(h_beams) > 0:
                    print("使用当前帧检测结果作为备选")
                    stable_h_beams = h_beams
                    stable_standalone = standalone_flanges
                
                # Publish results
                self.publish_scan_points(scan_msg)
                self.publish_h_beam_markers(stable_h_beams, stable_standalone)
                if self.debug_mode:
                    self.save_debug_images(occupancy_img, edges, merged_lines)

                self.publish_debug_line_info(stable_h_beams)

            else:
                print("No lines detected")
                self.publish_scan_points(scan_msg)
            
        except Exception as e:
            # rospy.logerr(f"Error in scan callback: {e}")
            # import traceback
            # traceback.print_exc()
            pass

    def save_debug_images(self, occupancy_img, edges, lines):
        """Save debug images showing the detection process"""
        # Create visualization image
        vis_img = cv2.cvtColor(occupancy_img, cv2.COLOR_GRAY2BGR)
        
        # Draw detected lines
        for line in lines:
            x1, y1, x2, y2 = line['image_coords']
            cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # Draw start and end points
            cv2.circle(vis_img, (int(x1), int(y1)), 3, (0, 255, 0), -1)
            cv2.circle(vis_img, (int(x2), int(y2)), 3, (255, 0, 0), -1)
        
        # 添加边缘检测结果的可视化
        edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 在占用图像上绘制处理范围边界
        occupancy_vis = cv2.cvtColor(occupancy_img, cv2.COLOR_GRAY2BGR)
        center = self.image_size // 2
        cv2.circle(occupancy_vis, (center, center), self.processing_radius_pixels, (0, 255, 0), 2)
        
        # 显示多个窗口以便调试
        cv2.imshow("Occupancy Image (with range)", occupancy_vis)
        # cv2.imshow("Edge Detection", edges_vis)
        cv2.imshow("Detected Lines", vis_img)
        cv2.waitKey(1)
        
        # 可选：保存调试图像
        # cv2.imwrite('/tmp/radar_occupancy.png', occupancy_img)
        # cv2.imwrite('/tmp/radar_edges.png', edges)
        # cv2.imwrite('/tmp/radar_lines.png', vis_img) 

    def run(self):
        """Main run loop"""
        rospy.loginfo("Stable Radar Line Detector started")
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = StableRadarLineDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass