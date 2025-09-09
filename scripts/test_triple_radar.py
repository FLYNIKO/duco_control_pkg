#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from triple_radar_front_distance import TripleRadarFrontDistance, DirectionalLaser

def test_triple_radar():
    """测试TripleRadarFrontDistance类的功能"""
    rospy.init_node('test_triple_radar', anonymous=True)
    
    print("=== 测试TripleRadarFrontDistance类 ===")
    
    # 测试1: 创建实例
    print("1. 创建TripleRadarFrontDistance实例...")
    triple_radar = TripleRadarFrontDistance()
    
    # 等待一段时间让数据初始化
    rospy.sleep(2)
    
    # 测试2: 检查数据有效性
    print("2. 检查数据有效性...")
    is_valid = triple_radar.is_data_valid()
    print(f"   数据有效: {is_valid}")
    
    # 测试3: 获取所有雷达的前方距离
    print("3. 获取所有雷达的前方距离...")
    front_distances = triple_radar.get_all_front_distances()
    for radar_name, distance in front_distances.items():
        if distance > 0:
            print(f"   {radar_name}_radar front: {distance:.3f} m")
        else:
            print(f"   {radar_name}_radar front: 无效")
    
    # 测试4: 测试兼容性接口
    print("4. 测试兼容性接口...")
    front_dist = triple_radar.get_distance("front")
    down_dist = triple_radar.get_distance("down")
    up_dist = triple_radar.get_distance("up")
    
    print(f"   front: {front_dist:.3f} m")
    print(f"   down: {down_dist:.3f} m")
    print(f"   up: {up_dist:.3f} m")
    
    # 测试5: 获取所有方向距离
    print("5. 获取所有方向距离...")
    all_distances = triple_radar.get_all_distances()
    for direction, distance in all_distances.items():
        if distance > 0:
            print(f"   {direction}: {distance:.3f} m")
        else:
            print(f"   {direction}: 无效")
    
    # 测试6: 测试DirectionalLaser别名
    print("6. 测试DirectionalLaser别名...")
    directional_laser = DirectionalLaser()
    rospy.sleep(1)
    
    if directional_laser.is_data_valid():
        dist = directional_laser.get_distance("front")
        print(f"   DirectionalLaser front: {dist:.3f} m")
    else:
        print("   DirectionalLaser 数据无效")
    
    print("=== 测试完成 ===")
    
    # 清理
    triple_radar.shutdown()
    directional_laser.shutdown()

if __name__ == "__main__":
    try:
        test_triple_radar()
    except rospy.ROSInterruptException:
        print("测试被中断")
    except Exception as e:
        print(f"测试出错: {e}")








