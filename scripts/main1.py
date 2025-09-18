#!/usr/bin/env python3
import sys
import time
import threading
import rospy
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'gen_py'))
sys.path.append(os.path.join(base_dir, 'lib'))
from DucoCobot import DucoCobot
from thrift import Thrift
from ManualControl import system_control
from std_msgs.msg import Float64MultiArray
from config import *

class DemoApp:
    def __init__(self):
        self.ip = IP
        self.stopheartthread = False
        self.duco_cobot = DucoCobot(IP, PORT)
        self.hearthread = threading.Thread(target=self.hearthread_fun)
        self.thread = threading.Thread(target=self.thread_fun)
        self.tcp_state = []  
        self.tcp_pub = rospy.Publisher('/Duco_state', Float64MultiArray, queue_size=20)
        # self.adjust_pub = rospy.Publisher('/Duco_adjust', Float64MultiArray, queue_size=20) 
        self.sys_ctrl = None 

    def robot_connect(self):
        rlt = self.duco_cobot.open()
        print("open:", rlt)
        rlt = self.duco_cobot.power_on(True)
        print("power_on:", rlt)
        rlt = self.duco_cobot.enable(True)
        print("enable:", rlt)
        self.duco_cobot.switch_mode(1)

    def hearthread_fun(self):
        self._stop_event = threading.Event()
        self.duco_heartbeat = DucoCobot(self.ip, PORT)
        self.duco_heartbeat.open()
        while not self.stopheartthread:
            self.duco_heartbeat.rpc_heartbeat()
            self._stop_event.wait(1)
        self.duco_heartbeat.close()

    def thread_fun(self):
        self.duco_thread = DucoCobot(self.ip, PORT)
        self.duco_thread.open()
        while not self.stopheartthread:
            tcp_pos = self.duco_thread.get_tcp_pose()
            tcp_state = self.duco_thread.get_robot_state()
            empty_state = [0] * 4
            # 从 system_control 获取车辆状态
            if self.sys_ctrl is not None:
                car_state, running_state, distances = self.sys_ctrl.get_car_state()
            else:
                car_state = [0, 0]  # 默认值
            self.tcp_state = tcp_state
            # 发布 ROS topic
            msg = Float64MultiArray()
            msg.data = tcp_state + tcp_pos + distances + car_state + [running_state]
            self.tcp_pub.publish(msg)

            self._stop_event.wait(0.2)
        self.duco_thread.close()

    def run(self):
        self.robot_connect()
        self.hearthread.start()
        self.thread.start()

        try:
            self.sys_ctrl = system_control(self.duco_cobot, self)
            self.sys_ctrl.run()
        finally:
            self.stopheartthread = True
            time.sleep(1)
            self.hearthread.join()
            self.thread.join()
            rlt = self.duco_cobot.close()
            print("close:", rlt)

    def main(self):
        try:
            self.run()
        except Thrift.TException as tx:
            print('%s' % tx.message)

if __name__ == '__main__':
    rospy.init_node('Duco_state_publisher', anonymous=True)
    app = DemoApp()
    try:
        app.main()
    except KeyboardInterrupt:
        print("主程序收到 KeyboardInterrupt，准备退出。")
        app.stopheartthread = True
        time.sleep(1)
        app.hearthread.join()
        app.thread.join()
        rlt = app.duco_cobot.close()
        print("close:", rlt)
