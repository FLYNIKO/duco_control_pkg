# coding:utf-8 
#!/usr/bin/env python
# from http import client

import sys
import glob
import threading
import time

sys.path.append('gen_py')
sys.path.append('lib')
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from gen_py.robot import RPCRobot
from gen_py.robot.ttypes import StateRobot, StateProgram, OperationMode, TaskState, Op, RealTimeControlData, PointOp


class DucoCobot:
    def __init__(self, ip, port):
        self.transport = TSocket.TSocket(ip, port)
        self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        self.client = RPCRobot.Client(self.protocol)
        
    op_ = Op()
    op_.time_or_dist_1 = 0        # 轨迹起始点触发类型, 0:不启用, 1:时间触发, 2:距离触发
    op_.trig_io_1 = 1             # 轨迹触发控制柜IO的输出序号, 范围1-16
    op_.trig_value_1 = False      # 轨迹触发控制柜IO的电平高低, false:低电平, true:高电平
    op_.trig_time_1 = 0.0         # 当time_or_dist_1为1时, 代表轨迹运行多少时间长度触发IO,单位: ms
    op_.trig_dist_1 = 0.0         # 当time_or_dist_1为2时, 代表轨迹运行多少距离长度触发IO,单位: m
    op_.trig_event_1 = ''         # 轨迹触发的用户自定义事件名称
    op_.time_or_dist_2 = 0        # 轨迹结束点触发类型, 0:不启用, 1:时间触发, 2:距离触发
    op_.trig_io_2 = 1             # 轨迹触发控制柜IO的输出序号, 范围1-16
    op_.trig_value_2 = False      # 轨迹触发控制柜IO的电平高低, false:低电平, true:高电平
    op_.trig_time_2 = 0.0         # 当time_or_dist_2为1时, 当trig_time_2 >= 0时, 代表轨迹运行剩余多少时间长度触发IO,单位: ms; 当trig_time_2 < 0时, 代表代表轨迹运行结束后多少时间长度触发IO
    op_.trig_dist_2 = 0.0         # 当time_or_dist_2为2时, 当trig_ dist _2 >= 0时, 代表轨迹运行剩余多少距离长度触发IO,单位: m;当trig_ dist _2 < 0时, 代表代表轨迹运行结束后多少距离长度触发IO
    op_.trig_event_2 = ''         # 轨迹触发的用户自定义事件名称
    op_.time_or_dist_3 = 0        # 轨迹暂停或停止触发类型，0：不启用，1：时间触发。
    op_.trig_io_3 = 1             # 轨迹触发控制柜IO的输出序号, 范围1-16
    op_.trig_value_3 = False      # 轨迹触发控制柜IO的电平高低, false:低电平, true:高电平
    op_.trig_time_3 = 0.0         # 当time_or_dist_3为1时, 当trig_time_3 >= 0时, 代表任务暂停或停止多少时间触发IO,单位: ms;
    op_.trig_dist_3 = 0.0         # 不适用
    op_.trig_event_3 = ''         # 轨迹触发的用户自定义事件名称
    '''
    * @brief 连接机器人
    * @return -1:打开失败; 0:打开成功
    '''
    def open(self):
        try:
            self.transport.open()
        except TTransport.TTransportException as e:
            print("open mesg:",repr(e))
            return -1
        else:
            return 0   
    '''
    * @brief 断开机器人连接
    * @return -1:打开失败; 0:打开成功
    '''
    def close(self):
        try:
            self.transport.close()
        except Exception as e:
            print("close mesg:",repr(e))
            return -1
        else:
            return 0
    '''
     * @brief 机器人上电
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def power_on(self, block):
        return self.client.power_on(block)
    '''
     * @brief 机器人下电
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def power_off(self, block):
        return self.client.power_off(block)
    '''
     * @brief 机器人上使能
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def enable(self, block):
        return self.client.enable(block)
    '''
     * @brief 机器人下使能
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def disable(self, block):
        return self.client.disable(block)
    '''
     * @brief 机器人关机
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def shutdown(self, block):
        return self.client.shutdown(block)
    '''
     * @brief 停止所有任务
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def stop(self, block):
        return self.client.stop(block)
    '''
     * @brief 暂停所有任务
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def pause(self, block):
        return self.client.pause(block)
    '''
     * @brief 恢复所有暂停的任务
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def resume(self, block):
        return self.client.resume(block)
    '''
     * @brief 运行程序脚本
     * @param name  脚本程序名称
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def run_program(self, name, block):
        return self.client.run_program(name, block)
    '''
     * @brief 设置工具末端相对于法兰面坐标系的偏移, 设置成功后, 
	 *        后续运动函数中的TCP设置为该TCP或者为空时,使用该TCP参数
     * @param name           工具坐标系的名字    
     * @param tool_offset    工具TCP偏移量 [x_off, y_off,z_off,rx,ry,rz], 单位: m, rad
     * @param payload        末端负载质量, 质心, [mass,x_cog,y_cog,z_cog], 单位: kg, m
     * @param inertia_tensor 末端工具惯量矩阵参数, 参数1-6分别对应矩阵xx、xy、xz、yy、yz、zz元素, 单位: kg*m^2
     * @return 返回当前任务结束时的状态
     '''
    def set_tool_data(self, name, tool_offset, payload, inertia_tensor):
        return self.client.set_tool_data(name, tool_offset, payload, inertia_tensor)
    '''
     * @brief 获取当前设置工具的负载质量及质心位置
     * @return 质量单位: kg,质心位置单位: m,[mass,x_cog,y_cog,z_cog]
     '''
    def get_tool_load(self):
        return self.client.get_tool_load()
    '''
     * @brief 获取当前状态下机械臂有效的末端工具的偏移量
     * @return TCP偏移量信息,单位: m, rad
     '''
    def get_tcp_offset(self):
        return self.client.get_tcp_offset()
    '''
     * @brief 设置工件坐标系
     * @param name 工件坐标系名称
     * @param wobj 工件坐标系
     * @return 返回当前任务结束时的状态
     '''
    def set_wobj(self, name, wobj):
        return self.client.set_wobj(name, wobj)
    '''
     * @brief 获取当前设置的工件坐标系的值
     * @return [x, y, z, rx, ry, rz]工件坐标系相对于基坐标系的位移, 单位: m, rad
     '''
    def get_wobj(self):
        return self.client.get_wobj()
    '''
     * @brief 基于当前的工件坐标系设置一个偏移量, 后续的move类脚本的参考工件坐标系上都将添加这个偏移量
     * @param wobj 工件坐标系相对于基坐标系的位移, 单位: m, rad
     * @param active 是否启用
     * @return 返回当前任务结束时的状态
     '''
    def set_wobj_offset(self, wobj, active = True):
        return self.client.set_wobj_offset(wobj, active)
    '''
     * @brief 计算机械臂的正运动学, 求解给定TCP在给定wobj下的值
     * @param joints_position 需要计算正解的关节角, 单位: rad
     * @param tool            工具坐标系信息,tcp偏移量[x_off,y_off,z_off,rx,ry,rz], 单位:m, rad, 为空使用当前tcp值
     * @param wobj            工件坐标系相对于基坐标系的位移[x, y, z, rx, ry, rz], 单位:m, rad, 为空使用当前wobj
     * @return                末端姿态列表[x,y,z,rx,ry,rz]
     '''
    def cal_fkine(self, joints_position, tool, wobj):
        return self.client.cal_fkine(joints_position, tool, wobj)
    '''
     * @brief 计算运动学逆解, 在求解过程中, 会选取靠近当前机械臂关节位置的解
     * @param p      需要计算的末端位姿在设置工件坐标系的值,包含当前有效的工具偏移量, 单位:m,rad
     * @param q_near 用于计算逆运动学的参考关节位置,为空使用当前关节值
     * @param tool   工具坐标系信息,tcp偏移量[x_off,y_off,z_off,rx,ry,rz], 单位:m, rad, 为空使用当前tcp值
     * @param wobj   工件坐标系相对于基坐标系的位移[x, y, z, rx, ry, rz], 单位:m, rad, 为空使用当前wobj
     * @return       关节位置列表[q1,q2,q3,q4,q5,q6]
     '''
    def cal_ikine(self, p, q_near, tool, wobj):
        return self.client.cal_ikine(p, q_near, tool, wobj)
    '''
     * @brief 设置通用IO输出的信号类型
     * @param num         控制柜上的IO输出口序号, 范围从1-16
     * @param type        输出的信号类型, 0: 高低电平, 1: 脉冲
     * @param freq        脉冲频率(Hz)
     * @param duty_cycle  脉冲占空比(%)
     * @return
     '''    
    def set_digital_out_mode(self, num, type, freq, duty_cycle):
        return self.client.set_digital_output_mode(num, type, freq, duty_cycle)
    '''
     * @brief 该函数可控制控制柜上的IO输出口的高低电平
     * @param num   控制柜上的IO输出口序号, 范围从1-16
     * @param value true为高电平, false为低电平
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def set_standard_digital_out(self, num, value, block):
        return self.client.set_standard_digital_out(num, value, block)
    '''
     * @brief set_tool_digital_out
     * @param num   机械臂末端的IO输出口序号, 范围从1-2
     * @param value true为高电平, false为低电平
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def set_tool_digital_out(self, num, value, block):
        return self.client.set_tool_digital_out(num, value, block)
    '''
     * @brief 读取控制柜上的用户IO输入口的高低电平, 返回true为高电平, false为低电平
     * @param num 控制柜上的IO输入口序号, 范围从1-16
     * @return true为高电平, false为低电平
     '''
    def get_standard_digital_in(self, num):
        return self.client.get_standard_digital_in(num)
    '''
     * @brief 获取控制柜上通用IO输出口的高低电平, 返回true为高电平, false为低电平
     * @param num 控制柜上的IO输出口序号, 范围从1-16
     * @return true为高电平, false为低电平
     '''    
    def get_standard_digital_out(self, num):
        return self.client.get_standard_digital_out(num)
    '''
     * @brief 读取机械臂末端的IO输入口的高低电平, 返回true为高电平, false为低电平
     * @param num 机械臂末端的IO输出口序号, 范围从1-2
     * @return true为高电平, false为低电平
     '''
    def get_tool_digital_in(self, num):
        return self.client.get_tool_digital_in(num)
    '''
     * @brief 读取机械臂末端的IO输入口的高低电平, 返回true为高电平, false为低电平
     * @param num 机械臂末端的IO输出口序号, 范围从1-2
     * @return true为高电平, false为低电平
     '''    
    def get_tool_digital_out(self, num):
        return self.client.get_tool_digital_out(num)

    def get_config_digital_in(self, num):
        return self.client.get_config_digital_in(num)
    '''
     * @brief 读取控制柜上的模拟电压输入
     * @param num 控制柜上的模拟电压通道序号, 范围从1-4
     * @return 对应通道的模拟电压值
     '''
    def get_standard_analog_voltage_in(self, num):
        return self.client.get_standard_analog_voltage_in(num)
    '''
     * @brief 读取机械臂末端的模拟电压输入
     * @param num 机械臂末端的模拟电压通道序号, 范围从1-2
     * @return 对应通道的模拟电压值
     '''
    def get_tool_analog_voltage_in(self, num):
        return self.client.get_tool_analog_voltage_in(num)
    '''
     * @brief 读取控制柜上的模拟电流输入
     * @param num 控制柜上的模拟电流通道序号, 范围从1-4
     * @return 对应通道的模拟电流值
     '''
    def get_standard_analog_current_in(self, num):
        return self.client.get_standard_analog_current_in(num)
    '''
     * @brief 读取功能输入寄存器的值
     * @param num 内部寄存器序号, 范围从1-16
     * @return bool寄存器的值
     '''    
    def get_function_reg_in(self, num):
        return self.client.get_function_reg_in(num)
    '''
     * @brief 读取功能输出寄存器的值
     * @param num 内部寄存器序号, 范围从1-16
     * @return bool寄存器的值
     '''    
    def get_function_reg_out(self, num):
        return self.client.get_function_reg_out(num)
    '''
     * @brief 读取控制柜功能输入IO高低电平, 返回true为高电平, false为低电平
     * @param num 控制柜功能IO输入口序号, 范围从1-8
     * @return true为高电平, false为低电平
     '''    
    def get_function_digital_in(self, num):
        return self.client.get_function_digital_in(num)
    '''
     * @brief 读取控制柜功能输出IO高低电平, 返回true为高电平, false为低电平
     * @param num 控制柜功能IO输出口序号, 范围从1-8
     * @return true为高电平, false为低电平
     '''   
    def get_function_digital_out(self, num):
        return self.client.get_function_digital_out(num)
    '''
     * @brief 485端口读取长度为len的字节数据
     * @param len 读取的长度
     * @return 读取到的数据, 未读到数据返回空列表
     '''
    def read_raw_data_485(self, len):
        return self.client.read_raw_data_485(len)
    '''
     * @brief 匹配头head后读取到长度为len的一帧数据
     * @param head 需要匹配的头数据
     * @param len  需要读取的长度
     * @return 读取到的数据, 未读到数据返回空列表
     '''
    def read_raw_data_485_h(self, head, len):
        return self.client.read_raw_data_485_h(head, len)
    '''
     * @brief 匹配头head和尾tail读取到一帧匹配的数据
     * @param head 需要匹配的头数据
     * @param tail 需要匹配的尾数据
     * @return 读取到的数据, 未读到数据返回空列表
     '''
    def read_raw_data_485_ht(self, head, tail):
        return self.client.read_raw_data_485_ht(head, tail)
    '''
     * @brief 485写原生数据, 将表value中的数据写入485端口
     * @param data 需要写入的数据列表
     * @return true:成功, false:失败
     '''
    def write_raw_data_485(self, data):
        return self.client.write_raw_data_485(data)
    '''
     * @brief 485写原生数据, 将列表value中的数据加上head写入485端口
     * @param data 需要写入的数据列表
     * @param head 需要添加的头
     * @return true:成功, false:失败
     '''
    def write_raw_data_485_h(self, data, head):
        return self.client.write_raw_data_485_h(data, head)
    '''
     * @brief 485写原生数据, 将列表value中的数据加上头head和尾tail写入485端口
     * @param data 写入的数据列表
     * @param head 需要添加的头
     * @param tail 需要添加的尾
     * @return true:成功, false:失败
     '''
    def write_raw_data_485_ht(self, data, head, tail):
        return self.client.write_raw_data_485_ht(data, head, tail)
    '''
     * @brief 末端485端口读取长度为len的字节数据
     * @param len  需要读取的长度
     * @return 读取到的数据, 未读到数据返回空列表
     '''
    def tool_read_raw_data_485(self, len):
        return self.client.tool_read_raw_data_485(len)
    '''
     * @brief 末端485匹配头head后读取到长度为len的一帧数据
     * @param head 需要匹配的头数据
     * @param len  需要读取的长度
     * @return 读取到的数据, 未读到数据返回空列表
     '''
    def tool_read_raw_data_485_h(self, head, len):
        return self.client.tool_read_raw_data_485_h(head, len)
    '''
     * @brief 末端485匹配头head和尾tail读取到一帧匹配的数据
     * @param head 需要匹配的头数据
     * @param tail 需要匹配的尾数据
     * @return 读取到的数据, 未读到数据返回空列表
     '''
    def tool_read_raw_data_485_ht(self, head, tail):
        return self.client.tool_read_raw_data_485_ht(head, tail)
    '''
     * @brief 末端485写原生数据, 将data写入485端口
     * @param data 写入的数据列表
     * @return true:成功, false:失败
     '''
    def tool_write_raw_data_485(self, data):
        return self.client.tool_write_raw_data_485(data)
    '''
     * @brief 末端485写原生数据, 将data中的数据加上head写入485端口
     * @param data 写入的数据列表
     * @param head 添加的头
     * @return true:成功, false:失败
     '''
    def tool_write_raw_data_485_h(self, data, head):
        return self.client.tool_write_raw_data_485_h(data, head)
    '''
     * @brief 末端485写原生数据, 将value中的数据加上头head和尾tail写入485端口
     * @param data 写入的数据列表
     * @param head 添加的头
     * @param tail 添加的尾
     * @return true:成功, false:失败
     '''
    def tool_write_raw_data_485_ht(self, data, head, tail):
        return self.client.tool_write_raw_data_485_ht(data, head, tail)
    '''
     * @brief 读取一帧can的字节数据
     * @return 读取到的数据, 未读到数据返回空列表, 读到数据时, 列表的第一个数据为发送端的can帧id
     '''
    def read_raw_data_can(self):
        return self.client.read_raw_data_can()
    '''
     * @brief can写帧为id, 数据为data的原生数据
     * @param id   数据帧的id
     * @param data 发送的数据列表
     * @return true:成功, false:失败
     '''
    def write_raw_data_can(self, id, data):
        return self.client.write_raw_data_can(id, data)
    '''
     * @brief 读取bool寄存器的值
     * @param num 寄存器序号, num范围为1-64
     * @return true or false
     '''
    def read_bool_reg(self, num):
        return self.client.read_bool_reg(num)
    '''
     * @brief 读取word寄存器的值
     * @param num 寄存器序号, num范围为1-32
     * @return 寄存器的值
     '''
    def read_word_reg(self, num):
        return self.client.read_word_reg(num)
    '''
     * @brief 读取float寄存器的值
     * @param num 寄存器序号, num范围为1-32
     * @return 寄存器的值
     '''
    def read_float_reg(self, num):
        return self.client.read_float_reg(num)
    '''
     * @brief 修改bool寄存器的值
     * @param num   寄存器序号, num范围为1-64
     * @param value true or false
     * @return 返回当前任务结束时的状态
     '''
    def write_bool_reg(self, num, value):
        return self.client.write_bool_reg(num, value)
    '''
     * @brief 修改word寄存器的值
     * @param num   寄存器序号, num范围为1-32
     * @param value 寄存器的值
     * @return 返回当前任务结束时的状态
     '''
    def write_word_reg(self, num, value):
        return self.client.write_word_reg(num, value)
    '''
     * @brief 修改float寄存器的值
     * @param num   寄存器序号, num范围为1-32
     * @param value 寄存器的值
     * @return 返回当前任务结束时的状态
     '''
    def write_float_reg(self, num, value):
        return self.client.write_float_reg(num, value)
    '''
     * @brief 控制机械臂从当前状态, 按照关节运动的方式移动到目标关节角状态
     * @param joints_list 1-6关节的目标关节角度, 单位: rad
     * @param v 关节角速度, 单位: 系统设定速度的百分比%, 取值范围(0,100]
     * @param a 关节角加速度, 单位: 系统设定加速度的百分比%, 取值范围(0,100]
     * @param r 融合半径, 单位: 系统设定最大融合半径的百分比%, 默认值为 0, 表示无融合, 取值范围[0,50)
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movej(self, joints_list, v, a, r, block, op=op_, def_acc = False):
        return self.client.movej(joints_list, v, a, r, block, op, def_acc)
    '''
     * @brief 控制机械臂从当前状态, 按照关节运动的方式移动到目标关节角状态
     * @param joints_list 1-6关节的目标关节角度, 单位: rad
     * @param v 关节角速度, 单位: rad/s
     * @param a 关节角加速度, 单位: rad/s^2
     * @param r 融合半径, 单位: m, 默认值为 0, 表示无融合.当数值大于0时表示与下一条运动融合
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movej2(self, joints_list, v, a, r, block, op=op_, def_acc = False):
        return self.client.movej2(joints_list, v, a, r, block, op, def_acc)
    '''
     * @brief 控制机械臂从当前状态, 按照关节运动的方式移动到末端目标位置
     * @param p 对应末端的位姿, 位置单位: m, 姿态以Rx、Ry、Rz表示, 单位: rad
     * @param v 关节角速度, 单位: 系统设定速度的百分比%, 取值范围(0,100]
     * @param a 关节加速度, 单位: 系统设定加速度的百分比%, 取值范围(0,100]
     * @param r 融合半径, 单位: 系统设定最大融合半径的百分比%, 默认值为 0, 表示无融合, 取值范围[0,50)
     * @param q_near 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool   设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj   设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block  是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movej_pose(self, p, v, a, r, q_near, tool, wobj, block, op=op_, def_acc = False):
        return self.client.movej_pose(p, v, a, r, q_near, tool, wobj, block, op, def_acc)
    '''
     * @brief 控制机械臂从当前状态, 按照关节运动的方式移动到末端目标位置
     * @param p 对应末端的位姿, 位置单位: m, 姿态以Rx、Ry、Rz表示, 单位: rad
     * @param v 关节角速度, 单位: rad/s
     * @param a 关节加速度, 单位: rad/s^2
     * @param r 融合半径, 单位: m, 默认值为 0, 表示无融合.当数值大于0时表示与下一条运动融合
     * @param q_near 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool   设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj   设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block  是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movej_pose2(self, p, v, a, r, q_near, tool, wobj, block, op=op_, def_acc = False):
        return self.client.movej_pose2(p, v, a, r, q_near, tool, wobj, block, op, def_acc)
    '''
     * @brief 控制机械臂末端从当前状态按照直线路径移动到目标状态
     * @param p 对应末端的位姿, 位置单位: m, 姿态以Rx、Ry、Rz表示, 单位: rad
     * @param v 末端速度, 单位: m/s
     * @param a 末端加速度, 单位: m/s^2
     * @param r 融合半径, 单位: m, 默认值为 0, 表示无融合.当数值大于0时表示与下一条运动融合
     * @param q_near 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool   设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj   设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block  是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movel(self, p, v, a, r, q_near, tool, wobj, block, op=op_, def_acc = False):
        return self.client.movel(p, v, a, r, q_near, tool, wobj, block, op, def_acc)
    '''
     * @brief 控制机械臂做圆弧运动, 起始点为当前位姿点, 途径p1点, 终点为p2点
     * @param p1 圆弧运动中间点
     * @param p2 圆弧运动结束点
     * @param v  末端速度, 单位: m/s
     * @param a  末端加速度, 单位: m/s^2
     * @param r  融合半径, 单位: m, 默认值为 0, 表示无融合.当数值大于0时表示与下一条运动融合
     * @param mode   姿态控制模式  0:姿态与终点保持一致;1:姿态与起点保持一致;2:姿态受圆心约束
     * @param q_near 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool   设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj   设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block  是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movec(self, p1, p2, v, a, r, mode, q_near, tool, wobj, block, op=op_, def_acc = False):
        return self.client.movec(p1, p2, v, a, r, mode, q_near, tool, wobj, block, op, def_acc)
    '''
     * @brief 控制机械臂做圆周运动, 起始点为当前位姿点, 途径p1点和p2点
     * @param p1 圆周运动经过点
     * @param p2 圆周运动经过点
     * @param v  末端速度, 单位: m/s
     * @param a  末端加速度, 单位: m/s^2
     * @param r  融合半径, 单位: m, 默认值为 0, 表示无融合.当数值大于0时表示与下一条运动融合
     * @param mode   姿态控制模式   1:姿态与终点保持一致;  2:姿态受圆心约束
     * @param q_near 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool   设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj   设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block  是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''    
    def move_circle(self, p1, p2, v, a, r, mode, q_near, tool, wobj, block, op=op_, def_acc = False):
        return self.client.move_circle(p1, p2, v, a, r, mode, q_near, tool, wobj, block, op, def_acc)
    '''
     * @brief 控制机械臂沿工具坐标系直线移动一个增量
     * @param pose_offset 工具坐标系下的位姿偏移量
     * @param v 直线移动的速度, 单位: m/s, 当x、y、z均为0时, 线速度按比例换算成角速度
     * @param a 加速度, 单位: m/s^2
     * @param r 融合半径, 单位: m, 默认值为 0, 表示无融合.当数值大于0时表示与下一条运动融合
     * @param tool   设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param block  是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def tcp_move(self, pose_offset, v, a, r, tool, block, op=op_, def_acc = False):
        return self.client.tcp_move(pose_offset, v, a, r, tool, block, op, def_acc)
    '''
     * @brief 控制机器人沿工具坐标系直线移动一个增量, 增量为p1与p2点之间的差, 运动的目标点为:当前点*p1^-1*p2
     * @param p1 工具坐标系下的位姿偏移量计算点1
     * @param p2 工具坐标系下的位姿偏移量计算点2
     * @param v  直线移动的速度, 单位: m/s, 当x、y、z均为0时, 线速度按比例换算成角速度
     * @param a  加速度, 单位: m/s^2
     * @param r  融合半径, 单位: m, 默认值为 0, 表示无融合.当数值大于0时表示与下一条运动融合
     * @param tool  设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj  设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def tcp_move_2p(self, p1, p2, v, a, r, tool, wobj, block, op=op_, def_acc = False):
        return self.client.tcp_move_2p(p1, p2, v, a, r, tool, wobj, block, op, def_acc)
    '''
     * @brief 样条运动函数, 控制机器人按照空间样条进行运动
     * @param pose_list 在设置工件坐标系下的末端位姿列表,最多不超过50个点
     * @param v 末端速度, 单位: m/s
     * @param a 末端加速度, 单位: m/s^2
     * @param tool  设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj  设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def spline(self, pose_list, v, a, tool, wobj, block, op=op_, r=0, def_acc = False):
        return self.client.spline(pose_list, v, a, tool, wobj, block, op, r, def_acc)
    '''
     * @brief 控制机械臂每个关节按照给定的速度一直运动, 函数执行后会直接运行后续指令.
	 *        运行speedj函数后, 机械臂会持续运动并忽略后续运动指令, 直到接收到speed_stop()函数后停止
     * @param joints_list 每个关节的速度, 单位: rad/s
     * @param a 主导轴的关节加速度, 单位: rad/s^2
     * @param time  运行时间, 到达时间后会停止运动,单位: ms.默认-1表示一直运行
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def speedj(self, joints_list, a, time, block):
        return self.client.speedj(joints_list, a, time, block)
    '''
     * @brief 控制机械臂末端按照给定的速度一直运动, 函数执行后会直接运行后续指令.
	 *        运行speedl函数后, 机械臂会持续运动并忽略后续运动指令, 直到接收到speed_stop()函数后停止
     * @param pose_list 末端速度向量, 线速度单位: m/s,角速度单位: rad/s
     * @param a 末端的线性加速度, 单位: rad/s^2
     * @param time  运行时间, 到达时间会停止运动, 单位: ms.默认-1表示一直运行
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def speedl(self, pose_list, a, time, block):
        return self.client.speedl(pose_list, a, time, block)
    '''
     * @brief 停止speedj及speedl函数的运动
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def speed_stop(self, block):
        return self.client.speed_stop(block)
    '''
     * @brief 控制机械臂从当前状态, 按照关节运动的方式移动到目标关节角状态,运动过程中不考虑笛卡尔空间路径
     * @param joints_list 目标关节角度, 单位: rad
     * @param v 最大关节角速度, 单位: m/s
     * @param a 最大关节加速度, 单位: m/s^2
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回, 可缺省, 默认为非阻塞
     * @param kp 比例参数, 默认值200, 可缺省, 建议使用默认参数
     * @param kd 微分参数, 默认值25, 可缺省, 建议使用默认参数
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def servoj(self, joints_list, v, a, block,kp=200, kd=25):
        return self.client.servoj(joints_list, v, a, block, kp, kd)
    '''
     * @brief 控制机械臂从当前状态, 按照关节运动的方式移动到目标笛卡尔状态,通过关节空间运动
     * @param pose_list 目标工件坐标系下的末端位姿, 单位: m, rad
     * @param v 关节速度, 单位: rad/s
     * @param a 关节加速度, 单位: m/s^2
     * @param q_near 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool   设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj   设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block  是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回, 可缺省, 默认为非阻塞
     * @param kp 比例参数, 默认值200, 可缺省, 建议使用默认参数
     * @param kd 微分参数, 默认值25, 可缺省, 建议使用默认参数
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def servoj_pose(self, pose_list, v, a, q_near, tool, wobj, block, kp=200, kd=25):
        return self.client.servoj_pose(pose_list, v, a, q_near, tool, wobj, block, kp, kd)
    '''
     * @brief 控制机械臂从当前状态, 按照关节运动的方式移动一个增量,通过关节空间运动
     * @param pose_offset 目标工件坐标系下的末端位姿, 单位: m, rad
     * @param v 关节速度, 范围(0, 1.25*PI], 单位: rad/s
     * @param a 关节加速度, 范围(0, 12.5*PI], 单位: rad/s^2
     * @param tool  设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回, 默认值false, 可缺省
     * @param kp 比例参数, 默认值200, 可缺省, 建议使用默认参数
     * @param kd 微分参数, 默认值25, 可缺省, 建议使用默认参数
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''    
    def servo_tcp(self, pose_offset, v, a, tool, block, kp=200, kd=25):
        return self.client.servo_tcp(pose_offset, v, a, tool, block, kp, kd)
    '''
     * @brief 控制机器人进入牵引示教模式
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def teach_mode(self, block):
        return self.client.teach_mode(block)
    '''
     * @brief 控制机器人退出牵引示教模式
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def end_teach_mode(self, block):
        return self.client.end_teach_mode(block)

    def modbus_add_signal(self, ip, slave_number, signal_address, signal_type, signal_name):
        return self.client.modbus_add_signal(ip, slave_number, signal_address, signal_type, signal_name)

    def modbus_delete_signal(self, signal_name):
        return self.client.modbus_delete_signal(signal_name)
    '''
     * @brief 读取modbus节点的数据, 返回值为double类型
     * @param signal_name modbus节点名
     * @return 节点返回值
     '''
    def modbus_read(self, signal_name):
        return self.client.modbus_read(signal_name)
    '''
     * @brief 对modbus节点进行写操作
     * @param signal_name modbus节点名
     * @param value 写入的数值, 寄存器节点取值为0-65535内的整数, 线圈节点取值为0或1
     * @return 返回当前任务结束时的状态
     '''
    def modbus_write(self, signal_name, value):
        return self.client.modbus_write(signal_name, value)
    '''
     * @brief 修改modbus节点的刷新频率, 默认频率为10Hz
     * @param signal_name modbus节点名
     * @param frequence 频率值, 取值范围:1~100Hz
     '''
    def modbus_set_frequency(self, signal_name, frequence):
        return self.client.modbus_set_frequency(signal_name, frequence)
    '''
     * @brief 读取机器人最新的错误列表
     * @return 错误列表
     '''
    def get_last_error(self):
        return self.client.get_last_error()
    '''
     * @brief 根据id查询当前的任务状态
     * @param id 任务的id
     * @return 任务的当前执行状态
     '''
    def get_noneblock_taskstate(self, id):
        return self.client.get_noneblock_taskstate(id)
    '''
     * @brief 插入log日志, 记录运行问题
     * @param message 日志描述
     '''
    def log_info(self, message):
        self.client.log_info(message)
    '''
     * @brief 在运行过程中产生弹窗, 并暂停当前所有任务
     * @param message 弹窗描述
     '''
    def log_error(self, message):
        self.client.log_error(message)
    '''
     * @brief 切换机器人到仿真或者真机模式
     * @param sim   true:仿真, false:真机
     * @param block
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def simulation(self, sim, block):
        return self.client.simulation(sim, block)
    '''
     * @brief 设置机器人全局速度
     * @param val 机器人全局速度, 范围[1,100]
     * @return 任务结束时状态
     '''
    def speed(self, val):
        return self.client.speed(val)
    '''
     * @brief 获取当前机器人状态
     * @return 机器人状态信息列表, data[0]表示机器人状态, data [1]表示程序状态, 
	 *             data [2]表示安全控制器状态, data [3]表示操作模式
     '''
    def get_robot_state(self):
        return self.client.get_robot_state()
    '''
     * @brief 获取当前状态下机械臂末端法兰在基坐标系下的位姿
     * @return 末端法兰位置
     '''
    def get_flange_pose(self):
        return self.client.get_flange_pose()
    '''
     * @brief 获取当前状态下机械臂末端法兰在基坐标系下的速度
     * @return 末端法兰速度列表, 单位: m/s,rad/s
     '''
    def get_flange_speed(self):
        return self.client.get_flange_speed()
    '''
     * @brief 获取当前状态下机械臂末端法兰在基坐标系下的加速度
     * @return 末端法兰加速度列表, 单位: m/s^2, rad/s^2
     '''
    def get_flange_acceleration(self):
        return self.client.get_flange_acceleration()
    '''
     * @brief 获取当前状态下机械臂工具末端点在基坐标系下的位姿
     * @return 末端位姿
     '''
    def get_tcp_pose(self):
        return self.client.get_tcp_pose()
    '''
     * @brief 获取当前状态下机械臂工具末端点的速度
     * @return 末端速度列表, 单位: m/s,rad/s
     '''
    def get_tcp_speed(self):
        return self.client.get_tcp_speed()
    '''
     * @brief 获取当前状态下机械臂工具末端点的加速度
     * @return 末端加速度列表, 单位: m/s^2, rad/s^2
     '''
    def get_tcp_acceleration(self):
        return self.client.get_tcp_acceleration()
    '''
     * @brief 获取当前末端的力矩信息
     * @return 末端力矩信息, [Fx,Fy,Fz,Mx,My,Mz],单位: N、N.m
     '''
    def get_tcp_force(self):
        return self.client.get_tcp_force()
    '''
     * @brief 获取当前状态下机械臂各关节的角度
     * @return 1-6轴关节角度列表, 单位: rad
     '''
    def get_actual_joints_position(self):
        return self.client.get_actual_joints_position()
    '''
     * @brief 获取当前状态下机械臂各关节的规划角度
     * @return 1-6轴目标关节角度列表, 单位: rad
     '''
    def get_target_joints_position(self):
        return self.client.get_target_joints_position()
    '''
     * @brief 获取当前状态下机械臂各关节角速度
     * @return 1-6轴关节速度列表, 单位: rad/s
     '''
    def get_actual_joints_speed(self):
        return self.client.get_actual_joints_speed()
    '''
     * @brief 获取当前状态下机械臂各关节规划角速度
     * @return 1-6轴目标关节速度列表, 单位: rad/s
     '''
    def get_target_joints_speed(self):
        return self.client.get_target_joints_speed()
    '''
     * @brief 获取当前状态下机械臂各关节角加速度
     * @return 1-6轴关节加速度列表, 单位: rad/s^2
     '''
    def get_actual_joints_acceleration(self):
        return self.client.get_actual_joints_acceleration()
    '''
     * @brief 取当前状态下机械臂各关节角规划加速度
     * @return 1-6轴关节加速度列表, 单位: rad/s^2
     '''
    def get_target_joints_acceleration(self):
        return self.client.get_target_joints_acceleration()
    '''
     * @brief 获取当前状态下机械臂各关节力矩
     * @return 1-6轴关节力矩列表, 单位: N.m
     '''
    def get_actual_joints_torque(self):
        return self.client.get_actual_joints_torque()
    '''
     * @brief 获取当前状态下机械臂各关节目标力矩
     * @return 1-6轴关节加速度列表, 单位: rad/s^2
     '''
    def get_target_joints_torque(self):
        return self.client.get_target_joints_torque()
    '''
     * @brief 停止轨迹记录
     * @return 任务结束时状态
     '''
    def stop_record_track(self):
        return self.client.stop_record_track()
    '''
     * @brief 开启轨迹记录,当超过允许记录的轨迹长度(针对基于位置记录)或允许记录的时长时(针对基于时间记录),
     *        会自动停止文件记录,并且暂停当前运行的程序.
	 *        文件会记录机器人的各关节弧度值和选定工具、工件坐标系下的笛卡尔空间位姿
     * @param name 轨迹名称
     * @param mode 轨迹类型, mode=0基于位置记录(与上一记录点所有关节偏移总量到达5°时记录新点);
	 *                       mode=1基于时间记录(与上一记录点间隔250ms记录新点)
     * @param tool 工具坐标系名称
     * @param wobj 工件坐标系名称
     * @return 当前任务的id
     '''
    def start_record_track(self, name, mode, tool, wobj):
        interval = 5.0
        if mode == 1:
            interval = 0.5
            
        return self.client.start_record_track(name, mode, tool, wobj, interval)
    '''
     * @brief 设置碰撞检测等级
     * @param value 0:关闭碰撞检测, 1-5:对应设置碰撞检测等级1到等级5
     * @return 任务结束时状态
     '''
    def collision_detect(self, value):
        return self.client.collision_detect(value)
    '''
     * @brief 对记录的轨迹基于关节空间(或基于笛卡尔空间)复现
     * @param name 轨迹名称
     * @param value 轨迹速度, (系统设定速度的百分比%), 取值范围(0,100]
     * @param mode 复现方式, 0:基于关节空间, 1:基于笛卡尔空间
     * @return 任务结束时状态
     '''
    def replay(self, name, value, mode):
        return self.client.replay(name, value, mode)
    '''
     * @brief 设置抓取负载.可以在程序运行过程中设置机器人当前的负载(质量、质心)
     * @param value 末端工具抓取负载质量, 质心, {mass,x_cog,y_cog,z_cog}, 
	 *              相对于工具坐标系, 质量范围[0, 35], 单位: kg, m
     * @return 任务结束时状态
     '''
    def set_load_data(self, value):
        return self.client.set_load_data(value)
    '''
     * @brief 控制机械臂开启末端力控.开启末端力控后所有运动函数除正常运动外,
	 *        会额外基于已配置的末端力控参数进行末端力控运动
     * @return 返回值代表当前任务的id信息
     '''
    def fc_start(self):
        return self.client.fc_start()
    '''
     * @brief 控制机械臂退出末端力控
     * @return 当前任务的id信息
     '''
    def fc_stop(self):
        return self.client.fc_stop()
    '''
     * @brief 修改并配置机器人末端力控参数
     * @param direction 6个笛卡尔空间方向末端力控开关, 开为true, 关为false
     * @param ref_ft 6个笛卡尔空间方向末端力控参考力, 范围[-1000, 1000], X/Y/Z方向单位: N, 
	                 RX/RY/RZ方向单位: Nm, 方向符号参考末端力控参考坐标系方向
     * @param damp  6个笛卡尔空间方向末端力控阻尼, 范围[-10000, 10000], 
	 *              X/Y/Z方向单位: N/(m/s), RX/RY/RZ方向单位: Nm/(°/s)
     * @param max_vel 6个笛卡尔空间方向末端力控最大调整速度, 范围[-5, 5], X/Y/Z方向单位: m/s, 
	 *                范围[-2*PI, 2*PI], RX/RY/RZ方向单位: rad/s
     * @param number_list 6个笛卡尔空间方向末端与环境接触力死区, 范围[-1000, 1000], 
	 *                     X/Y/Z方向单位: N, RX/RY/RZ方向单位: Nm
     * @param toolname 设置使用的末端力控工具的名称, 默认为当前使用的工具
     * @param wobjname 设置使用的末端力控工件坐标系的名称, 默认为当前使用的工件坐标系
     * @param value 末端力控参考坐标系选择标志位, 0为参考工具坐标系, 1位参考工件坐标系
     * @return 当前任务的id信息
     '''
    def fc_config(self, direction, ref_ft, damp, max_vel, toolname, wobjname, value):
        return self.client.fc_config(direction, ref_ft, damp, max_vel, toolname, wobjname, value)
    '''
     * @brief 控制机械臂仅产生末端力控运动, 为阻塞型指令
     * @return 任务结束时状态
     '''
    def fc_move(self):
        return self.client.fc_move()
    '''
     * @brief 控制机械臂在末端力控过程中进行力安全监控
     * @param direction 6个笛卡尔空间方向末端力安全监控开关, 开为true, 关为false
     * @param ref_ft 6个笛卡尔空间方向末端力安全监控参考力, X/Y/Z方向单位: N, RX/RY/RZ方向单位: Nm, 
	 *               方向符号参考末端力安全监控参考坐标系方向
     * @param toolname 设置使用的末端力安全监控工具的名称
     * @param wobjname 设置使用的末端力安全监控工件坐标系的名称
     * @param type 末端力安全监控参考坐标系选择标志位, 0为参考工具坐标系, 1位参考工件坐标系
     * @param force_property 监控力属性, 0为末端负载力及外力, 1为末端外力(不含负载),可缺省, 默认为0
     * @return 当前任务的id信息
     '''
    def fc_guard_act(self, direction, ref_ft, toolname, wobjname, type, force_property = 0):
        return self.client.fc_guard_act(direction, ref_ft, toolname, wobjname, type, force_property)
    '''
     * @brief 控制机械臂在末端力控过程中禁用力安全监控
     * @return 当前任务的id信息
     '''
    def fc_guard_deact(self):
        return self.client.fc_guard_deact()
    '''
     * @brief 控制机械臂末端力传感器读数设置为指定值
     * @param direction 6个末端力传感器输出力设置标志位, 需要设置为true, 不需要设置为false
     * @param ref_ft 6个末端力传感器输出力设置目标值, X/Y/Z方向单位: N, RX/RY/RZ方向单位: Nm
     * @return 当前任务的id信息
     '''
    def fc_force_set_value(self, direction, ref_ft):
        return self.client.fc_force_set_value(direction, ref_ft)
    '''
     * @brief 控制机械臂在执行fc_start()函数后的末端力控过程中满足指定位置判断条件时自动停止当前运动函数并调过后续运动函数, 
	 *        直到fc_stop()函数被执行停止末端力控
     * @param middle 位置判断条件绝对值, X/Y/Z方向单位: m, RX/RY/RZ方向单位: rad
     * @param range  位置判断条件偏移范围大小, X/Y/Z方向单位: m, RX/RY/RZ方向单位: rad
     * @param absolute 绝对/增量条件判断标志位, true为绝对位置判断, false为增量位置判断
     * @param duration 条件满足触发保持时间, 单位: ms
     * @param timeout  条件满足触发超时时间, 单位: ms
     * @return 当前任务的id信息
     '''
    def fc_wait_pos(self, middle, range, absolute, duration, timeout):
        return self.client.fc_wait_pos(middle, range, absolute, duration, timeout)
    '''
     * @brief 控制机械臂在执行fc_start()函数后的末端力控过程中满足指定速度判断条件时自动停止当前运动函数并跳过后续运动函数, 
	 *        直到fc_stop()函数被执行停止末端力控
     * @param middle 速度判断条件绝对值, X/Y/Z方向范围[-5, 5], 单位: m/s, RX/RY/RZ方向范围[-2*PI, 2*PI], 单位: rad/s
     * @param range  速度判断条件偏移范围大小, X/Y/Z方向单位: m/s, RX/RY/RZ方向单位: rad/s
     * @param absolute 绝对/增量条件判断标志位, true为绝对速度判断, false为增量速度判断
     * @param duration 条件满足触发保持时间, 单位: ms
     * @param timeout  条件满足触发超时时间, 单位: ms
     * @return 当前任务的id信息
     '''
    def fc_wait_vel(self, middle, range, absolute, duration, timeout):
        return self.client.fc_wait_vel(middle, range, absolute, duration, timeout)
    '''
     * @brief 控制机械臂在执行fc_start()函数后的末端力控过程中满足指定力判断条件时自动停止当前运动函数并跳过后续运动函数, 
	 *        直到fc_stop()函数被执行停止末端力控
     * @param middle 力判断条件绝对值, 范围[-1000, 1000], X/Y/Z方向单位: N, RX/RY/RZ方向单位: Nm
     * @param range  力判断条件偏移范围大小, X/Y/Z方向单位: N, RX/RY/RZ方向单位: Nm
     * @param absolute 绝对/增量条件判断标志位, true为绝对力判断, false为增量力判断
     * @param duration 条件满足触发保持时间, 单位: ms
     * @param timeout  条件满足触发超时时间, 单位: ms
     * @return 当前任务的id信息
     '''
    def fc_wait_ft(self, middle, range, absolute, duration, timeout):
        return self.client.fc_wait_ft(middle, range, absolute, duration, timeout)
    '''
     * @brief 控制机械臂在执行fc_start()函数后的末端力控过程中位置条件判断、速度条件判断与力条件判断间的逻辑关系.不配置时默认三个条件判断都禁用
     * @param value 三维整形列表, 0代表不启用, 1代表与逻辑, 2代表或逻辑.例如开启位置条件判断, 禁用速度条件判断, 开启力条件判断, 并且位置与力的关系为或, 则输入[1,0,2]
     * @return 当前任务的id信息
     '''
    def fc_wait_logic(self, value):
        return self.client.fc_wait_logic(value)
    '''
     * @brief 获取当前机器人末端传感器的反馈读数
     * @return 6自由度末端力读数, X/Y/Z方向单位: N, RX/RY/RZ方向单位: Nm
     '''
    def fc_get_ft(self):
        return self.client.fc_get_ft()
    '''
     * @brief 获取当前机器人末端力控功能启用状态
     * @return 机器人末端力控启用返回true, 未启用返回false
     '''
    def fc_mode_is_active(self):
        return self.client.fc_mode_is_active()
    '''
     * @brief 控制机械臂开启速度优化功能.开启该功能后, 在满足系统约束前提下,
	 *        机械臂以尽可能高的速度跟踪路径
     * @return 当前任务结束时的状态
     '''
    def enable_speed_optimization(self):
        return self.client.enable_speed_optimization()
    '''
     * @brief 控制机械臂退出速度优化
     * @return 当前任务结束时的状态
     '''
    def disable_speed_optimization(self):
        return self.client.disable_speed_optimization()
    '''
     * @brief 将一组points点位信息输入到机器人控制器中的轨迹池
     * @param track 一组points点位信息.每个point以6个double类型数据构成
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def trackEnqueue(self, track, block):
        return self.client.trackEnqueue(track, block)
    '''
     * @brief 将机器人控制器中的轨迹池清空
     * @return 当前任务结束时的状态
     '''    
    def trackClearQueue(self):
        return self.client.trackClearQueue()
    '''
     * @brief 获取机器人控制器中的当前轨迹池大小
     * @return 当前轨迹池大小
     '''    
    def getQueueSize(self):
        return self.client.getQueueSize()
    '''
     * @brief 执行时, 机器人的各关节将顺序到达轨迹池中的点位值直到轨迹池中无新的点位.
     *        执行过程中, 主导关节(关节位置变化最大的关节)将以speed与acc规划运动, 其他关节按比例缩放.
     *        注:如果已经开始执行停止规划, 将不再重新获取轨迹池中的数据, 直到完成停止.
     *            停止后如果轨迹池中有新的点位, 将重新执行跟随.为保证运动连续性, 建议至少保证轨迹池中有10个数据
     * @param speed 最大关节速度, 单位: rad/s
     * @param acc   最大关节加速度, 单位: rad/s^2
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def trackJointMotion(self, speed, acc, block):
        return self.client.trackJointMotion(speed, acc, block)
    '''
     * @brief 执行时, 机器人的工具末端tool将顺序到达轨迹池中的点位值直到轨迹池中无新的点位.
     *        执行过程中, 工具末端tool将以speed与acc在工件坐标系wobj下规划运动.
     *        注:如果已经开始执行停止规划, 将不再重新获取轨迹池中的数据, 直到完成停止.
                  停止后如果轨迹池中有新的点位, 将重新执行跟随.为保证运动连续性, 建议至少保证轨迹池中有10个数据
     * @param speed 最大末端速度, 单位: m/s
     * @param acc   最大末端加速度, 单位: m/s^2
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @param tool  设置使用的工件坐标系的名称, 为空字符串时默认为当前使用的工件坐标系
     * @param wobj  设置使用的工具的名称,为空字符串时默认为当前使用的工具
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def trackCartMotion(self, speed, acc, block, tool, wobj):
        return self.client.trackCartMotion(speed, acc, block, tool, wobj, 0.005)
    '''
     * @brief 确保机器人远程连接断开时, 机器人自动产生一条stop指令以停止当前运动.使用该函数需要单独创建一个线程周期性调用
     * @param time 心跳延时时间, 单位: ms
     '''    
    def rpc_heartbeat(self,time=1000):
        return self.client.rpc_heartbeat(time)
    '''
     * @brief 通过参数或者结束点两种设置方式, 在笛卡尔空间做螺旋轨迹运动
     * @param p1   螺旋线中心点位姿
     * @param p2   螺旋线的目标点位姿, 参数设置模式时不参考此参数
     * @param rev  总旋转圈数, rev < 0, 表示顺时针旋转;rev > 0, 表示逆时针旋转
     * @param len  轴向移动距离, 正负号遵循右手定则, 结束点设置模式时不参考此参数, 单位: m
     * @param r    目标点半径, 结束点设置模式时不参考此参数, 单位: m
     * @param mode 螺旋线示教模式, 0:参数设置, 1:结束点设置
     * @param v    末端速度, 范围(0, 5], 单位: m/s
     * @param a    末端加速度, 范围(0, ∞], 单位: m/s^2
     * @param q_near 目标点位置对应的关节角度, 用于确定逆运动学选解, 单位: rad
     * @param tool   设置使用的工具的名称, 为空字符串时默认为当前使用的工具
     * @param wobj   设置使用的工件坐标系的名称, 为空字符串默认为当前使用的工件坐标系
     * @param block  指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op     详见上方Op特殊类型说明,可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''        
    def move_spiral(self, p1, p2, rev, len, r, mode, v, a, q_near, tool, wobj, block, op=op_, def_acc = False):
        return self.client.move_spiral(p1, p2, rev, len, r, mode, v, a, q_near, tool, wobj, block, op, def_acc)
    '''
     * @brief 控制机械臂开启加速度优化功能.开启该功能后, 系统会根据机器人动力学模型、电功率模型计算得到最优加速度大小,
     *        在满足速度约束前提下, 机械臂以尽可能高的加速度进行规划.当速度优化同时打开后, 该函数不起作用
     * @return 当前任务结束时的状态
     '''        
    def enable_acc_optimization(self):
        return self.client.enable_acc_optimization()
    '''
     * @brief 控制机械臂退出加速度优化
     * @return 当前任务结束时的状态
     '''        
    def disable_acc_optimization(self):
        return self.client.disable_acc_optimization()
    '''
     * @brief 设置控制柜上的模拟电压输出
     * @param num   控制柜上的模拟电压通道序号, 范围从1-4
     * @param value 设置的模拟电压值
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def set_standard_analog_voltage_out(self, num, value, block):
        return self.client.set_standard_analog_current_out(num, value, block)
    '''
     * @brief 设置控制柜上的模拟电流输出
     * @param num   控制柜上的模拟电流通道序号, 范围从1-4
     * @param value 设置的模拟电流值
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def set_standard_analog_current_out(self, num, value, block):
        return self.client.set_standard_analog_voltage_out(num, value, block)
    '''
     * @brief 设置485的波特率
     * @param value 波特率
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def set_baudrate_485(self, value, block):
        return self.client.set_baudrate_485(value, block)
    '''
     * @brief 设置CAN的波特率
     * @param value 波特率
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return  阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def set_baudrate_can(self, value, block):
        return self.client.set_baudrate_can(value, block)
    '''
     * @brief set_analog_output_mode
     * @param num
     * @param mode
     * @param block
     * @return
     '''    
    def set_analog_output_mode(self, num, mode, block):
        return self.client.set_analog_output_mode(num, mode, block)
    '''
     * @brief 判断机器人是否在运动
     * @return True:机器人在运动, False:机器人没有运动
     '''    
    def robotmoving(self):
        return self.client.robotmoving()
    '''
     * @brief 对多线圈进行写操作
     * @param slave_num modbus节点号
     * @param name modbus节点名
     * @param len 需要写入数据的线圈长度
     * @param byte_list 需要写入的数据
     * @return 任务结束时状态
     '''    
    def modbus_write_multiple_coils(self, slave_num, name, len, byte_list):
        return self.client.modbus_write_multiple_coils(slave_num, name, len, byte_list)
    '''
     * @brief 对多寄存器进行写操作
     * @param slave_num modbus节点号
     * @param name modbus节点名
     * @param len 需要写入数据的寄存器长度
     * @param word_list 需要写入的数据
     * @return 任务结束时状态
     '''    
    def modbus_write_multiple_regs(self, slave_num, name, len, word_list):
        return self.client.modbus_write_multiple_regs(slave_num, name, len, word_list)
    '''
     * @brief 获取当前工程的路径
     * @param project_path 当前工程路径
     '''    
    def get_current_project(self):
        return self.client.get_current_project()
    '''
     * @brief 获取指定路径下的文件列表
     * @param fileslist 文件列表和类型;0:文件夹;1:文件
     * @param path 当前工程路径
     '''    
    def get_files_list(self, path):
        return self.client.get_files_list(path)
    '''
     * @brief 获取当前RPC库的版本号
     * @return 当前RPC库的版本号
     '''    
    def get_version(self):
        return "3.4.0"
    '''
     * @brief 获取机器人当前的位姿等信息
     * @param status 机器人位姿等信息
     '''    
    def getRobotStatus(self):
        return self.client.getRobotStatus()
    '''
     * @brief 获取机器人当前IO和寄存器信息
     * @param status 当前IO和寄存器信息
     '''    
    def getRobotIOStatus(self):
        return self.client.getRobotIOStatus()
    
    def restart(self, block):
        return self.client.restart(block)
    '''
     * @brief 获取末端法兰在工具坐标系和工件坐标系下的位姿
     * @param tool 工具坐标系名称, 为空字符串默认为当前使用的坐标系
     * @param wobj 工件坐标系名称, 为空字符串默认为当前使用的坐标系
     * @return 末端法兰的位姿
     '''    
    def get_tcp_pose_coord(self, tool, wobj):
        return self.client.get_tcp_pose_coord(tool, wobj)
    '''
     * @brief 获取机械臂工具末端在工具坐标系下的力矩信息
     * @param tool 工具坐标系名称, 默认为当前使用的坐标系
     * @return 末端力矩信息, [Fx,Fy,Fz,Mx,My,Mz],单位: N、N.m
     '''    
    def get_tcp_force_tool(self, tool):
        return self.client.get_tcp_force_tool(tool)
    '''
     * @brief 修改伺服参数
     * @param axis_num 关节索引号
     * @param id       参数的ID号
     * @param value    要设置的值
     * @param qfmt     要设置的qfmt值
     * @param block    指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def set_servo_config(self, axis_num, id, value, qfmt):
        return self.client.set_servo_config(axis_num, id, value, qfmt)
    '''
     * @brief 将伺服参数应用到实际控制
     * @param axis_num 关节索引号
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def apply_servo_config(self, axis_num, block):
        return self.client.apply_servo_config(axis_num, block)
    '''
     * @brief 获取电机极对数
     * @return 电机极对数
     '''    
    def get_motor_pole_pair_number(self):
        return self.client.get_motor_pole_pair_number()
    '''
     * @brief 获取机器人轴电机定子插槽编号
     * @return 轴电机定子插槽编号
     '''    
    def get_motor_stator_slots(self):
        return self.client.get_motor_stator_slots()
    '''
     * @brief 获取机器人轴减速器比率
     * @return 轴减速器比率
     '''   
    def get_axis_ratio(self):
        return self.client.get_axis_ratio()
    '''
     * @brief 重置碰撞检测警告
     * @return 任务结束时状态
     '''    
    def collision_detection_reset(self):
        return self.client.collision_detection_reset()
    '''
     * @brief 控制机械臂在关节或者笛卡尔空间做点动
     * @param param Jog运动的相关参数, 参考MoveJogTaskParams
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def move_jog(self, param, block):
        return self.client.move_jog(param, block)
    '''
     * @brief 结束机械臂的关节或者笛卡尔Jog
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def stop_manual_move(self, block):
        return self.client.stop_manual_move(block)
    '''
     * @brief 获取机器人控制器的软件版本号
     * @return 机器人控制器的软件版本号
     '''    
    def get_robot_version(self):
        return self.client.get_robot_version()
    '''
     * @brief 启用或禁用示教器的物理按键
     * @param enable true:启动示教器物理按键, false:禁用示教器物理按键
     * @return 任务结束时状态
     '''    
    def set_teach_pendant(self, enable):
        return self.client.set_teach_pendant(enable)
    '''
     * @brief 获取示教速度的百分比
     * @return 示教速度的百分比
     '''    
    def get_teach_speed(self):
        return self.client.get_teach_speed()
    '''
     * @brief 获取全局速度的百分比
     * @return 全局速度的百分比
     '''
    def get_global_speed(self):
        return self.client.get_global_speed()
    '''
     * @brief 设置示教速度的百分比
     * @param v 示教速度的百分比, 范围[1,100]
     * @return 任务结束时状态
     '''
    def set_teach_speed(self, v):
        return self.client.set_teach_speed(v)
    '''
     * @brief 设置复合运动的相关参数
     * @param type 复合运动类型.1:平面三角形轨迹, 2:平面正旋轨迹,
     *                           3:平面圆形轨迹, 4:平面梯形轨迹, 5:平面8字形轨迹
     * @param ref_plane  参考平面, 0:工具XOY, 1:工具XOZ
     * @param fq 频率, 单位: Hz
     * @param amp 振幅, 单位: m
     * @param el_offset 仰角偏移, 单位: m.(参数预留)
     * @param az_offset 方向角偏移, 单位: m.(参数预留)
     * @param up_height 中心隆起高度, 单位: m.(参数预留)
     * @param time 左右停留时间
     * @param op_list 二维的OP参数列表
     * @return 任务结束时状态
     '''    
    def combine_motion_config(self, type, ref_plane, fq, amp, el_offset, az_offset, up_height, time, op_list):
        return self.client.combine_motion_config(type, ref_plane, fq, amp, el_offset, az_offset, up_height, time, op_list)
    '''
     * @brief 开启复合运动
     * @return 任务结束时状态
     '''
    def enable_combine_motion(self):
        return self.client.enable_combine_motion()
    '''
     * @brief 结束复合运动
     * @return 任务结束时状态
     '''
    def disable_combine_motion(self):
        return self.client.disable_combine_motion()
    '''
     * @brief 开启机械臂奇异点规避功能
     * @return 任务结束时状态
     '''
    def enable_singularity_control(self):
        return self.client.enable_singularity_control()
    '''
     * @brief 关闭机械臂奇异点规避功能
     * @return 任务结束时状态
     '''
    def disable_singularity_control(self):
        return self.client.disable_singularity_control()
    '''
     * @brief 启动外部轴方案
     * @param scheme_name 外部轴方案名称
     * @return 任务结束时的状态
     '''
    def enable_eaxis_scheme(self, scheme_name):
        return self.client.enable_eaxis_scheme(scheme_name)
    '''
     * @brief 结束外部轴方案
     * @param scheme_name 外部轴方案名称
     * @return 任务结束时的状态
     '''    
    def disable_eaxis_scheme(self, scheme_name):
        return self.client.disable_eaxis_scheme(scheme_name)
    '''
     * @brief 控制外部轴移动
     * @param scheme_name 目标外部轴方案名称
     * @param epose 目标外部轴方案所对应自由度位置(三维), 
	 *              记录位置自由度及单位根据外部轴方案所设置自由度及外部轴方案类型改变, 单位: rad, m
     * @param v     外部轴最大规划速度, 根据对应外部轴方案类型改变, 单位: rad/s, m/s
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op    详见上方Op特殊类型说明(距离触发无效),可缺省参数
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def move_eaxis(self, scheme_name, epose, v, block, op):
        return self.client.move_eaxis(scheme_name, epose, v, block, op)
    '''
     * @brief 控制外部轴和机器人执行关节运动
     * @param joints_list 目标关节位置, 单位: rad
     * @param v 关节角速度, 单位: rad/s
     * @param a 关节加速度, 单位: rad/s^2
     * @param rad  融合半径, 单位: m
     * @param scheme_name 目标外部轴方案名称
     * @param epose 目标外部轴方案所对应自由度位置(三维), 
	 *              记录位置自由度及单位根据外部轴方案所设置自由度及外部轴方案类型改变, 单位: rad, m
     * @param eaxis_v 外部轴最大规划速度, 根据对应外部轴方案类型改变, 单位: rad/s, m/s
     * @param block   指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op      详见上方Op特殊类型说明(距离触发无效), 可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''    
    def movej2_eaxis(self, joints_list, v, a, rad, scheme_name, epose, eaxis_v, block, op = op_, def_acc = False):
        return self.client.movej2_eaxis(joints_list, v, a, rad, scheme_name, epose, eaxis_v, block, op, def_acc)
    '''
     * @brief 控制外部轴和机器人从当前状态, 按照关节运动的方式移动到末端目标位置
     * @param p 对应末端的位姿, 位置单位: m, 姿态以Rx、Ry、Rz表示, 范围[-2*PI, 2*PI], 单位: rad
     * @param v 关节角速度, 单位: rad/s
     * @param a 关节加速度, 单位: rad/s^2
     * @param rad 融合半径, 单位: m
     * @param qnear 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool  设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj  设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param scheme_name 目标外部轴方案名称
     * @param epose 目标外部轴方案所对应自由度位置(三维), 记录位置自由度及单位根据外部轴方案所设置自由度及外部轴方案类型改变, 单位: rad, m
     * @param eaxis_v 外部轴最大规划速度, 根据对应外部轴方案类型改变, 单位: rad/s, m/s
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 详见上方Op特殊类型说明(距离触发无效),可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movej2_pose_eaxis(self, p, v, a, rad, qnear, tool, wobj, scheme_name, epose, eaxis_v, block, op = op_, def_acc = False):
        return self.client.movej2_pose_eaxis(p, v, a, rad, qnear, tool, wobj, scheme_name, epose, eaxis_v, block, op, def_acc)
    '''
     * @brief 控制外部轴和机器人从当前状态按照直线路径移动到目标状态
     * @param p 对应末端的位姿, 位置单位: m, 姿态以Rx、Ry、Rz表示, 范围[-2*PI, 2*PI], 单位: rad
     * @param v 末端速度, 范围(0, 5], 单位: m/s
     * @param a 末端加速度, 范围(0, ∞], 单位: m/s^2
     * @param rad 融合半径, 单位: m
     * @param qnear 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool  设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj  设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param scheme_name 目标外部轴方案名称
     * @param epose 目标外部轴方案所对应自由度位置(三维), 记录位置自由度及单位根据外部轴方案所设置自由度及外部轴方案类型改变, 单位: rad, m
     * @param eaxis_v 外部轴最大规划速度, 根据对应外部轴方案类型改变, 单位: rad/s, m/s
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 详见上方Op特殊类型说明(距离触发无效),可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movel_eaxis(self, p, v, a, rad, qnear, tool, wobj, scheme_name, epose, eaxis_v, block, op = op_, def_acc = False):
        return self.client.movel_eaxis(p, v, a, rad, qnear, tool, wobj, scheme_name, epose, eaxis_v, block, op, def_acc)
    '''
     * @brief 控制外部轴和机器人做圆弧运动, 起始点为当前位姿点, 途径p1点, 终点为p2点
     * @param p1 圆弧运动中间点位姿
     * @param p2 圆弧运动结束点位姿
     * @param v 末端速度, 范围(0, 5], 单位: m/s
     * @param a 末端加速度, 范围(0, ∞], 单位: m/s^2
     * @param rad 融合半径, 单位: m
     * @param qnear 目标点附近位置对应的关节角度, 用于确定逆运动学选解, 为空时使用当前位置
     * @param tool  设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj  设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param scheme_name 目标外部轴方案名称
     * @param epose 目标外部轴方案所对应自由度位置(三维), 记录位置自由度及单位根据外部轴方案所设置自由度及外部轴方案类型改变, 单位: rad, m
     * @param eaxis_v 外部轴最大规划速度, 根据对应外部轴方案类型改变, 单位: rad/s, m/s
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 详见上方Op特殊类型说明(距离触发无效),可缺省参数
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态, 若无融合为Finished, 若有融合为Interrupt.
	 *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''
    def movec_eaxis(self, p1, p2, v, a, rad, qnear, tool, wobj, scheme_name, epose, eaxis_v, block, op = op_, def_acc = False):
        return self.client.movec_eaxis(p1, p2, v, a, rad, qnear, tool, wobj, scheme_name, epose, eaxis_v, block, op, def_acc)
    '''
     * @brief 可达性检查
     * @param base 基坐标在世界坐标系中的位置
     * @param wobj 工件坐标系在世界坐标系中的位置
     * @param tool 工具坐标系在法兰坐标系中的描述
     * @param ref_pos 机器人关节参考角度
     * @param check_points 需要确认可达性检查的点位列表
     * @return 可达性确认结果
     '''    
    def reach_check(self, base, wobj, tool, ref_pos, check_points):
        return self.client.reach_check(base, wobj, tool, ref_pos, check_points)
    '''
     * @brief 控制外部轴和机器人执行点动
     * @param name 目标外部轴方案名称
     * @param direction 运动方向, -1:负方向, 1:正方向
     * @param vel 速度百分比
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''    
    def move_jog_eaxis(self, name, direction, vel, block):
        return self.client.move_jog_eaxis(name, direction, vel, block)
    '''
     * @brief 获取外部轴当前位置和激活状态信息
     * @param info 当前位置和激活状态信息
     '''    
    def get_eaxis_info(self):
        return self.client.get_eaxis_info()
    '''
     * @brief 设置牵引时的参数
     * @param space        牵引类型
     * @param joint_scale  关节柔顺度
     * @param cart_scale   笛卡尔柔顺度
     * @param coord_type   类型
     * @param direction    牵引方向激活
     * @return 任务结束时的状态
     '''    
    def set_hand_teach_parameter(self, space, joint_scale, cart_scale, coord_type, direction):
        return self.client.set_hand_teach_parameter(space, joint_scale, cart_scale, coord_type, direction)
    '''
     * @brief 示教器按键的jog类型
     * @param type 1:关节, 2:笛卡尔
     * @return 任务结束时的状态
     '''    
    def set_pendant_type(self, type):
        return self.client.set_pendant_type(type)
    '''
     * @brief 开启末端震动抑制功能
     * @return 任务结束时状态
     '''    
    def enable_vibration_control(self):
        return self.client.enable_vibration_control()
    '''
     * @brief 关闭末端震动抑制功能
     * @return 任务结束时状态
     '''    
    def disable_vibration_control(self):
        return self.client.disable_vibration_control()
    '''
     * @brief 设置融合预读取配置
     * @param per 百分比(%)
     * @return 任务结束时的状态
     '''    
    def set_blend_ahead(self, per):
        return self.client.set_blend_ahead(per)
    '''
     * @brief 开启实时控制模式 (不支持)
     * @param mode  实时控制模式, 1:关节位置, 2:关节速度, 3:空间位置, 4:空间速度
     * @param filter_bandwidth 实时控制指令滤波器带宽, 单位Hz, 默认100Hz
     * @param com_lost_time 实时控制通讯数据丢失监控保护时间, 单位s, 默认0.02s
     * @return 任务结束时的状态
     '''    
    def start_realtime_mode(self, mode, fileter_bandwidth, com_lost_time):
        return self.client.start_realtime_mode(mode, fileter_bandwidth, com_lost_time)
    '''
     * @brief 结束实时控制模式 (不支持)
     * @return 任务结束时的状态
     '''    
    def end_realtime_mode(self):
        return self.client.end_realtime_mode()
    '''
     * @brief 实时数据入队 (不支持)
     * @param realtime_data 实时数据
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 任务结束时的状态
     '''    
    def realtime_data_enqueue(self, realtime_data, block):
        return self.client.realtime_data_enqueue(realtime_data, block)
    '''
     * @brief 清空实时数据队列 (不支持)
     * @return 任务结束时的状态
     '''    
    def clear_realtime_data_queue(self):
        return self.client.clear_realtime_data_queue()
    '''
     * @brief 获取当前实时队列池数据的数量 (不支持)
     * @return 当前实时队列池数据的数量
     '''    
    def get_realtime_data_queue_size(self):
        return self.client.get_realtime_data_queue_size()
    '''
     * @brief 样条运动函数, 控制机器人按照空间样条进行运动, 在运动过程中触发对应点位的OP操作
     * @param pose_list 在设置工件坐标系下的末端位姿和OP列表, 最多不超过50个点
     * @param v 末端速度, 单位: m/s
     * @param a 末端加速度, 单位: m/s^2
     * @param tool  设置使用的工具的名称, 为空时默认为当前使用的工具
     * @param wobj  设置使用的工件坐标系的名称, 为空时默认为当前使用的工件坐标系
     * @param block 是否阻塞, 如果为false表示非阻塞指令, 指令会立即返回
     * @param op 可缺省参数
     * @param r  融合半径, 可缺省参数, 单位: m, 默认值为 0, 表示无融合.当数值大于0时表示与下一条运动融合
     * @param def_acc 是否使用系统默认加速度, false表示使用自定义的加速度值, true表示使用系统自动规划的加速度值, 可缺省, 默认为false
     * @return 当配置为阻塞执行, 返回值代表当前任务结束时的状态.
     *         当配置为非阻塞执行, 返回值代表当前任务的id, 用户可以调用get_noneblock_taskstate(id)函数查询当前任务的执行状态
     '''    
    def spline_op(self, pose_list, v, a, tool, wobj, block, op=op_, r=0, def_acc = False):
        return self.client.spline_op(pose_list, v, a, tool, wobj, block, op, r, def_acc)
    '''
     * @brief 将一组points点位和该点位下的OP信息输入到机器人控制器中的轨迹池, 在运动过程中触发对应点位的OP操作
     * @param track 点位信息和该点位下的OP列表.
     * @param block 指令是否阻塞型指令, 如果为false表示非阻塞指令, 指令会立即返回
     * @return 阻塞执行代表任务结束时状态, 非阻塞执行代表任务的ID
     '''
    def trackEnqueueOp(self, track, block):
        return self.client.trackEnqueueOp(track, block)
    '''
     * @brief 手自动模式切换
     * @param mode 0:手动模式, 1:自动模式.
     * @return 阻塞执行代表任务结束时状态
     '''
    def switch_mode(self, mode):
        return self.client.switch_mode(mode)
