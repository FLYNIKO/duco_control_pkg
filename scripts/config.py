'''雷达安装位置参数(相对于机械臂末端)'''
# 机械臂运动为  前后X（后），左右Y（右），上下Z（上）
#前后z 上下y 左右x （定义方法是机械臂的rx、ry、rz 0点的XYZ坐标系 ，雷达放置为雷达正方向对向X、Z正方向）
MAIN_RADAR_OFFSET =     [ 0.0, -0.14, -0.05, 1.57, 1.57, 0] # 主雷达偏移
LEFT_RADAR_OFFSET =     [-0.28, 0.0, 0.13, 0.0, 1.57, 0] # 左雷达偏移
RIGHT_RADAR_OFFSET =    [ 0.24, 0.0, 0.15, 0.0, 1.57, 0] # 右雷达偏移
'''避障区域参数'''
# 根据机械臂初始位置姿态的坐标系进行区域划分，当前为后x正，右y正，上z正
OB_THRESHOLD_L = -0.35 # 左区阈值
OB_THRESHOLD_R = 0.35 # 右区阈值
OB_THRESHOLD_M = 0.35 # 中区阈值（左右区绝对值）
OB_THRESHOLD_U = 0.2 # 上区阈值
OB_THRESHOLD_D = -0.2 # 下区阈值
OB_THRESHOLD_FRONT_MID = -0.2 # 前后分界线
OB_THRESHOLD_MID_REAR = 0.12
OB_SAFE_DISTANCE_F = 0.6 # 上、下、前安全距离
OB_SAFE_DISTANCE = 0.85 # 安全距离
OB_FORWARD_MIN = -0.1 # 前方最小距离
OB_FORWARD_MAX = -2 # 前方最大距离

# IP = '192.168.100.10' # 虚拟机IP地址 虚拟机
IP = '192.168.0.168' # 机械臂IP地址 0
# IP = '192.168.0.96' # 机械臂IP地址 1

PORT = 7003 # 机械臂端口号
DEBUG_MODE = False # 是否显示openCV调试窗口

AUTOSPEED = 0.2 # 自动喷涂速度
DEFAULT_VEL = 0.2 # 机械臂末端手动速度
DEFAULT_ACC = 0.3 # 机械臂末端加速度
OB_ACC = 0.6 # 避障加速度
OB_VELOCITY = 0.36 # 避障速度
KP = 0.68
KI = 0.0
KD = 0.05
DEADZONE = 0.02 # PID死区 (mm)

PAINTDEG = 90 # 喷涂角度(圆柱)
PAINTWIDTH = 0.15 # 喷涂宽度(圆柱)

INIT_POS = [-0.8, -0.2, 1.2, -1.57, 0.0, 1.57] # 初始位置
SERV_POS = [-1.0, -0.2, -0.2, -1.57, 0.0, 1.57] # 维修位置
CLOG_POS = [-1.0, -0.2, 0.2, -1.57, 0.0, 1.57] # 堵枪位置
SAFE_POS = [-0.8, -0.2, 1.2, -1.57, 0.0, 1.57] # 安全位置

KEYTIMEOUT = 2 # 键盘输入超时时间(s)
TIMEOUT = 2 # 传感器超时时间(s)

'''
'/obstacle_avoidance/pointcloud'话题： 避障点云
'/Duco_state'话题： 机械臂状态

'/left_radar/scan'话题： 左雷达数据
'/left_radar/filtered_scan'话题： 左雷达数据(滤波后)
'/right_radar/scan'话题： 右雷达数据
'/right_radar/filtered_scan'话题： 右雷达数据(滤波后)
'/main_radar/scan'话题： 主雷达数据
'/main_radar/filtered_scan'话题： 主雷达数据(滤波后)

'/obstacle_flags'话题： 避障标志(自定义消息)
'/twin_radar/line_detection_info'话题： 双线检测信息(自定义消息)
'/left_radar/H_detection_info'话题： 左雷达H检测信息(自定义消息)
'''