

IP = '192.168.100.10' # 机械臂IP地址--上
PORT = 7003 # 机械臂端口号

"""传感器防撞阈值（mm），若阈值为0则不开启防撞"""
ANTICRASH_UP = 0 # --上
ANTICRASH_FRONT = 600 # 同时用作喷涂距离--上
ANTICRASH_LEFT = 0 # --上
ANTICRASH_RIGHT = 0 # --上

AUTOSPEED = 0.2 # 自动喷涂速度
DEFAULT_VEL = 0.2 # 机械臂末端速度
DEFAULT_ACC = 0.8 # 机械臂末端加速度
KP = 0.005
KI = 0.0
KD = 0.0001
DEADZONE = 20 # PID死区 (mm)

SCAN_RANGE = 0.7 # 扫描范围(m)_比钢梁高度大0.2--上
SCAN_STEP = 0.01 # 扫描步长(m)_default: 0.01
SCAN_PAUSE = 0.05 # 扫描暂停时间(s)_default: 0.05
SCAN_JUMP = 50 # 突变阈值(mm)_default: 50--上
SCAN_ADJUST = 0 # 扫描校准(m)_传感器到喷嘴的z轴距离，传感器高于喷嘴时为正值，低于喷嘴时为负值--上

PAINTDEG = 90 # 喷涂角度(圆柱)
PAINTWIDTH = 0.15 # 喷涂宽度(圆柱)

INIT_POS = [-0.8, -0.2, 1.2, -1.57, 0.0, 1.57] # 初始位置
SERV_POS = [-1.0, -0.2, -0.2, -1.57, 0.0, 1.57] # 维修位置
CLOG_POS = [-1.0, -0.2, 0.2, -1.57, 0.0, 1.57] # 堵枪位置
SAFE_POS = [-0.17, -0.18, 0.73, -1.57, 0.0, 1.57] # 安全位置

KEYTIMEOUT = 2 # 键盘输入超时时间(s)
SENSORTIMEOUT = 1 # 传感器超时时间(s)
