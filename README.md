```bash
૮  ⚆⚆  ა
I      ●）
|    _ |
```
```
   ↓       上表面喷涂
===== ↙   下翼面喷涂
  |     ←  腹板喷涂
===== ↖   上翼面喷涂
   ↑       下表面喷涂

in config.py:
机械臂运动坐标系为  前后X（后），左右Y（右），上下Z（上）,而雷达放置的坐标系是 前后z（后） 上下y（上） 左右x（右）
（定义方法是机械臂的rx、ry、rz 0点的XYZ坐标系 ，雷达放置为雷达正方向对向X、Z正方向）
in laser_scan_averager.py：
将每个雷达的数据每5帧过滤平均后输出1帧，原10HZ过滤后2HZ
in cv2_H_detecter.py:
使用openCV将左雷达扫描到的数据转换为二维图像，并根据预设：两条平行线一条垂直线，找到符合U型条件的三根线，根据长度
和角度，判断是web（腹板）或是flange（翼板）并输出相应的线数据，之后在ManualControl.py中使用这些数据来找5个喷涂
位姿以及在喷涂234位置时的喷涂位置保持。
in cv2_line_detecher_twin.py：
使用openCV将左右两个雷达扫描到的数据转换为单独的二维图像，并且将两个雷达扫描到的线信息转换坐标系到两个雷达的中间点，
如果每个雷达都能扫描到一条符合角度、长度条件的线，则会连接两条线的中点，在ManualControl.py中会使用这根线的位置和
角度来调整机械臂喷涂上下表面时的位姿。
in radar_three_cloud.py：
将三个雷达的LaserScan类型的/scan，读取机械臂末端位姿后统一坐标系到机械臂底座，3in1，转换成PointCloud2类型的
/obstacle_avoidance/pointcloud话题
in radar_obstacle_avoidance.py：
读取/obstacle_avoidance/pointcloud，并根据config.py中机械臂运动的坐标系划分区域识别避障，最后输出每个区域的
bool值到/obstacle_flags作为该区域的避障开关。

位置变化，cv2_H_detecter.py:识别出来的东西其实是延圆心180度顺时针旋转的结果

/Duco_state                        话题： 机械臂状态
/left_radar/scan                   话题： 左雷达数据
/left_radar/filtered_scan          话题： 左雷达数据(滤波后)
/right_radar/scan                  话题： 右雷达数据
/right_radar/filtered_scan         话题： 右雷达数据(滤波后)
/main_radar/scan                   话题： 主雷达数据
/main_radar/filtered_scan          话题： 主雷达数据(滤波后)
/obstacle_avoidance/pointcloud     话题： 避障点云
/obstacle_flags                    话题： 避障标志(自定义消息)
/twin_radar/line_detection_info    话题： 双线检测信息(自定义消息)
/left_radar/H_detection_info       话题： 左雷达H检测信息(自定义消息)
```
```
机器人状态信息列表 get_robot_state()
data[0]表示机器人状态
SR_Start =      0, //机器人启动
SR_Initialize = 1, //机器人初始化
SR_Logout =     2, //机器人登出, 暂未使用
SR_Login =      3, //机器人登陆,暂未使用
SR_PowerOff =   4, //机器人下电
SR_Disable =    5, //机器人下使能
SR_Enable =     6, //机器人上使能
SR_Update=      7, //机器人更新

data[1]表示程序状态
SP_Stopped =    0, //程序停止
SP_Stopping =   1, //程序正在停止中
SP_Running =    2, //程序正在运行
SP_Paused =     3, //程序已经暂停
SP_Pausing =    4, //程序暂停中
SP_TaskRuning = 5, //手动示教任务执行中

data[2]表示安全控制器状态
SS_INIT =       0, //初始化
SS_WAIT =       2, //等待
SS_CONFIG =     3, //配置模式
SS_POWER_OFF =  4, //下电状态
SS_RUN =        5, //正常运行状态
SS_RECOVERY=    6, //恢复模式
SS_STOP2 =      7, //Stop2
SS_STOP1 =      8, //Stop1
SS_STOP0 =      9, //Stop0
SS_MODEL=       10, //模型配置状态
SS_REDUCE =     12, //缩减模式状态
SS_BOOT =       13, //引导
SS_FAIL =       14, //致命错误状态
SS_UPDATE =     99, //更新状态

data[3]表示操作模式
kManual =       0, //手动模式
kAuto =         1, //自动模式
kRemote =       2, //远程模式

任务状态信息 get_noneblock_taskstate (int:id)
ST_Idle =               0, //任务未执行
ST_Running =            1, //任务正在执行
ST_Paused =             2, //任务已经暂停
ST_Stopped =            3, //任务已经停止
ST_Finished =           4, //任务已经正常执行完成,唯一表示任务正常完成（任务已经结束）
ST_Interrupt =          5, //任务被中断（任务已经结束）
ST_Error =              6, //任务出错（任务已经结束）
ST_Illegal =            7, //任务非法, 当前状态下任务不能执行（任务已经结束）
ST_ParameterMismatch =  8, //任务参数错误（任务已经结束）
```
