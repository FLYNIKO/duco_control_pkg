```bash
૮  ⚆⚆  ა
I      ●）
|    _ |
```
1 信息输入，通过/key_input || 
2 传感器数据输入，通过/STP23 || 
3 机械臂状态输出，通过/Duco_state || 
4 PID Kp0.001-0.01,Kd0.0001 || 
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
