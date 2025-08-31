# ObstacleFlags.msg
# 自定义消息类型，用于发布各区域的避障标志

# 各区域避障标志
bool left_front    # 左前区有障碍
bool left_mid      # 左中区有障碍
bool left_rear     # 左后区有障碍
bool right_front   # 右前区有障碍  
bool right_mid     # 右中区有障碍
bool right_rear    # 右后区有障碍
bool center        # 中区有障碍
bool up            # 上区有障碍
bool down          # 下区有障碍
# 可选：添加时间戳和安全距离信息
time stamp              # 检测时间戳
float32 safe_distance   # 使用的安全距离