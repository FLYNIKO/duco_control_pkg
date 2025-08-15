# ObstacleFlags.msg
# 自定义消息类型，用于发布各区域的避障标志

# 各区域避障标志
bool left      # 左区有障碍
bool right     # 右区有障碍  
bool center    # 中区有障碍
bool up        # 上区有障碍
bool down      # 下区有障碍

# 可选：添加时间戳和安全距离信息
time stamp              # 检测时间戳
float32 safe_distance   # 使用的安全距离