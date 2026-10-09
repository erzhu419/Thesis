import traci


class Lane:    # 创建一个车道类,用于描述每一个车道的属性和行为

    def __init__(self, lane_id_s, length_n, from_junction_s, to_junction_s):
        """初始化车道的函数"""
        """静态属性"""
        self.lane_id_s = lane_id_s                # 车道标签，字符串类型
        self.length_n = length_n                  # 车道长度，数值类型
        self.from_junction_s = from_junction_s    # 起始节点，字符串类型
        self.to_junction_s = to_junction_s        # 终止节点，字符串类型
        """动态属性"""
        self.veh_num_n = 0                        # 车道上的车辆数，数值类型（每秒更新）
        self.flow_n = 0                           # 车道上的流量，数值类型（每秒更新）
        self.speed_n = 0                          # 车道上的速度，数值类型（每秒更新）
        self.density_n = 0                        # 车道上的密度，数值类型（每秒更新）
        """容器属性"""
        self.state_data_l = []                    # 车道状态信息，列表类型（每秒更新） [记录时间,车辆数,流量,速度,密度]

    def update_lane_state(self, time_ex):   # 控制需要更新
        self.veh_num_n = traci.lane.getLastStepVehicleNumber(self.lane_id_s)
        self.speed_n = traci.lane.getLastStepMeanSpeed(self.lane_id_s)
        self.density_n = self.veh_num_n / self.length_n
        self.flow_n = self.speed_n * self.density_n
        self.state_data_l.append([time_ex, self.veh_num_n, self.flow_n, self.speed_n, self.density_n])
