import traci


class Passenger:    # 创建一个乘客类,用于描述每一个乘客的属性和行为

    def __init__(self, passenger_id_s, start_time_n, start_stop_id_s, transfer_stop_id_s, end_stop_id_s, start_edge_id_s, transfer_edge_id_s, end_edge_id_s):
        """初始化乘客的函数"""
        """静态属性"""
        self.passenger_id_s = passenger_id_s             # 乘客标签，字符串类型
        self.start_time_n = start_time_n                 # 乘客产生的时间，数值类型
        self.start_stop_id_s = start_stop_id_s           # 出发地公交站标签，字符串类型
        self.transfer_stop_id_s = transfer_stop_id_s     # 换乘地公交站标签，字符串类型，无需换乘的乘客为“”
        self.end_stop_id_s = end_stop_id_s               # 目的地公交站标签，字符串类型
        self.start_edge_id_s = start_edge_id_s           # 出发地路段标签，字符串类型
        self.transfer_edge_id_s = transfer_edge_id_s     # 换乘地路段标签，字符串类型，无需换乘的乘客为“”
        self.end_edge_id_s = end_edge_id_s               # 目的地路段标签，字符串类型
        """动态属性"""
        self.passenger_state_s = "No"                    # 乘客的状态，字符串类型，No还没有产生；Lane在车道上；Stop在公交站；Bus在车上（触发更新）
        self.arriver_time_n = 0                          # 到达出发地公交站时间，数值类型（到站更新）
        self.passable_line_l = []                        # 下一行程可搭乘的线路，列表类型（到站更新） [线路标签,线路标签,...]
        self.take_bus_id_s = ""                          # 搭乘的公交车标签，字符串类型（上车更新）
        self.boarding_time_n = 0                         # 上车时间，数值类型（上车更新）
        self.alighting_time_n = 0                        # 下车时间，数值类型（下车更新）
        self.last_appearance_n = 0                       # 最后出现的时间（每秒更新）

        """容器属性"""
        self.travel_data_l = []                          # 乘客旅行信息，列表类型（下车更新） [到达时间，上车时间，下车时间，等待时间，旅行时间，公交车标签]

    def passenger_activate(self, time_ex, line_obj_dic_ex):
        self.passenger_state_s = "Lane"
        self.last_appearance_n = time_ex
        passable_line_l = []
        if self.transfer_stop_id_s == "":
            next_stop = self.end_stop_id_s
        else:
            next_stop = self.transfer_stop_id_s
        for line_id in line_obj_dic_ex.keys():
            if self.start_stop_id_s in line_obj_dic_ex[line_id].stop_id_l and next_stop in line_obj_dic_ex[line_id].stop_id_l:
                passable_line_l.append(line_id)
        self.passable_line_l = passable_line_l

    def passenger_run(self, time_ex, line_obj_dic_ex):
        # 如果上一步仿真，乘客在车道上
        if self.passenger_state_s == "Lane":
            # 判断乘客是否到达公交站
            if not traci.person.getLaneID(self.passenger_id_s):
                self.passenger_state_s = "Stop"
                self.arriver_time_n = time_ex
                passable_line_l = []
                if self.transfer_stop_id_s == "":
                    next_stop = self.end_stop_id_s
                else:
                    next_stop = self.transfer_stop_id_s
                for line_id in line_obj_dic_ex.keys():
                    if self.start_stop_id_s in line_obj_dic_ex[line_id].stop_id_l and next_stop in line_obj_dic_ex[line_id].stop_id_l:
                        passable_line_l.append(line_id)
                self.passable_line_l = passable_line_l
        # 如果上一步仿真，乘客在公交站
        if self.passenger_state_s == "Stop":
            # 判断乘客是否上车
            if traci.person.getVehicle(self.passenger_id_s):
                self.passenger_state_s = "Bus"
                self.take_bus_id_s = traci.person.getVehicle(self.passenger_id_s)
                self.boarding_time_n = time_ex
        # 如果上一步仿真，公交车在公交车
        if self.passenger_state_s == "Bus":
            # 判断公交车是否下车换乘
            if not traci.person.getLaneID(self.passenger_id_s):
                self.passenger_state_s = "Stop"
                self.alighting_time_n = time_ex
                self.travel_data_l.append([self.arriver_time_n, self.boarding_time_n, self.alighting_time_n,
                                           self.boarding_time_n - self.arriver_time_n,
                                           self.alighting_time_n - self.arriver_time_n, self.take_bus_id_s])
                self.arriver_time_n = time_ex
                passable_line_l = []
                for line_id in line_obj_dic_ex.keys():
                    if self.transfer_stop_id_s in line_obj_dic_ex[line_id].stop_id_l and self.end_stop_id_s in line_obj_dic_ex[line_id].stop_id_l:
                        passable_line_l.append(line_id)
                self.passable_line_l = passable_line_l
        # 更新每秒都需更新的状态
        self.last_appearance_n = time_ex

    def passenger_end(self):
        self.alighting_time_n = self.last_appearance_n + 1
        self.travel_data_l.append([self.arriver_time_n, self.boarding_time_n, self.alighting_time_n,
                                   self.boarding_time_n - self.arriver_time_n,
                                   self.alighting_time_n - self.arriver_time_n, self.take_bus_id_s])
