import traci


class Line:    # 创建一个线路类,用于描述每一个线路的属性和行为

    def __init__(self, line_id_s, edge_id_l, stop_id_l, signal_id_l, edge_length_d, edge_average_speed_d, distance_between_stop_d, distance_between_signal_d, edge_between_stop_d, edge_between_signal_d):
        """初始化车道的函数"""
        """静态属性"""
        self.line_id_s = line_id_s            # 线路标签，字符串类型
        self.edge_id_l = edge_id_l            # 公交线路经过的路段标签，列表类型 [路段标签,路段标签,...]
        self.stop_id_l = stop_id_l            # 公交线路经过的车站标签，列表类型 [公交站标签,公交站标签,...]
        self.signal_id_l = signal_id_l        # 公交线路经过的信号灯标签，列表类型 [信号灯标签,信号灯标签,...]
        self.edge_length_d = edge_length_d                           # 公交线路经过的每条路段的长度，字典类型 路段标签:长度
        self.edge_average_speed_d = edge_average_speed_d             # 公交线路经过的每条路段在每个小时的平均速度，字典类型 路段标签:[第1小时速度,第2小时速度,...]
        self.distance_between_stop_d = distance_between_stop_d       # 公交从上一站（第一个为起始点）到达本站的距离，字典类型 公交站标签:长度
        self.distance_between_signal_d = distance_between_signal_d   # 公交从上一信号灯（第一个为起始点）到达本信号灯的距离，字典类型 信号灯标签:长度
        self.edge_between_stop_d = edge_between_stop_d               # 公交从上一站（第一个为起始点）到达本站经过的路段，字典类型 公交站标签:[路段标签,路段标签,...]
        self.edge_between_signal_d = edge_between_signal_d           # 公交从上一信号灯（第一个为起始点）到达本信号灯经过的路段，字典类型 信号灯标签:[路段标签,路段标签,...]

        """动态属性"""
        self.veh_num_d = {}                   # 每个路段上的车辆数，字典类型（触发更新） 路段标签:车辆数
        self.flow_d = {}                      # 每个路段上的流量，字典类型（触发更新） 路段标签:流量
        self.speed_d = {}                     # 每个路段上的速度，字典类型（触发更新） 路段标签:速度
        self.density_d = {}                   # 每个路段上的密度，字典类型（触发更新） 路段标签:密度
        """容器属性"""
        self.state_data_l = []                # 每个路段上信息，列表类型（触发更新） [记录时间,[路段标签,车辆数,流量,速度,密度],[路段标签,车辆数,流量,速度,密度],...]

    def update_line_state(self, time_ex):   # 控制需要更新
        self.veh_num_d = {}
        self.flow_d = {}
        self.speed_d = {}
        self.density_d = {}
        state_data = [time_ex]
        for edge_id in self.edge_id_l:
            veh_num_n = traci.edge.getLastStepVehicleNumber(edge_id)
            speed_n = traci.edge.getLastStepMeanSpeed(edge_id)
            density_n = veh_num_n / traci.lane.getLength(edge_id + "_0")
            flow_n = speed_n * density_n
            self.veh_num_d[edge_id] = veh_num_n
            self.flow_d[edge_id] = flow_n
            self.speed_d[edge_id] = speed_n
            self.density_d[edge_id] = density_n
            state_data.append([edge_id, veh_num_n, flow_n, speed_n, density_n])
        self.state_data_l.append(state_data)

    def get_next_stop_id_by_this_stop_id(self, stop_id_ex):
        next_stop_id = "no_find"
        for n in range(0, len(self.stop_id_l)):
            if self.stop_id_l[n] == stop_id_ex:
                if n == len(self.stop_id_l)-1:
                    next_stop_id = ""
                else:
                    next_stop_id = self.stop_id_l[n+1]
        if next_stop_id == "no_find":
            print("警告，line的get_next_stop_id_by_this_stop_id没有找到下一个公交站！")
        return next_stop_id

    def get_distance_between_edge_to_edge(self, start_edge_ex, end_edge_ex):
        """根据线路上两个路段标签获得两个路段之间的距离，开始的路段不记录距离，结束的路段记录距离"""
        total_length = 0
        if_add_length = "no"
        for edge_id in self.edge_id_l:
            if if_add_length == "yes":
                total_length = total_length + self.edge_length_d[edge_id]
            if edge_id == start_edge_ex:
                if_add_length = "yes"
            if edge_id == end_edge_ex:
                break
        if total_length == 0:
            print("警告，initialize_line的get_distance_between_stop没获得距离！")
        return total_length
