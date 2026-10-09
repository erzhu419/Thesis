import traci


class Signal:    # 创建一个信号灯类,用于描述每一个信号灯的属性和行为

    def __init__(self, signal_id_s, in_lane_d, out_lane_d, detector_d, connection_d, phase_d):
        """初始化信号灯的函数"""
        """静态属性"""
        self.signal_id_s = signal_id_s              # 信号灯标签，字符串类型
        self.in_lane_d = in_lane_d                  # 进口车道，延申到500米或者是上一交叉口，字典类型 进口道:[进口道1,进口道2,...]
        self.out_lane_d = out_lane_d                # 出口车道，字典类型 出口道:[出口道]
        self.detector_d = detector_d                # 进口车道检测器，字典类型 进口道:检测器
        self.connection_d = connection_d            # 连接信息，字典类型 连接编号:[进口道,出口道,连接编号,相位编号]
        self.phase_d = phase_d                      # 相位信息，字典类型 相位编号:[[进口道,出口道,连接编号,相位编号],[进口道,出口道,连接编号,相位编号],...]
        self.cycle_length_n = 0                     # 原始信号周期长度，数值类型
        self.old_signal_program_o = ""              # 原始信号配时方案，对象类型
        self.pass_line_l = []                       # 经过的公交线路，列表类型 [线路标签,线路标签,...]
        """动态属性"""
        self.lane_last_flow_d = {}                  # 各车道的流量数据，字典类型（周期开始时更新） 进口道:上周期的流量
        self.lane_last_speed_d = {}                 # 各车道的速度数据，字典类型（周期开始时更新） 进口道:上周期的速度
        self.lane_last_density_d = {}               # 各车道的密度数据，字典类型（周期开始时更新） 进口道:上周期的密度
        self.current_phase_remain_time_l = []       # 当前信号灯所处相位，以及当前相位剩余时间，列表类型（公交车到站更新） [当前相位ID，当前相位剩余时间]
        self.cur_cycle_remain_time = 0              # 当前信号周期剩余时间，数值类型（公交车到站更新）
        self.cur_lanes_queue_length_d = {}          # 当前时刻信号灯控制的各车道的排队长度，字典类型（公交车到站更新）{车道ID:排队长度,...}
        """容器属性"""
        self.round_bus_state_l = []                 # 周围500米内公交车的状态，列表类型（周期开始时更新） [记录时间,车辆标签,线路标签,时刻表偏离,乘客数,信号灯连接,信号灯车道,信号灯相位,信号灯距离]
        self.service_data_l = []                    # 服务公交车的信息，列表类型（离灯更新） [车辆标签,线路标签,到灯时间,离灯时间]
        self.lane_queue_number_l = []               # 各车道每秒的排队车辆数，列表类型（每秒更新） [记录时间,[进口道,排队车辆数],[进口道,排队车辆数],...]

    def get_attribute_by_traci(self):
        self.cycle_length_n = float(traci.trafficlight.getParameter(self.signal_id_s, "cycleTime"))
        self.old_signal_program_o = traci.trafficlight.getAllProgramLogics(self.signal_id_s)[0]

    def get_pass_line(self, line_obj_dic_ex):
        for line_id in line_obj_dic_ex.keys():
            if self.signal_id_s in line_obj_dic_ex[line_id].signal_id_l:
                self.pass_line_l.append(line_id)

    def update_signal_state(self, time_ex, bus_obj_dic_ex):    # 每周期更新
        for land_id in self.detector_d.keys():
            if traci.inductionloop.getTimeSinceDetection(self.detector_d[land_id]) == 0:
                self.lane_last_flow_d[land_id] = traci.inductionloop.getLastIntervalVehicleNumber(self.detector_d[land_id]) / 60
                self.lane_last_speed_d[land_id] = traci.inductionloop.getLastIntervalMeanSpeed(self.detector_d[land_id])
                if self.lane_last_speed_d[land_id] == 0:
                    self.lane_last_density_d[land_id] = -1
                else:
                    self.lane_last_density_d[land_id] = self.lane_last_flow_d[land_id] / self.lane_last_speed_d[land_id]
        for bus_id in bus_obj_dic_ex.keys():
            if bus_obj_dic_ex[bus_id].next_signal_id_s == self.signal_id_s and bus_obj_dic_ex[bus_id].next_signal_length_n <= 500:
                self.round_bus_state_l.append([time_ex, bus_obj_dic_ex[bus_id].bus_id_s, bus_obj_dic_ex[bus_id].belong_line_id_s, bus_obj_dic_ex[bus_id].timetable_deviation_n, bus_obj_dic_ex[bus_id].passenger_num_n,
                                               bus_obj_dic_ex[bus_id].next_signal_link_s, bus_obj_dic_ex[bus_id].next_signal_lane_s, bus_obj_dic_ex[bus_id].next_signal_phase_s, bus_obj_dic_ex[bus_id].next_signal_length_n])

    def update_service_data(self, bus_obj_ex):    # 公交离灯更新
        if self.signal_id_s in bus_obj_ex.stop_signal_id_list_l:
            self.service_data_l.append([bus_obj_ex.bus_id_s, bus_obj_ex.belong_line_id_s, bus_obj_ex.arriver_signal_time_d[self.signal_id_s], bus_obj_ex.depart_signal_time_d[self.signal_id_s]])
        else:
            self.service_data_l.append([bus_obj_ex.bus_id_s, bus_obj_ex.belong_line_id_s, 0, 0])

    def update_queue_number(self, time_ex):    # 每秒更新
        lane_queue_number = [time_ex]
        for land_id in self.in_lane_d.keys():
            lane_queue_number.append([land_id, traci.lane.getLastStepHaltingNumber(land_id)])
        self.lane_queue_number_l.append(lane_queue_number)

    def update_current_phase_and_remain_time(self, cur_time, connection_linkindex):  # 触发更新
        """更新信号灯当前所处相位ID及当前相位剩余时间、当前信号周期剩余时间"""
        cur_phase_remain_time = traci.trafficlight.getNextSwitch(self.signal_id_s)-cur_time
        cur_phase_index = traci.trafficlight.getPhase(self.signal_id_s)

        # 获取针对connection_linkindex的红绿灯时间，调整为周期开始是红灯，以绿灯相位结束周期
        phase_states = [self.old_signal_program_o.phases[state_num].state for state_num in list(range(0, len(self.old_signal_program_o.phases)))]
        linkindex_phase_states = [s[connection_linkindex] for s in phase_states if len(s) > connection_linkindex]

        # 获取当前linkindex绿灯相位的起止索引
        first_index = next((i for i, c in enumerate(linkindex_phase_states) if c in ('g', 'G', 'y', 'Y')), None)
        last_index = next((len(linkindex_phase_states) - 1 - i for i, c in enumerate(reversed(linkindex_phase_states)) if c in ('g', 'G', 'y', 'Y')), None)
        green_state_phase_index_l = list(range(first_index, last_index + 1))

        # 获取当前linkindex红灯相位的起止索引
        A = list(range(0, len(linkindex_phase_states)))
        A_filtered = [x for x in A if x not in green_state_phase_index_l]
        start_index = A.index(green_state_phase_index_l[-1]) + 1 if green_state_phase_index_l[-1] in A else 0
        red_state_phase_index_l = A_filtered[start_index - len(A):] + A_filtered[:start_index - len(A)]

        resorted_phase_index_l = red_state_phase_index_l + green_state_phase_index_l

        # 计算当前linkindex绿灯或者红灯相位的已经经过的时间
        phase_over_time = 0
        if cur_phase_index in red_state_phase_index_l:  # 当前处于红灯相位
            for phase_num in red_state_phase_index_l[:red_state_phase_index_l.index(cur_phase_index)+1]:
                phase_over_time += self.old_signal_program_o.phases[phase_num].duration
        else:  # 当前处于绿灯相位
            for phase_num in green_state_phase_index_l[:green_state_phase_index_l.index(cur_phase_index)+1]:
                phase_over_time += self.old_signal_program_o.phases[phase_num].duration
        phase_over_time -= cur_phase_remain_time

        # 计算当前linkindex周期剩余时间
        cycle_over_time = 0
        for phase_num in resorted_phase_index_l[:resorted_phase_index_l.index(cur_phase_index)+1]:
            cycle_over_time += self.old_signal_program_o.phases[phase_num].duration
        cycle_over_time -= cur_phase_remain_time

        return cur_phase_index, phase_over_time, cycle_over_time

    def updata_current_queue_length(self):  # 触发更新
        """更新当前时刻信号灯控制各车道的排队长度"""
        for lane_id in self.in_lane_d:
            self.cur_lanes_queue_length_d[lane_id] = traci.lane.getLastStepLength(lane_id) * traci.lane.getLastStepHaltingNumber(lane_id)

        debug = 0
