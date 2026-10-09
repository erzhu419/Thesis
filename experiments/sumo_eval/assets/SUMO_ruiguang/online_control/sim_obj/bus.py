import traci


class Bus:    # 创建一个公交车类,用于描述每一个公交车的属性和行为

    def __init__(self, bus_id_s, start_time_n, belong_line_id_s):
        """初始化信号灯的函数"""
        """静态属性"""
        self.bus_id_s = bus_id_s                      # 公交车标签，字符串类型
        self.start_time_n = start_time_n              # 公交车发车时间，数值类型
        self.belong_line_id_s = belong_line_id_s      # 公交车所属线路标签，字符串类型
        self.arriver_timetable_d = {}                 # 公交车的到站时刻表，字典类型
        """动态属性"""
        self.bus_state_s = "No"                       # 公交车状态，字符串类型，No还没有产生；Edge在路段上；Stop在公交站；Signal在信号交叉口（触发更新）
        self.bus_speed_n = 0                          # 公交车速度，数值类型（每秒更新）
        self.distance_n = 0                           # 公交车行程，数值类型（每秒更新）
        self.bus_cur_lane_s = ""                      # 公交车所处车道ID，字符串类型（每秒更新）
        self.timetable_deviation_n = 0                # 公交时刻表偏差，数值类型（到站更新）
        self.passenger_num_n = 0                      # 公交车乘客数，数值类型（离站更新）
        self.next_stop_id_s = ""                      # 下一个公交站标签，字符串类型，无下一公交站为“”（离站更新）(需要修改为到站更新，因为加入模型触发时刻，公交车在站内，会默认其已经完成乘客服务，也因此原因，需要预测其离站时间)
        self.next_stop_length_n = 0                   # 距下一个公交站停止线的长度，数值类型（每秒更新）
        self.next_signal_id_s = ""                    # 下一个信号灯标签，字符串类型，无下一信号灯为“”（离信号灯更新）
        self.next_signal_link_s = ""                  # 下一个信号灯所处连接，字符串类型，无下一信号灯为“”（离信号灯更新）
        self.next_signal_lane_s = ""                  # 下一个信号灯所处车道，字符串类型，无下一信号灯为“”（离信号灯更新）
        self.next_signal_phase_s = ""                 # 下一个信号灯所处相位，字符串类型，无下一信号灯为“”（离信号灯更新）
        self.next_signal_length_n = 0                 # 距下一个信号灯停止线的长度，数值类型（每秒更新）

        self.just_server_stop_data_d = {}             # 公交车刚刚服务结束的公交站ID及到发时间，字典类型 (到站更新，离站更新)（到站更新的为预测的公交离站时间，有离站时间之后再进行更新）
        self.unserved_stop_l = []                     # 公交车还未服务的公交站，列表类型（到站更新）（若触发时刻公交车处于站内，则公交车该属性不包括当前公交站）
        self.bus_passenger_d = {}                     # 公交车车内各od乘客数量，字典类型（到站更新，离站更新）（到站更新的为预测的公交车离站时的乘客信息，有实际离站信息之后再进行更新）
        self.arriver_signal_queue_info_l = ["", 0, 0]   # 公交车到达交叉口处排队处，距离信号灯停止线的距离，列表类型（到信号灯并进行排队才更新）[公交车到达队尾时所处路段，公交车到达队尾的时间，到达队尾时与信号灯的距离]
        self.subsequent_signal_info_d = {}            # 公交车后面要经过的各信号灯的信息，字典类型（触发更新）{信号灯ID:[当前时间，当前相位ID，绿灯时长，信号周期时长，绿灯或红灯已经经过的时长，信号周期已经经过的时长，当前车道排队长度，平均延误时间]...}
        self.trajectory_dict = {}                     # 轨迹记录 {stop_id: [arrive_time, ...]}，用于计算Headway
        
        # Optimization: Cache for getNextLinks
        self.cached_next_links = None
        self.cached_lane = None

        """容器属性"""
        self.bus_speed_l = []                         # 公交车的速度，列表类型（每秒更新） [速度值,速度值,...]
        self.distance_l = []                          # 公交车的行程，列表类型（每秒更新） [行驶距离,行驶距离,...]
        self.arriver_stop_time_d = {}                 # 公交车的到站时间，字典类型（到站更新） 公交站:时间
        self.depart_stop_time_d = {}                  # 公交车的离站时间，字典类型（离站更新） 公交站:时间
        self.alight_num_d = {}                        # 公交车的下车人数，字典类型（离站更新） 公交站:人数
        self.want_board_num_d = {}                    # 公交车的想要上车人数，字典类型（离站更新） 公交站:人数
        self.board_num_d = {}                         # 公交车的上车人数，字典类型（离站更新） 公交站:人数
        self.strand_num_d = {}                        # 公交车的滞留人数，字典类型（离站更新） 公交站:人数
        self.arriver_signal_time_d = {}               # 公交车信号灯排队开始时间，字典类型（到信号灯并进行排队才更新） 信号灯:时间
        self.depart_signal_time_d = {}                # 公交车信号灯排队结束时间，字典类型（离信号灯并存在排队才更新） 信号灯:时间
        self.no_stop_signal_id_list_l = []            # 没在信号灯排队的信号灯标签，列表类型（离信号灯排队更新） [信号灯标签,信号灯标签,...]
        self.stop_signal_id_list_l = []               # 有在信号灯排队的信号灯标签，列表类型（离信号灯排队更新） [信号灯标签,信号灯标签,...]
        self.bus_served_lane_l = []                   # 公交车经过的车道id，列表类型（每秒更新） [车道id,车道id,...]


    def _get_next_links(self):
        """
        Optimized wrapper for traci.vehicle.getNextLinks.
        Caches the result based on the current lane ID.
        """
        current_lane = traci.vehicle.getLaneID(self.bus_id_s)
        if current_lane != self.cached_lane or self.cached_next_links is None:
            self.cached_lane = current_lane
            self.cached_next_links = traci.vehicle.getNextLinks(self.bus_id_s)
        return self.cached_next_links

    def get_arriver_timetable(self, line_obj_ex):
        stop_id_list = line_obj_ex.stop_id_l
        edge_length_dic = line_obj_ex.edge_length_d
        edge_average_speed_dic = line_obj_ex.edge_average_speed_d
        edge_between_stop_dic = line_obj_ex.edge_between_stop_d
        total_time = self.start_time_n
        for stop_id in stop_id_list:
            for edge_id in edge_between_stop_dic[stop_id]:
                total_time = total_time + edge_length_dic[edge_id] / (edge_average_speed_dic[edge_id][int(self.start_time_n // 3600)] * 0.8)
            self.arriver_timetable_d[stop_id] = total_time

    def bus_activate(self, line_obj_ex, stop_obj_dic_ex, signal_obj_dic_ex, time_ex):
        self.bus_state_s = "Edge"
        self.bus_speed_n = 0
        self.distance_n = 0
        self.timetable_deviation_n = 0
        self.passenger_num_n = 0
        self.next_stop_id_s = line_obj_ex.stop_id_l[0]
        next_stop_length_n = 0
        self.just_server_stop_data_d = {}
        self.unserved_stop_l = []
        self.bus_passenger_d = {}

        # 计算到下一个站点的距离
        # 判断公交车是否已在下一个站点所在的车道上
        if traci.vehicle.getLanePosition(self.bus_id_s) == stop_obj_dic_ex[self.next_stop_id_s].at_lane_s:
            # 如果在同一车道：距离 = 目标车道总长度 - 公交车当前位置
            next_stop_length_n += (traci.lane.getLength(stop_obj_dic_ex[self.next_stop_id_s].at_lane_s) -
                                   traci.vehicle.getLanePosition(self.bus_id_s))
        else:
            # 如果不在同一车道：需要计算当前车道剩余距离 + 中间车道距离
            # 先计算当前车道的剩余距离
            next_stop_length_n += (traci.lane.getLength(traci.vehicle.getLaneID(self.bus_id_s)) -
                                   traci.vehicle.getLanePosition(self.bus_id_s))
            # 遍历后续车道，累加距离直到到达目标站点车道
            for traci_lane_obj in self._get_next_links():
                next_stop_length_n += traci.lane.getLength(traci_lane_obj[0]) + traci_lane_obj[-1]
                # 检查是否到达目标站点所在车道（通过车道ID前缀匹配）
                if traci_lane_obj[0][:-2] == stop_obj_dic_ex[self.next_stop_id_s].at_lane_s[:-2]:
                    break
        self.next_stop_length_n = next_stop_length_n
        # Signal processing disabled for Holding Control optimization
        next_signal_list = traci.vehicle.getNextTLS(self.bus_id_s)
        self.next_signal_id_s = next_signal_list[0][0]
        self.next_signal_link_s = str(next_signal_list[0][1])
        self.next_signal_lane_s = signal_obj_dic_ex[self.next_signal_id_s].connection_d[self.next_signal_link_s][0]
        self.next_signal_phase_s = signal_obj_dic_ex[self.next_signal_id_s].connection_d[self.next_signal_link_s][3]
        # 计算到下一个交通信号灯的距离
        next_signal_length_n = 0
        # 判断公交车是否已在下一个信号灯所在的车道上
        if traci.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
            # 如果在同一车道：距离 = 目标车道总长度 - 公交车当前位置
            next_signal_length_n += (traci.lane.getLength(self.next_signal_lane_s) -
                                     traci.vehicle.getLanePosition(self.bus_id_s))
        else:
            # 如果不在同一车道：需要计算当前车道剩余距离 + 中间车道距离
            # 先计算当前车道的剩余距离
            next_signal_length_n += (traci.lane.getLength(traci.vehicle.getLaneID(self.bus_id_s)) -
                                     traci.vehicle.getLanePosition(self.bus_id_s))
            # 遍历后续车道，累加距离直到到达目标信号灯车道
            for traci_lane_obj in self._get_next_links():
                next_signal_length_n += traci.lane.getLength(traci_lane_obj[0]) + traci_lane_obj[-1]
                # 检查是否到达目标信号灯所在车道（通过车道ID前缀匹配）
                if traci_lane_obj[0][:-2] == self.next_signal_lane_s[:-2]:
                    break
        self.next_signal_length_n = next_signal_length_n
        # self.next_signal_length_n = -1
        # self.next_signal_id_s = "" # Explicitly clear to prevent crash in bus_running
        self.bus_speed_l.append(self.bus_speed_n)
        self.distance_l.append(self.distance_n)
        self.alight_num_d[self.next_stop_id_s] = 0
    def bus_running(self, line_obj_ex, stop_obj_dic_ex, signal_obj_dic_ex, passenger_obj_dic_ex, time_ex, BusCap, AveAlightingTime,
                    AveBoardingTime, bus_arrstation_od_otd_dict, bus_obj_dic_ex, involved_tl_ID_l, sorted_busline_edge_d):
        # 如果上一步仿真，公交车在路段上
        if self.bus_state_s == "Edge":
            # 判断公交车是否到达公交站
            if traci.vehicle.isAtBusStop(self.bus_id_s):
                self.bus_state_s = "Stop"
                # self.timetable_deviation_n = time_ex - self.arriver_timetable_d[self.next_stop_id_s]# TODO 这里没处理好当后车还没出现的时候的情况，让codex仿照前面的写法进行补充
                if self.next_stop_id_s and self.next_stop_id_s in self.arriver_timetable_d:
                    self.timetable_deviation_n = time_ex - self.arriver_timetable_d[self.next_stop_id_s]
                else:
                    self.timetable_deviation_n = 0.0  # 或者维持上一次的值/做其他处理
                    # 提前 return 也可以，看后续逻辑是否依赖该站
                    return
                self.arriver_stop_time_d[self.next_stop_id_s] = time_ex
                
                # Record trajectory for robust headway calculation
                if self.next_stop_id_s not in self.trajectory_dict:
                    self.trajectory_dict[self.next_stop_id_s] = []
                self.trajectory_dict[self.next_stop_id_s].append(time_ex)

                next_stop_index = line_obj_ex.stop_id_l.index(self.next_stop_id_s)
                self.unserved_stop_l = line_obj_ex.stop_id_l[next_stop_index+1:]
                # 预测公交车离站时间，计算just_server_stop_data_d参数
                stop_passenger_list = traci.busstop.getPersonIDs(self.next_stop_id_s)
                want_boarding_passenger_num = 0
                want_boarding_passenger_od = {}
                for passenger_id in stop_passenger_list:  # 计算当前时刻站内可上车的乘客数量
                    if self.belong_line_id_s in passenger_obj_dic_ex[passenger_id].passable_line_l:
                        want_boarding_passenger_num += 1
                        if not passenger_obj_dic_ex[passenger_id].transfer_edge_id_s:  # 表明该乘客是OD类型，无换乘
                            if f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}" not in want_boarding_passenger_od:
                                want_boarding_passenger_od[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] = 1
                            else:
                                want_boarding_passenger_od[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] += 1
                        else: # 现在没有效果，可能因为没有OTD乘客
                            if f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}" not in want_boarding_passenger_od:
                                want_boarding_passenger_od[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] = 1
                            else:
                                want_boarding_passenger_od[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] += 1

                alighting_passenger_num = 0  # 计算在该站下车乘客数量
                for person_id in traci.vehicle.getPersonIDList(self.bus_id_s):
                    if passenger_obj_dic_ex[person_id].end_stop_id_s == self.next_stop_id_s or (passenger_obj_dic_ex[person_id].transfer_stop_id_s and passenger_obj_dic_ex[person_id].transfer_stop_id_s == self.next_stop_id_s):
                        alighting_passenger_num += 1

                actual_boarding_passenger_num = min(want_boarding_passenger_num, BusCap - traci.vehicle.getPersonNumber(self.bus_id_s) + alighting_passenger_num)  # 计算实际可上车的乘客数量

                predicted_depart_time = time_ex + max(AveAlightingTime * alighting_passenger_num, AveBoardingTime * actual_boarding_passenger_num) + 4.0
                self.just_server_stop_data_d = {}
                self.just_server_stop_data_d[self.next_stop_id_s] = [time_ex, predicted_depart_time]

                # 预测离站时公交车内的不同od乘客数量
                strand_passenger_od = {}
                for od_otd in want_boarding_passenger_od:
                    if od_otd not in self.bus_passenger_d:
                        self.bus_passenger_d[od_otd] = want_boarding_passenger_od[od_otd] * (actual_boarding_passenger_num/want_boarding_passenger_num)
                    else:  # (这个应该不会被触发，因为该站上车的乘客的OD或者OTD类型，必然车上不会已经有)
                        self.bus_passenger_d[od_otd] += want_boarding_passenger_od[od_otd] * (actual_boarding_passenger_num / want_boarding_passenger_num)

                    strand_passenger_od[od_otd] = want_boarding_passenger_od[od_otd] - self.bus_passenger_d[od_otd]

                if self.next_stop_id_s in bus_arrstation_od_otd_dict[self.belong_line_id_s] and self.next_stop_id_s != list(bus_arrstation_od_otd_dict[self.belong_line_id_s].keys())[-1]:  # 表示不是第一个车站，且不是最后一个车站
                    for od_otd in list(set(bus_arrstation_od_otd_dict[self.belong_line_id_s][self.next_stop_id_s])-set(bus_arrstation_od_otd_dict[self.belong_line_id_s][line_obj_ex.stop_id_l[next_stop_index+1]])):
                        if od_otd in self.bus_passenger_d:
                            del self.bus_passenger_d[od_otd]  # 删除在该站到站的od乘客

                stop_obj_dic_ex[self.next_stop_id_s].update_stop_unserved_bus(self.bus_id_s, line_obj_ex, stop_obj_dic_ex)  # 更新车站信息

                if self.next_stop_id_s != list(bus_arrstation_od_otd_dict[self.belong_line_id_s].keys())[-1]:  # 表明该站不是该线路终点站
                    stop_obj_dic_ex[self.next_stop_id_s].update_service_data(bus_obj_dic_ex, self.bus_id_s, passenger_obj_dic_ex, strand_passenger_od, predicted_depart_time, "arrive")  # 更新公交站的属性

                self.next_stop_id_s = line_obj_ex.get_next_stop_id_by_this_stop_id(self.next_stop_id_s)  # 进行next_stop属性更新
                debug = 0
            else:
                # 判断公交车是否到达信号灯（信号灯进口车道）
                if traci.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
                    self.bus_state_s = "Signal"

            # LRG 判断公交车是否处于普通路段上，且该路段是否有公交专用道
            # LRG 判断公交车是否处于普通路段上，且该路段是否有公交专用道
            # (Disabled for RL holding task to avoid emergency stops)
            # if self.bus_cur_lane_s[:-2] in sorted_busline_edge_d[self.belong_line_id_s] and sorted_busline_edge_d[self.belong_line_id_s][self.bus_cur_lane_s[:-2]][0] == "w" and sorted_busline_edge_d[self.belong_line_id_s][self.bus_cur_lane_s[:-2]][2] == "1":
            #     impacted_edge = sorted_busline_edge_d[self.belong_line_id_s][self.bus_cur_lane_s[:-2]][3]
            #     if sorted_busline_edge_d[self.belong_line_id_s][impacted_edge][0] == "i":  # 该路段不会与后面的以交叉口结束的路段进行合并
            #         debug = 0
            #     # 满足走公交专用道的条件，但是没有走公交专用道，则强制换道
            #     if self.bus_cur_lane_s != self.bus_cur_lane_s[:-2]+"_0":
            #         traci.vehicle.changeLane(self.bus_id_s, 0, 2)
            #         deubg = 0

        # 如果上一步仿真，公交车在公交站
        if self.bus_state_s == "Stop":
            # 判断公交车是否离开公交站
            if not traci.vehicle.isAtBusStop(self.bus_id_s):
                # 判断公交车是否到达信号灯（信号灯进口车道）
                if traci.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
                    self.bus_state_s = "Signal"
                else:
                    self.bus_state_s = "Edge"
                board_num = traci.vehicle.getPersonNumber(self.bus_id_s) - (self.passenger_num_n - self.alight_num_d[next(iter(self.just_server_stop_data_d))])
                self.board_num_d[next(iter(self.just_server_stop_data_d))] = board_num
                strand_num = 0
                stop_passenger_list = traci.busstop.getPersonIDs(next(iter(self.just_server_stop_data_d)))
                for passenger_id in stop_passenger_list:
                    if self.belong_line_id_s in passenger_obj_dic_ex[passenger_id].passable_line_l:
                        strand_num += 1
                self.strand_num_d[next(iter(self.just_server_stop_data_d))] = strand_num
                want_board_num = board_num + strand_num
                self.want_board_num_d[next(iter(self.just_server_stop_data_d))] = want_board_num
                self.depart_stop_time_d[next(iter(self.just_server_stop_data_d))] = time_ex

                if next(iter(self.just_server_stop_data_d)) != list(bus_arrstation_od_otd_dict[self.belong_line_id_s].keys())[-1]:  # 表明该站不是该线路终点站
                    stop_obj_dic_ex[next(iter(self.just_server_stop_data_d))].update_service_data(bus_obj_dic_ex, self.bus_id_s, passenger_obj_dic_ex, {}, 0, "depart")    # 更新公交站的属性

                self.passenger_num_n = traci.vehicle.getPersonNumber(self.bus_id_s)
                self.just_server_stop_data_d[next(iter(self.just_server_stop_data_d))][1] = time_ex
                self.bus_passenger_d = {}
                for passenger_id in traci.vehicle.getPersonIDList(self.bus_id_s):
                    if not passenger_obj_dic_ex[passenger_id].transfer_edge_id_s:  # 表示非换乘乘客
                        od_otd = f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"
                    else:
                        od_otd = f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"

                    if od_otd not in self.bus_passenger_d:
                        self.bus_passenger_d[od_otd] = 1
                    else:
                        self.bus_passenger_d[od_otd] += 1

                # self.next_stop_id_s = line_obj_ex.get_next_stop_id_by_this_stop_id(self.next_stop_id_s)  # 修改为到站更新了
                if self.next_stop_id_s != "":
                    alight_num = 0
                    bus_passenger_list = traci.vehicle.getPersonIDList(self.bus_id_s)
                    for passenger_id in bus_passenger_list:
                        if passenger_obj_dic_ex[passenger_id].transfer_stop_id_s == self.next_stop_id_s or passenger_obj_dic_ex[passenger_id].end_stop_id_s == self.next_stop_id_s:
                            alight_num += 1
                    self.alight_num_d[self.next_stop_id_s] = alight_num
        # 如果上一步仿真，公交车在信号灯（信号灯进口车道）
        # Signal processing disabled
        if self.bus_state_s == "Signal":
            # 判断公交车是否离开信号灯（信号灯进口车道）
            next_signal_list = traci.vehicle.getNextTLS(self.bus_id_s)
            if len(next_signal_list) == 0 or next_signal_list[0][0] != self.next_signal_id_s:
                # 记录排队结束时间
                if self.next_signal_id_s in self.arriver_signal_time_d.keys() and self.next_signal_id_s not in self.depart_signal_time_d.keys():
                    self.depart_signal_time_d[self.next_signal_id_s] = time_ex
                signal_obj_dic_ex[self.next_signal_id_s].update_service_data(self)    # 更新信号灯的属性
                if len(next_signal_list) == 0:
                    self.bus_state_s = "Edge"
                    self.next_signal_id_s = ""
                    self.next_signal_link_s = ""
                    self.next_signal_lane_s = ""
                    self.next_signal_phase_s = ""
                else:
                    self.next_signal_id_s = next_signal_list[0][0]
                    self.next_signal_link_s = str(next_signal_list[0][1])
                    self.next_signal_lane_s = signal_obj_dic_ex[self.next_signal_id_s].connection_d[self.next_signal_link_s][0]
                    self.next_signal_phase_s = signal_obj_dic_ex[self.next_signal_id_s].connection_d[self.next_signal_link_s][3]
                    if traci.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
                        self.bus_state_s = "Signal"
                    else:
                        self.bus_state_s = "Edge"
            else:
                # 判断公交车是否在排队
                if traci.vehicle.getSpeed(self.bus_id_s) <= 0.1:
                    # 如果有排队记录还存在<0.1的情况，说明上一次不是最后一次最接近信号灯的排队，需要删除更新排队
                    if self.next_signal_id_s in self.arriver_signal_time_d.keys() and self.next_signal_id_s in self.depart_signal_time_d.keys():
                        del self.arriver_signal_time_d[self.next_signal_id_s]
                        del self.depart_signal_time_d[self.next_signal_id_s]
                    # 记录排队开始时间
                    if self.next_signal_id_s not in self.arriver_signal_time_d.keys():
                        self.arriver_signal_time_d[self.next_signal_id_s] = time_ex
                        self.arriver_signal_queue_info_l[0] = self.bus_cur_lane_s[:-2]
                        self.arriver_signal_queue_info_l[1] = time_ex
                        self.arriver_signal_queue_info_l[2] = traci.vehicle.getNextTLS(self.bus_id_s)[0][2]
                else:
                    # 记录排队结束时间
                    if self.next_signal_id_s in self.arriver_signal_time_d.keys() and self.next_signal_id_s not in self.depart_signal_time_d.keys():
                        self.depart_signal_time_d[self.next_signal_id_s] = time_ex
        # 更新每秒都需更新的状态
        self.bus_speed_n = traci.vehicle.getSpeed(self.bus_id_s)
        self.distance_n = traci.vehicle.getDistance(self.bus_id_s)
        self.bus_cur_lane_s = traci.vehicle.getLaneID(self.bus_id_s)
        # 保存公交车经过的车道id
        if self.bus_cur_lane_s not in self.bus_served_lane_l:
            self.bus_served_lane_l.append(self.bus_cur_lane_s)

        if self.next_stop_id_s != "":
            next_stop_length_n = 0
            # Use cached lane ID
            current_lane = self.bus_cur_lane_s
            
            if not current_lane:
                # If lane is empty (e.g. teleporting), we cannot calculate distance accurately.
                # Set to -1 or handle gracefully.
                next_stop_length_n = -1
            elif current_lane == stop_obj_dic_ex[self.next_stop_id_s].at_lane_s:  # 这里的traci.vehicle.getLaneID(self.bus_id_s)包括了路网内部连接的长度
                next_stop_length_n += (traci.lane.getLength(stop_obj_dic_ex[self.next_stop_id_s].at_lane_s) -
                                       traci.vehicle.getLanePosition(self.bus_id_s))
            else:
                next_stop_length_n += (traci.lane.getLength(current_lane) -
                                       traci.vehicle.getLanePosition(self.bus_id_s))
                for traci_lane_obj in self._get_next_links():
                    # 判断是否公交车处于临进入公交站情况，这种情况下traci.vehicle.getNextLinks(self.bus_id_s)会直接跳过临进入的公交站，导致无法获取准确的公交车到达下一个公交站的距离
                    busline_edge_l = list(sorted_busline_edge_d[self.belong_line_id_s].keys())
                    
                    # Safety check: ensure keys exist in list before index()
                    current_edge_key = traci_lane_obj[0][:-2]
                    target_edge_key = stop_obj_dic_ex[self.next_stop_id_s].at_lane_s[:-2]
                    
                    if target_edge_key in busline_edge_l and current_edge_key in busline_edge_l:
                        temp_next_stop_index = busline_edge_l.index(target_edge_key)
                        traci_lane_obj_0_index = busline_edge_l.index(current_edge_key)
                        if temp_next_stop_index < traci_lane_obj_0_index:  # 表示 traci.vehicle.getNextLinks(self.bus_id_s)找到的下一个路段已经跳过了马上要进入的这个车站所在的路段
                            break

                    next_stop_length_n += traci.lane.getLength(traci_lane_obj[0]) + traci_lane_obj[-1]
                    if traci_lane_obj[0][:-2] == stop_obj_dic_ex[self.next_stop_id_s].at_lane_s[:-2]:
                        break
            self.next_stop_length_n = next_stop_length_n
        else:
            self.next_stop_length_n = -1

    def update_subsequent_signal_info(self, time_ex, signal_obj_dic_ex, busline_tl_time_d):
        next_signal_list = traci.vehicle.getNextTLS(self.bus_id_s)
        for signal_list in next_signal_list:
            self.subsequent_signal_info_d[signal_list[0]] = [time_ex]

            cur_phase_index, phase_over_time, cycle_over_time = signal_obj_dic_ex[signal_list[0]].update_current_phase_and_remain_time(time_ex, signal_list[1])
            self.subsequent_signal_info_d[signal_list[0]].extend([cur_phase_index])

            self.subsequent_signal_info_d[signal_list[0]].extend([float(num) for num in busline_tl_time_d[self.belong_line_id_s][signal_list[0]]])  # 添加 要经过该信号灯相位的时长以及信号周期时长

            signal_inlane_id = signal_obj_dic_ex[signal_list[0]].connection_d[str(signal_list[1])][0]
            queue_length = traci.lane.getLastStepLength(signal_inlane_id) * traci.lane.getLastStepHaltingNumber(signal_inlane_id)
            self.subsequent_signal_info_d[signal_list[0]].extend([phase_over_time, cycle_over_time, queue_length])

            # 在增加一个交叉口的平均延误时间
            red_phase_time = self.subsequent_signal_info_d[signal_list[0]][3] - self.subsequent_signal_info_d[signal_list[0]][2]
            self.subsequent_signal_info_d[signal_list[0]].append(red_phase_time*red_phase_time/self.subsequent_signal_info_d[signal_list[0]][3])

            debug = 0
