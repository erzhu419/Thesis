import copy

import traci


class Stop:    # 创建一个公交站类,用于描述每一个公交站的属性和行为

    def __init__(self, stop_id_s, at_edge_s, at_lane_s, service_line_l):
        """初始化公交站的函数"""
        """静态属性"""
        self.stop_id_s = stop_id_s               # 公交站标签，字符串类型
        self.at_edge_s = at_edge_s               # 公交站所在的路段，字符串类型
        self.at_lane_s = at_lane_s               # 公交站所在的车道，字符串类型
        self.service_line_l = service_line_l     # 公交站可以服务的线路，列表类型 [线路标签,线路标签,...]
        self.accessible_stop_d = {}              # 不需要换乘可直接到达的公交站，字典类型 下游公交站:[线路标签,线路标签,...]
        self.passenger_arriver_rate_d = {}       # 公交站的乘客需求（每小时去往每个下游站的到达率），字典类型 下游公交站:[第1小时到达率,第2小时到达率,...]
        """动态属性"""
        self.passenger_num_n = 0                 # 公交站的站内乘客数量，数值类型（触发更新）
        self.passenger_id_l = []                 # 公交站的站内乘客标签，列表类型（触发更新） [乘客标签,乘客标签,...]
        self.bus_num_n = 0                       # 公交站的站内公交车数量，数值类型（触发更新）
        self.bus_id_l = []                       # 公交站的站内公交车标签，列表类型（触发更新） [公交车标签,公交车标签,...]
        self.just_leave_data_l = []              # 该站刚刚服务公交车的信息，列表类型（到站、离站更新）（到站时预测公交离开时的信息，离站时根据实际再次更新） [公交车标签,线路标签,到达时间,离开时间,离站后站内乘客OD分布]
        self.unserved_bus_l = []                 # 该站还未服务的公交车，列表类型（到站更新） [公交车标签,公交车标签,...]
        """容器属性"""
        self.service_data_l = []                 # 服务公交车的信息，列表类型（离站更新） [公交车标签,线路标签,到达时间,离开时间,下车乘客数,上车乘客数,滞留乘客数]

    def get_accessible_stop(self, line_obj_dic_ex):
        accessible_stop_list = []
        for line_id in line_obj_dic_ex.keys():
            if self.stop_id_s in line_obj_dic_ex[line_id].stop_id_l:
                this_stop_id = self.stop_id_s
                while line_obj_dic_ex[line_id].get_next_stop_id_by_this_stop_id(this_stop_id) != "":
                    this_stop_id = line_obj_dic_ex[line_id].get_next_stop_id_by_this_stop_id(this_stop_id)
                    if this_stop_id not in accessible_stop_list:
                        accessible_stop_list.append(this_stop_id)
        for stop_id in accessible_stop_list:
            available_line_list = []
            for line_id in line_obj_dic_ex.keys():
                if self.stop_id_s in line_obj_dic_ex[line_id].stop_id_l and stop_id in line_obj_dic_ex[line_id].stop_id_l:
                    available_line_list.append(line_id)
            self.accessible_stop_d[stop_id] = available_line_list

    def get_passenger_arriver_rate(self, passenger_obj_dic_ex):
        for stop_id in self.accessible_stop_d.keys():
            self.passenger_arriver_rate_d[stop_id] = [0, 0, 0, 0, 0]
        for passenger_id in passenger_obj_dic_ex.keys():
            if passenger_obj_dic_ex[passenger_id].start_stop_id_s == self.stop_id_s:
                start_time = passenger_obj_dic_ex[passenger_id].start_time_n
                if passenger_obj_dic_ex[passenger_id].transfer_stop_id_s == "":
                    alight_stop = passenger_obj_dic_ex[passenger_id].end_stop_id_s
                else:
                    alight_stop = passenger_obj_dic_ex[passenger_id].transfer_stop_id_s
                self.passenger_arriver_rate_d[alight_stop][int(start_time // 3600)] += 1
            if passenger_obj_dic_ex[passenger_id].transfer_stop_id_s == self.stop_id_s:
                start_time = passenger_obj_dic_ex[passenger_id].start_time_n
                alight_stop = passenger_obj_dic_ex[passenger_id].end_stop_id_s
                self.passenger_arriver_rate_d[alight_stop][int(start_time // 3600)] += 1
        for stop_id in self.accessible_stop_d.keys():
            self.passenger_arriver_rate_d[stop_id] = [n / 3600 for n in self.passenger_arriver_rate_d[stop_id]]

    def get_initial_just_leave_data(self, line_station_od_otd_dict, trigger_time):
        """初始时刻，各公交站刚服务的公交车信息[公交车ID，公交线路，公交到站时间，公交发车时间，站内各OD乘客数量]"""
        self.just_leave_data_l = {}
        stop_service_buslines = self.service_line_l
        for busline in list(set(stop_service_buslines) & set(list(line_station_od_otd_dict.keys()))):  # 后面的set(list(line_station_od_otd_dict.keys())是选择部分线路实验所需
            if self.stop_id_s in line_station_od_otd_dict[busline]:  # 表明该站不是该线路终点站
                od_otd_data = {}
                for od_otd_id in line_station_od_otd_dict[busline][self.stop_id_s]:
                    od_otd_data[od_otd_id] = 0
                self.just_leave_data_l = [f"{busline}_0", busline, trigger_time-100, trigger_time-100, od_otd_data]  # 之前这两行都是减60
                self.service_data_l = [[f"{busline}_0", busline, trigger_time-100, trigger_time-100, od_otd_data]]
                break

        if not self.just_leave_data_l:  # 表明是所涉及到所有线路的终点站
            self.just_leave_data_l = [f"{stop_service_buslines[0]}_0", stop_service_buslines[0], trigger_time-100, trigger_time-100, {}]
            self.service_data_l = [[f"{stop_service_buslines[0]}_0", stop_service_buslines[0], trigger_time-100, trigger_time-100, {}]]


    def update_stop_state(self):    # 控制需要更新
        self.passenger_id_l = traci.busstop.getPersonIDs(self.stop_id_s)
        self.passenger_num_n = len(self.passenger_id_l)
        self.bus_id_l = traci.busstop.getVehicleIDs(self.stop_id_s)
        self.bus_num_n = len(self.bus_id_l)

    def update_service_data(self, bus_obj_ex, bus_id, passenger_obj_dic_ex, strand_passenger_od, predicted_depart_time, flag):    # 公交离站更新，到站也许更新（预测）

        passenger_l = traci.busstop.getPersonIDs(self.stop_id_s)
        if flag == "depart":  # 表示为公交车离站信息更新过程
            just_leave_data = [bus_obj_ex[bus_id].bus_id_s, bus_obj_ex[bus_id].belong_line_id_s,
                               bus_obj_ex[bus_id].arriver_stop_time_d[self.stop_id_s],
                               bus_obj_ex[bus_id].depart_stop_time_d[self.stop_id_s]]
            no_boarding_passenger_od_d = {}
            for passenger_id in passenger_l:
                if not passenger_obj_dic_ex[passenger_id].transfer_edge_id_s:  # 表示换乘乘客
                    if f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}" not in no_boarding_passenger_od_d:
                        no_boarding_passenger_od_d[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] = 1
                    else:
                        no_boarding_passenger_od_d[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] += 1
                else:  # 表示非换乘乘客
                    if f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}" not in no_boarding_passenger_od_d:
                        no_boarding_passenger_od_d[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] = 1
                    else:
                        no_boarding_passenger_od_d[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] += 1
        else:  # 表示为公交车到站信息预测过程
            just_leave_data = [bus_obj_ex[bus_id].bus_id_s, bus_obj_ex[bus_id].belong_line_id_s,
                               bus_obj_ex[bus_id].arriver_stop_time_d[self.stop_id_s],
                               predicted_depart_time]
            no_boarding_passenger_od_d = {}  # 不想上车的各OD乘客数量
            for passenger_id in passenger_l:
                if not passenger_obj_dic_ex[passenger_id].transfer_edge_id_s:  # 表示换乘乘客
                    if f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}" not in strand_passenger_od:
                        if f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}" not in no_boarding_passenger_od_d:
                            no_boarding_passenger_od_d[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] = 1
                        else:
                            no_boarding_passenger_od_d[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].transfer_edge_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] += 1

                else:
                    if f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}" not in strand_passenger_od:
                        if f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}" not in no_boarding_passenger_od_d:
                            no_boarding_passenger_od_d[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] = 1
                        else:
                            no_boarding_passenger_od_d[f"{passenger_obj_dic_ex[passenger_id].start_stop_id_s}-{passenger_obj_dic_ex[passenger_id].end_stop_id_s}"] += 1

        just_leave_data.append({**self.just_leave_data_l[-1], **strand_passenger_od, **no_boarding_passenger_od_d})
        self.just_leave_data_l = just_leave_data
        # service_data = just_leave_data + [bus_obj_ex.alight_num_d[self.stop_id_s], bus_obj_ex.board_num_d[self.stop_id_s], bus_obj_ex.strand_num_d[self.stop_id_s]]
        service_data = just_leave_data

        # 这里应该用离站站数据更新到站的估计数据
        service_data_bus_l = [sublist[0] for sublist in self.service_data_l]
        if service_data[0] in service_data_bus_l:
            index = service_data_bus_l.index(service_data[0])
            self.service_data_l[index] = copy.deepcopy(service_data)
            debug = 0
        else:
            self.service_data_l.append(service_data)

    def update_stop_unserved_bus(self, bus_id, line_obj_ex, stop_obj_dic_ex):  # 到站更新
        if bus_id not in self.unserved_bus_l:  # 若不在，则表明该站是线路第一个公交站，应当对后面公交站添加该公交车
            for stop_id in line_obj_ex.stop_id_l[1:]:  # 对除第一个站（即当前站bus_id）外的所有站添加该车作为未服务公交车
                stop_obj_dic_ex[stop_id].unserved_bus_l.append(bus_id)
        else:
            self.unserved_bus_l.remove(bus_id)







