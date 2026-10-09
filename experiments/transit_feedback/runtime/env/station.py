from env.passenger import Passenger
import numpy as np


class Station(object):
    def __init__(self, station_type, station_id, station_name, direction, od):
        # if the station is terminal or not terminal,
        self.station_type = station_type
        # the id of stations
        self.station_id = station_id
        self.station_name = station_name
        # waiting passengers in this station
        self.waiting_passengers = np.array([])
        self.total_passenger = []
        # the direction is True if upstream, else False
        self.direction = direction
        # od is the passengers demand of every hour
        self.od = od

    # def station_update(self, current_time, stations):
    #     # 自己写的
    #     # if self.od is not None:
    #     #     # effective_period_str = effective_period[current_time//3600].strftime("%H:%M:%S")
    #     #     effective_period_str = '0'+str(6+current_time//3600)+':00:00' if 6+current_time//3600 < 10 else str(6+current_time//3600)+':00:00'
    #     #     period_od = self.od[effective_period_str]
    #     #     for destination_name, demand in period_od.items():
    #     #     # for destination_name in effective_station_name:
    #     #     # 对如果period_od[destination_name] == 0,则不计算泊松分布，因为太慢，且太多
    #     #         destination_demand_num = 0 if demand == 0 else np.random.poisson(demand/3600)
    #     #         for _ in range(destination_demand_num):
    #     #             destination = list(filter(lambda x: x.station_name == destination_name and x.direction == self.direction, stations))[0]
    #     #             passenger = Passenger(current_time, self, destination)
    #     #             self.waiting_passengers = np.append(self.waiting_passengers, passenger)
    #     #             self.total_passenger.append(passenger)
    #     #     sorted(self.waiting_passengers, key=lambda i: i.appear_time)
    #
    #     if self.od is not None: # GPT优化的，减少不必要的操作
    #
    #         effective_period_str = f"{6 + current_time // 3600:02}:00:00"
    #         period_od = self.od[effective_period_str]
    #
    #         for destination_name, demand in period_od.items():
    #             if demand > 0:  # 直接过滤掉不需要计算的需求
    #                 destination_demand_num = np.random.poisson(demand / 3600)
    #                 if destination_demand_num > 0:
    #                     destination = next(x for x in stations if x.station_name == destination_name and x.direction == self.direction)
    #                     new_passengers = [Passenger(current_time, self, destination) for _ in range(destination_demand_num)]
    #                     self.waiting_passengers = np.append(self.waiting_passengers, new_passengers)
    #                     self.total_passenger.extend(new_passengers)

    def station_update(self, current_time, stations, passenger_update_interval=1,
                        demand_multipliers=None, peak_shift=0):
        """
        每秒更新一次，减少不必要的泊松分布计算
        demand_multipliers: dict {hour: float} episode-level demand noise
        peak_shift: int, shift OD lookup hour by this amount
        """
        if self.od is not None:
            hour = 6 + min(current_time // 3600, 13)
            # Apply peak shift: look up demand from shifted hour
            lookup_hour = max(6, min(19, hour + peak_shift))
            effective_period_str = f"{lookup_hour:02}:00:00"
            period_od = self.od[effective_period_str]

            # Episode-level demand multiplier
            demand_mult = 1.0
            if demand_multipliers is not None and hour in demand_multipliers:
                demand_mult = demand_multipliers[hour]

            for destination_name, demand in period_od.items():
                if demand > 0:
                    demand_per_second = demand * demand_mult / 3600.0

                    destination_demand_num = np.random.poisson(demand_per_second * passenger_update_interval)

                    if destination_demand_num > 0:
                        destination = next(
                            x for x in stations
                            if x.station_name == destination_name and x.direction == self.direction
                        )

                        # 创建新乘客并更新等候队列
                        new_passengers = [
                            Passenger(current_time, self, destination)
                            for _ in range(destination_demand_num)
                        ]
                        self.waiting_passengers = np.append(self.waiting_passengers, new_passengers)
                        self.total_passenger.extend(new_passengers)
