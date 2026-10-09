"""Small sufficient statistics for paired actions, without state trajectories."""


class Response:
    def __init__(self, bin_seconds=1800):
        self.bin_seconds = bin_seconds
        self.total = self.empty()
        self.bins = {"time": {}, "queue_ratio": {}, "occupancy": {}}

    @staticmethod
    def empty():
        return dict(decisions=0, old_hold_sum=0., new_hold_sum=0., actual_hold_sum=0.,
                    new_longer_count=0, old_longer_count=0,
                    actual_below1_count=0, actual_above10_count=0, actual_above59_count=0)

    def record(self, time, queue_ratio, occupancy, old_hold, new_hold, actual_hold):
        qbin = next((str(upper) for upper in [.1, .25, .5, 1.] if queue_ratio <= upper), "above1")
        loadbin = "empty" if occupancy == 0 else ("up_to_half" if occupancy <= .5 else "above_half")
        selected = [self.total]
        for name, key in [("time", str(int(time // self.bin_seconds))),
                          ("queue_ratio", qbin), ("occupancy", loadbin)]:
            selected.append(self.bins[name].setdefault(key, self.empty()))
        for row in selected:
            row["decisions"] += 1
            row["old_hold_sum"] += old_hold
            row["new_hold_sum"] += new_hold
            row["actual_hold_sum"] += actual_hold
            row["new_longer_count"] += int(new_hold - old_hold > 1.)
            row["old_longer_count"] += int(old_hold - new_hold > 1.)
            row["actual_below1_count"] += int(actual_hold < 1.)
            row["actual_above10_count"] += int(actual_hold > 10.)
            row["actual_above59_count"] += int(actual_hold >= 59.)

    def summary(self):
        return dict(total=self.total, bins=self.bins)


def merge(rows):
    output = Response().summary()
    for row in rows:
        for key, value in row["total"].items():
            output["total"][key] += value
        for name, bins in row["bins"].items():
            for label, values in bins.items():
                target = output["bins"][name].setdefault(label, Response.empty())
                for key, value in values.items():
                    target[key] += value
    return output


def describe(raw):
    def values(row):
        n = row["decisions"]
        return dict(decisions=n, old_holding_seconds=row["old_hold_sum"] / n,
                    new_holding_seconds=row["new_hold_sum"] / n,
                    new_minus_old_seconds=(row["new_hold_sum"] - row["old_hold_sum"]) / n,
                    actual_holding_seconds=row["actual_hold_sum"] / n,
                    new_longer_fraction=row["new_longer_count"] / n,
                    old_longer_fraction=row["old_longer_count"] / n,
                    actual_below1_fraction=row["actual_below1_count"] / n,
                    actual_above10_fraction=row["actual_above10_count"] / n,
                    actual_above59_fraction=row["actual_above59_count"] / n)
    return dict(total=values(raw["total"]),
                bins={name: {key: values(row) for key, row in bins.items()}
                      for name, bins in raw["bins"].items()})
