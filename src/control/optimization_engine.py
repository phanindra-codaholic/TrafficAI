"""Traffic signal scheduling."""

from dataclasses import dataclass
from typing import List


@dataclass
class LaneData:
    lane_id: int
    vehicle_count: int
    wait_time: int
    has_emergency: bool


@dataclass
class SignalDecision:
    lane_id: int
    green_time: int
    priority_rank: int
    vehicle_count: int
    is_emergency: bool


class OptimizationEngine:
    """Build a priority-ordered signal cycle."""

    def __init__(self):
        self.base_green_time = 5
        self.time_per_vehicle = 2
        self.max_green_time = 30
        self.aging_factor = 10

    def calculate_signal_cycle(self, all_lanes: List[LaneData]) -> List[SignalDecision]:
        """Return an ordered schedule for all lanes."""
        cycle_plan: List[SignalDecision] = []
        current_rank = 1

        for lane in all_lanes:
            if lane.has_emergency:
                cycle_plan.append(SignalDecision(
                    lane_id=lane.lane_id,
                    green_time=self.max_green_time,
                    priority_rank=current_rank,
                    vehicle_count=lane.vehicle_count,
                    is_emergency=True,
                ))
                current_rank += 1

        weighted: List[tuple[int, LaneData]] = []
        for lane in all_lanes:
            if not lane.has_emergency:
                weight = lane.vehicle_count + (lane.wait_time // self.aging_factor)
                weighted.append((weight, lane))

        weighted.sort(key=lambda x: x[0], reverse=True)

        for _weight, lane in weighted:
            calc_time = self.base_green_time + (lane.vehicle_count * self.time_per_vehicle)
            if calc_time > self.max_green_time:
                calc_time = self.max_green_time

            cycle_plan.append(SignalDecision(
                lane_id=lane.lane_id,
                green_time=calc_time,
                priority_rank=current_rank,
                vehicle_count=lane.vehicle_count,
                is_emergency=False,
            ))
            current_rank += 1

        return cycle_plan
