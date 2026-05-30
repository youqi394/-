from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_SCHEDULE = "节假日发车时刻表(1).csv"
DEFAULT_HOURLY = "2026-05-26T06-59_export (1).csv"

ENDPOINTS = ("A", "B")
ENDPOINT_LABEL = {"A": "四惠枢纽站", "B": "老山公交场站"}


@dataclass(frozen=True)
class HourParam:
    hour: int
    energy_fraction: float
    runtime_min: float
    carbon_factor: float


@dataclass(frozen=True)
class Trip:
    id: int
    origin: str
    dest: str
    depart_min: float
    hour: int
    runtime_min: float
    energy_fraction: float
    carbon_factor: float
    label: str


@dataclass(frozen=True)
class Slot:
    id: int
    start_min: float
    end_min: float
    hour: int
    price: float
    carbon_factor: float


@dataclass
class Config:
    real_total_vehicles: int = 87
    initial_inventory: dict[str, int] = field(default_factory=lambda: {"A": 43, "B": 44})
    min_end_inventory: dict[str, int] = field(default_factory=lambda: {"A": 40, "B": 40})
    charger_capacity: dict[str, int] = field(default_factory=lambda: {"A": 40, "B": 40})
    slot_minutes: int = 5
    max_late_minutes: float = 5.0
    rest_minutes: float = 25.0
    soc_full: float = 1.0
    soc_min: float = 0.2
    recharge_from_min_to_full_minutes: float = 20.0
    late_penalty_per_min: float = 1.49
    vehicle_fixed_cost: float = 300.0
    carbon_cost_per_kg: float = 0.082
    battery_kwh_for_cost: float = 138.55
    ready_time_min: float = 29.0 * 60.0
    op_charge_end_min: float = 24.0 * 60.0

    @property
    def charge_rate_per_min(self) -> float:
        return (self.soc_full - self.soc_min) / self.recharge_from_min_to_full_minutes

    @property
    def charge_per_slot(self) -> float:
        return self.charge_rate_per_min * self.slot_minutes


@dataclass
class VehicleState:
    id: int
    start_endpoint: str
    loc: str
    current_end: float
    soc: float
    activities: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Solution:
    algorithm: str
    feasible: bool
    objective: float
    lower_bound: float
    relative_gap_to_lb: float
    runtime_sec: float
    trips: int
    vehicles_used: int
    total_late_min: float
    operating_charge_q: float
    post_charge_q: float
    operating_charge_cost: float
    post_charge_cost: float
    fixed_vehicle_cost: float
    late_cost: float
    start_inventory: dict[str, int]
    end_inventory: dict[str, int]
    final_inventory: dict[str, int]
    min_soc_observed: float
    max_slot_occupancy: dict[str, int]
    violation_messages: list[str]
    routes: list[VehicleState]
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_time_to_minutes(value: str) -> float:
    hour_str, minute_str = value.strip().split(":")
    return int(hour_str) * 60 + int(minute_str)


def fmt_time(minutes: float) -> str:
    minutes_i = int(round(minutes))
    hour = minutes_i // 60
    minute = minutes_i % 60
    return f"{hour:02d}:{minute:02d}"


def parse_percent(value: str) -> float:
    return float(value.strip().replace("%", "")) / 100.0


def tou_price(hour: int) -> float:
    hour %= 24
    if 23 <= hour or hour < 7:
        return 1.1946
    if 7 <= hour < 10 or 15 <= hour < 18 or 21 <= hour < 23:
        return 1.4950
    return 1.8044


def read_hour_params(path: Path) -> dict[int, HourParam]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    params: dict[int, HourParam] = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        hour = int(row[0])
        params[hour] = HourParam(
            hour=hour,
            energy_fraction=parse_percent(row[4]),
            runtime_min=float(row[5]),
            carbon_factor=float(row[6]),
        )
    if set(params) != set(range(24)):
        raise ValueError("Hourly parameter CSV must contain rows for hours 0..23.")
    return params


def read_trips(path: Path, hour_params: dict[int, HourParam]) -> list[Trip]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    raw: list[tuple[str, str, float]] = []
    for row in rows[1:]:
        if len(row) >= 1 and row[0].strip():
            raw.append(("A", "B", parse_time_to_minutes(row[0])))
        if len(row) >= 2 and row[1].strip():
            raw.append(("B", "A", parse_time_to_minutes(row[1])))
    raw.sort(key=lambda item: (item[2], item[0], item[1]))
    trips: list[Trip] = []
    for idx, (origin, dest, depart_min) in enumerate(raw):
        hour = int(depart_min // 60) % 24
        hp = hour_params[hour]
        trips.append(
            Trip(
                id=idx,
                origin=origin,
                dest=dest,
                depart_min=depart_min,
                hour=hour,
                runtime_min=hp.runtime_min,
                energy_fraction=hp.energy_fraction,
                carbon_factor=hp.carbon_factor,
                label=f"T{idx + 1:03d}_{origin}_to_{dest}_{int(depart_min // 60):02d}{int(depart_min % 60):02d}",
            )
        )
    if not trips:
        raise ValueError("Trip CSV contains no trips.")
    return trips


def load_instance(
    data_dir: Path = DATA_DIR,
    schedule_file: str = DEFAULT_SCHEDULE,
    hourly_file: str = DEFAULT_HOURLY,
) -> tuple[list[Trip], dict[int, HourParam]]:
    hour_params = read_hour_params(data_dir / hourly_file)
    trips = read_trips(data_dir / schedule_file, hour_params)
    return trips, hour_params


def make_slots(trips: list[Trip], hour_params: dict[int, HourParam], config: Config) -> list[Slot]:
    start = math.floor(min(t.depart_min for t in trips) / config.slot_minutes) * config.slot_minutes
    slots: list[Slot] = []
    t = start
    sid = 0
    while t < config.ready_time_min - 1e-9:
        hour = int((t % (24 * 60)) // 60)
        slots.append(
            Slot(
                id=sid,
                start_min=t,
                end_min=t + config.slot_minutes,
                hour=hour,
                price=tou_price(hour),
                carbon_factor=hour_params[hour].carbon_factor,
            )
        )
        sid += 1
        t += config.slot_minutes
    return slots


def q_cost_coeff(slot: Slot, config: Config) -> float:
    return config.battery_kwh_for_cost * (slot.price + config.carbon_cost_per_kg * slot.carbon_factor)


def compute_vehicle_count_lb(trips: list[Trip], config: Config) -> int:
    events: list[tuple[float, int]] = []
    for trip in trips:
        events.append((trip.depart_min, 1))
        events.append((trip.depart_min + trip.runtime_min + config.rest_minutes, -1))
    active = 0
    best = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        best = max(best, active)
    return best


def lower_bound(trips: list[Trip], slots: list[Slot], config: Config) -> float:
    lb_vehicles = compute_vehicle_count_lb(trips, config)
    total_energy = sum(t.energy_fraction for t in trips)
    min_q_cost = min(q_cost_coeff(slot, config) for slot in slots)
    return lb_vehicles * config.vehicle_fixed_cost + total_energy * min_q_cost


def charge_cost_for_slots(slot_ids: list[int], q_need: float, slots_by_id: dict[int, Slot], config: Config) -> float:
    remaining = q_need
    cost = 0.0
    for gid in sorted(slot_ids, key=lambda sid: (q_cost_coeff(slots_by_id[sid], config), slots_by_id[sid].start_min)):
        if remaining <= 1e-9:
            break
        amount = min(config.charge_per_slot, remaining)
        cost += q_cost_coeff(slots_by_id[gid], config) * amount
        remaining -= amount
    if remaining > 1e-7:
        raise ValueError("Insufficient selected charging slots for requested energy.")
    return cost


def find_contiguous_charge(
    endpoint: str,
    earliest_start: float,
    latest_end: float,
    q_need: float,
    slots: list[Slot],
    slots_by_id: dict[int, Slot],
    op_slot_ids: list[int],
    occupancy: dict[tuple[str, int], int],
    config: Config,
) -> dict[str, Any] | None:
    if q_need <= 1e-9:
        return None
    slots_needed = max(1, math.ceil(q_need / config.charge_per_slot - 1e-9))
    feasible: list[dict[str, Any]] = []
    op_set = set(op_slot_ids)
    for idx, first_gid in enumerate(op_slot_ids):
        first = slots_by_id[first_gid]
        if first.start_min + 1e-9 < math.ceil(earliest_start / config.slot_minutes) * config.slot_minutes:
            continue
        block = op_slot_ids[idx : idx + slots_needed]
        if len(block) < slots_needed:
            break
        if any(gid not in op_set for gid in block):
            continue
        if any(abs(slots_by_id[block[p]].end_min - slots_by_id[block[p + 1]].start_min) > 1e-9 for p in range(len(block) - 1)):
            continue
        if slots_by_id[block[-1]].end_min > latest_end + 1e-9:
            continue
        if any(occupancy.get((endpoint, gid), 0) >= config.charger_capacity[endpoint] for gid in block):
            continue
        feasible.append(
            {
                "start_min": slots_by_id[block[0]].start_min,
                "end_min": slots_by_id[block[-1]].end_min,
                "slot_ids": block,
                "q": q_need,
                "cost": charge_cost_for_slots(block, q_need, slots_by_id, config),
            }
        )
    if not feasible:
        return None
    return min(feasible, key=lambda item: (item["cost"], item["start_min"], item["end_min"]))


def candidate_sort_key(candidate: dict[str, Any], trip: Trip) -> tuple[float, float, int]:
    idle = max(0.0, candidate.get("idle_min", 0.0))
    soc_after = candidate.get("soc_after", 0.0)
    balance = candidate["incremental_cost"] + 0.035 * idle + 0.75 * max(0.0, 0.45 - soc_after)
    return (balance, candidate.get("actual_depart", trip.depart_min), candidate.get("vehicle_id", 9999))


def decode_with_random_keys(
    trips: list[Trip],
    hour_params: dict[int, HourParam],
    config: Config,
    genes: list[float] | None = None,
    top_k: int = 1,
    algorithm: str = "greedy",
) -> Solution:
    start_time = time.perf_counter()
    slots = make_slots(trips, hour_params, config)
    slots_by_id = {slot.id: slot for slot in slots}
    op_slot_ids = [slot.id for slot in slots if slot.start_min < config.op_charge_end_min - 1e-9]
    occupancy: dict[tuple[str, int], int] = {}
    vehicles: list[VehicleState] = []
    start_counts = {s: 0 for s in ENDPOINTS}
    total_late = 0.0
    op_charge_q = 0.0
    op_charge_cost = 0.0
    min_soc_seen = config.soc_full
    violation_messages: list[str] = []
    infeasible_penalty = 0.0

    for seq, trip in enumerate(sorted(trips, key=lambda t: (t.depart_min, t.origin, t.id))):
        latest_depart = trip.depart_min + config.max_late_minutes
        candidates: list[dict[str, Any]] = []
        for vehicle in vehicles:
            if vehicle.loc != trip.origin:
                continue
            ready_by_rest = vehicle.current_end + config.rest_minutes
            if ready_by_rest > latest_depart + 1e-9:
                continue

            if vehicle.soc - trip.energy_fraction >= config.soc_min - 1e-9:
                actual_depart = max(trip.depart_min, ready_by_rest)
                if actual_depart <= latest_depart + 1e-9:
                    late = max(0.0, actual_depart - trip.depart_min)
                    candidates.append(
                        {
                            "kind": "existing_direct",
                            "vehicle_id": vehicle.id,
                            "vehicle": vehicle,
                            "actual_depart": actual_depart,
                            "late": late,
                            "soc_before": vehicle.soc,
                            "soc_after": vehicle.soc - trip.energy_fraction,
                            "idle_min": actual_depart - vehicle.current_end,
                            "incremental_cost": config.late_penalty_per_min * late,
                        }
                    )

            if config.soc_full - trip.energy_fraction >= config.soc_min - 1e-9 and vehicle.soc < config.soc_full - 1e-9:
                q_need = config.soc_full - vehicle.soc
                charge = find_contiguous_charge(
                    trip.origin,
                    vehicle.current_end,
                    latest_depart,
                    q_need,
                    slots,
                    slots_by_id,
                    op_slot_ids,
                    occupancy,
                    config,
                )
                if charge is not None:
                    actual_depart = max(trip.depart_min, ready_by_rest, charge["end_min"])
                    if actual_depart <= latest_depart + 1e-9:
                        late = max(0.0, actual_depart - trip.depart_min)
                        candidates.append(
                            {
                                "kind": "existing_charge",
                                "vehicle_id": vehicle.id,
                                "vehicle": vehicle,
                                "charge": charge,
                                "actual_depart": actual_depart,
                                "late": late,
                                "soc_before": config.soc_full,
                                "soc_after": config.soc_full - trip.energy_fraction,
                                "idle_min": actual_depart - vehicle.current_end,
                                "incremental_cost": charge["cost"] + config.late_penalty_per_min * late,
                            }
                        )

        if (
            len(vehicles) < config.real_total_vehicles
            and start_counts[trip.origin] < config.initial_inventory[trip.origin]
            and config.soc_full - trip.energy_fraction >= config.soc_min - 1e-9
        ):
            candidates.append(
                {
                    "kind": "new_vehicle",
                    "vehicle_id": len(vehicles),
                    "actual_depart": trip.depart_min,
                    "late": 0.0,
                    "soc_before": config.soc_full,
                    "soc_after": config.soc_full - trip.energy_fraction,
                    "idle_min": 0.0,
                    "incremental_cost": config.vehicle_fixed_cost,
                }
            )

        if not candidates:
            violation_messages.append(f"No feasible assignment for {trip.label}.")
            infeasible_penalty += 1_000_000.0
            continue

        candidates.sort(key=lambda cand: candidate_sort_key(cand, trip))
        if genes is None:
            chosen = candidates[0]
        else:
            k = min(max(1, top_k), len(candidates))
            idx = min(k - 1, int(max(0.0, min(0.999999, genes[trip.id])) * k))
            chosen = candidates[idx]

        if chosen["kind"] == "new_vehicle":
            vehicle = VehicleState(
                id=len(vehicles),
                start_endpoint=trip.origin,
                loc=trip.dest,
                current_end=chosen["actual_depart"] + trip.runtime_min,
                soc=chosen["soc_after"],
            )
            vehicles.append(vehicle)
            start_counts[trip.origin] += 1
        else:
            vehicle = chosen["vehicle"]
            if chosen["kind"] == "existing_charge":
                charge = chosen["charge"]
                for gid in charge["slot_ids"]:
                    occupancy[(trip.origin, gid)] = occupancy.get((trip.origin, gid), 0) + 1
                op_charge_q += charge["q"]
                op_charge_cost += charge["cost"]
                vehicle.activities.append(
                    {
                        "type": "op_charge",
                        "endpoint": trip.origin,
                        "start_min": charge["start_min"],
                        "end_min": charge["end_min"],
                        "q": charge["q"],
                        "cost": charge["cost"],
                        "soc_before": vehicle.soc,
                        "soc_after": config.soc_full,
                        "slot_ids": charge["slot_ids"],
                    }
                )
                vehicle.soc = config.soc_full
            vehicle.loc = trip.dest
            vehicle.current_end = chosen["actual_depart"] + trip.runtime_min
            vehicle.soc = chosen["soc_after"]

        total_late += chosen["late"]
        min_soc_seen = min(min_soc_seen, chosen["soc_before"], chosen["soc_after"])
        vehicle.activities.append(
            {
                "type": "trip",
                "trip_id": trip.id,
                "trip_label": trip.label,
                "origin": trip.origin,
                "dest": trip.dest,
                "scheduled_depart": trip.depart_min,
                "actual_depart": chosen["actual_depart"],
                "arrival_min": chosen["actual_depart"] + trip.runtime_min,
                "late_min": chosen["late"],
                "runtime_min": trip.runtime_min,
                "energy_fraction": trip.energy_fraction,
                "soc_before": chosen["soc_before"],
                "soc_after": chosen["soc_after"],
            }
        )

    end_counts = {s: 0 for s in ENDPOINTS}
    for vehicle in vehicles:
        end_counts[vehicle.loc] += 1
    final_inventory = {
        s: config.initial_inventory[s] - start_counts[s] + end_counts[s]
        for s in ENDPOINTS
    }
    for s in ENDPOINTS:
        if final_inventory[s] < config.min_end_inventory[s]:
            shortage = config.min_end_inventory[s] - final_inventory[s]
            violation_messages.append(f"End inventory at {s} is short by {shortage}.")
            infeasible_penalty += shortage * 1_000_000.0

    post_charge_q, post_charge_cost = allocate_post_charging(
        vehicles,
        slots,
        slots_by_id,
        occupancy,
        trips,
        config,
        violation_messages,
    )
    if violation_messages:
        infeasible_penalty += 1_000_000.0 * len(violation_messages)

    fixed_cost = config.vehicle_fixed_cost * len(vehicles)
    late_cost = config.late_penalty_per_min * total_late
    objective = fixed_cost + late_cost + op_charge_cost + post_charge_cost + infeasible_penalty
    lb = lower_bound(trips, slots, config)
    gap = (objective - lb) / objective if objective > 0 else float("inf")
    max_slot_occupancy = {
        s: max([occupancy.get((s, slot.id), 0) for slot in slots] or [0])
        for s in ENDPOINTS
    }
    return Solution(
        algorithm=algorithm,
        feasible=not violation_messages,
        objective=objective,
        lower_bound=lb,
        relative_gap_to_lb=gap,
        runtime_sec=time.perf_counter() - start_time,
        trips=len(trips),
        vehicles_used=len(vehicles),
        total_late_min=total_late,
        operating_charge_q=op_charge_q,
        post_charge_q=post_charge_q,
        operating_charge_cost=op_charge_cost,
        post_charge_cost=post_charge_cost,
        fixed_vehicle_cost=fixed_cost,
        late_cost=late_cost,
        start_inventory=start_counts,
        end_inventory=end_counts,
        final_inventory=final_inventory,
        min_soc_observed=min_soc_seen,
        max_slot_occupancy=max_slot_occupancy,
        violation_messages=violation_messages,
        routes=vehicles,
        metadata={
            "vehicle_count_lower_bound": compute_vehicle_count_lb(trips, config),
            "total_trip_energy_fraction": sum(t.energy_fraction for t in trips),
            "charge_per_slot": config.charge_per_slot,
            "config": asdict(config),
        },
    )


def allocate_post_charging(
    vehicles: list[VehicleState],
    slots: list[Slot],
    slots_by_id: dict[int, Slot],
    occupancy: dict[tuple[str, int], int],
    trips: list[Trip],
    config: Config,
    violation_messages: list[str],
) -> tuple[float, float]:
    latest_trip_end = max((v.current_end for v in vehicles), default=config.op_charge_end_min)
    post_start = math.ceil(max(config.op_charge_end_min, latest_trip_end) / config.slot_minutes) * config.slot_minutes
    post_slot_ids = [slot.id for slot in slots if slot.start_min >= post_start - 1e-9]
    ranked_by_endpoint = {
        s: sorted(post_slot_ids, key=lambda gid: (q_cost_coeff(slots_by_id[gid], config), slots_by_id[gid].start_min))
        for s in ENDPOINTS
    }
    total_q = 0.0
    total_cost = 0.0
    for vehicle in vehicles:
        q_need = config.soc_full - vehicle.soc
        if q_need <= 1e-9:
            continue
        remaining = q_need
        selected: list[tuple[int, float]] = []
        for gid in ranked_by_endpoint[vehicle.loc]:
            if remaining <= 1e-9:
                break
            if occupancy.get((vehicle.loc, gid), 0) >= config.charger_capacity[vehicle.loc]:
                continue
            amount = min(config.charge_per_slot, remaining)
            occupancy[(vehicle.loc, gid)] = occupancy.get((vehicle.loc, gid), 0) + 1
            selected.append((gid, amount))
            remaining -= amount
        if remaining > 1e-7:
            violation_messages.append(f"Post-operation charging capacity is insufficient for vehicle {vehicle.id}.")
            continue
        total_q += q_need
        cost = sum(q_cost_coeff(slots_by_id[gid], config) * amount for gid, amount in selected)
        total_cost += cost
        vehicle.activities.extend(make_post_charge_activities(vehicle, selected, slots_by_id, config, cost))
        vehicle.soc = config.soc_full
    return total_q, total_cost


def make_post_charge_activities(
    vehicle: VehicleState,
    selected: list[tuple[int, float]],
    slots_by_id: dict[int, Slot],
    config: Config,
    total_cost: float,
) -> list[dict[str, Any]]:
    if not selected:
        return []
    selected_sorted = sorted(selected, key=lambda item: slots_by_id[item[0]].start_min)
    groups: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for gid, amount in selected_sorted:
        if not current:
            current.append((gid, amount))
            continue
        prev_gid = current[-1][0]
        if abs(slots_by_id[prev_gid].end_min - slots_by_id[gid].start_min) <= 1e-9:
            current.append((gid, amount))
        else:
            groups.append(current)
            current = [(gid, amount)]
    if current:
        groups.append(current)

    activities = []
    q_total = sum(amount for _, amount in selected)
    for group in groups:
        q = sum(amount for _, amount in group)
        group_cost = sum(q_cost_coeff(slots_by_id[gid], config) * amount for gid, amount in group)
        activities.append(
            {
                "type": "post_charge",
                "endpoint": vehicle.loc,
                "start_min": slots_by_id[group[0][0]].start_min,
                "end_min": slots_by_id[group[-1][0]].end_min,
                "q": q,
                "cost": group_cost,
                "soc_before": None,
                "soc_after": None,
                "slot_ids": [gid for gid, _ in group],
                "cost_share_check": total_cost * q / q_total if q_total > 0 else 0.0,
            }
        )
    return activities


def solution_summary_dict(solution: Solution) -> dict[str, Any]:
    return {
        "algorithm": solution.algorithm,
        "feasible": solution.feasible,
        "objective": solution.objective,
        "lower_bound": solution.lower_bound,
        "relative_gap_to_lb": solution.relative_gap_to_lb,
        "runtime_sec": solution.runtime_sec,
        "trips": solution.trips,
        "vehicles_used": solution.vehicles_used,
        "vehicle_count_lower_bound": solution.metadata.get("vehicle_count_lower_bound"),
        "total_late_min": solution.total_late_min,
        "fixed_vehicle_cost": solution.fixed_vehicle_cost,
        "late_cost": solution.late_cost,
        "operating_charge_q": solution.operating_charge_q,
        "post_charge_q": solution.post_charge_q,
        "operating_charge_cost": solution.operating_charge_cost,
        "post_charge_cost": solution.post_charge_cost,
        "start_inventory_A": solution.start_inventory["A"],
        "start_inventory_B": solution.start_inventory["B"],
        "end_inventory_A": solution.end_inventory["A"],
        "end_inventory_B": solution.end_inventory["B"],
        "final_inventory_A": solution.final_inventory["A"],
        "final_inventory_B": solution.final_inventory["B"],
        "min_soc_observed": solution.min_soc_observed,
        "max_slot_occupancy_A": solution.max_slot_occupancy["A"],
        "max_slot_occupancy_B": solution.max_slot_occupancy["B"],
        "violations": " | ".join(solution.violation_messages),
    }


def write_solution_outputs(solution: Solution, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = solution_summary_dict(solution)
    with (out_dir / f"{solution.algorithm}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with (out_dir / f"{solution.algorithm}_summary.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    trip_rows: list[dict[str, Any]] = []
    charge_rows: list[dict[str, Any]] = []
    for vehicle in solution.routes:
        trip_seq = 0
        charge_seq = 0
        for act in sorted(vehicle.activities, key=lambda item: (item.get("start_min", item.get("actual_depart", 0)), item["type"])):
            if act["type"] == "trip":
                trip_seq += 1
                trip_rows.append(
                    {
                        "vehicle_id": vehicle.id,
                        "sequence": trip_seq,
                        "trip_id": act["trip_id"] + 1,
                        "trip_label": act["trip_label"],
                        "origin": act["origin"],
                        "dest": act["dest"],
                        "scheduled_depart": fmt_time(act["scheduled_depart"]),
                        "actual_depart": fmt_time(act["actual_depart"]),
                        "arrival": fmt_time(act["arrival_min"]),
                        "late_min": round(act["late_min"], 6),
                        "runtime_min": round(act["runtime_min"], 6),
                        "energy_fraction": round(act["energy_fraction"], 8),
                        "soc_before": round(act["soc_before"], 8),
                        "soc_after": round(act["soc_after"], 8),
                    }
                )
            elif act["type"] in {"op_charge", "post_charge"}:
                charge_seq += 1
                charge_rows.append(
                    {
                        "vehicle_id": vehicle.id,
                        "sequence": charge_seq,
                        "kind": act["type"],
                        "endpoint": act["endpoint"],
                        "start": fmt_time(act["start_min"]),
                        "end": fmt_time(act["end_min"]),
                        "q": round(act["q"], 8),
                        "cost": round(act["cost"], 8),
                        "soc_before": "" if act.get("soc_before") is None else round(act["soc_before"], 8),
                        "soc_after": "" if act.get("soc_after") is None else round(act["soc_after"], 8),
                        "slot_ids": " ".join(str(gid) for gid in act.get("slot_ids", [])),
                    }
                )

    with (out_dir / f"{solution.algorithm}_trip_schedule.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        fields = [
            "vehicle_id",
            "sequence",
            "trip_id",
            "trip_label",
            "origin",
            "dest",
            "scheduled_depart",
            "actual_depart",
            "arrival",
            "late_min",
            "runtime_min",
            "energy_fraction",
            "soc_before",
            "soc_after",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trip_rows)

    with (out_dir / f"{solution.algorithm}_charge_schedule.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        fields = ["vehicle_id", "sequence", "kind", "endpoint", "start", "end", "q", "cost", "soc_before", "soc_after", "slot_ids"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(charge_rows)
