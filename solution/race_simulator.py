#!/usr/bin/env python3
"""Race simulator with optional built-in calibration mode.

Default mode (used by evaluator):
- reads one race JSON from stdin
- outputs one prediction JSON to stdout

Calibration mode:
- samples historical races
- optimizes tunable tire/temperature constants
- writes best constants back into this file automatically
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TireParams:
    base_delta: float
    deg_linear: float
    deg_quadratic: float
    age_temp_interaction: float


@dataclass
class ModelParams:
    tire_model: Dict[str, TireParams]
    temp_reference_c: float
    temp_sensitivity: Dict[str, float]

    def clone(self) -> "ModelParams":
        return ModelParams(
            tire_model={
                k: TireParams(v.base_delta, v.deg_linear, v.deg_quadratic, v.age_temp_interaction)
                for k, v in self.tire_model.items()
            },
            temp_reference_c=self.temp_reference_c,
            temp_sensitivity=dict(self.temp_sensitivity),
        )


# Baseline coefficients. Tune these from historical races.
TIRE_MODEL: Dict[str, TireParams] = {
    "SOFT": TireParams(base_delta=-0.701817, deg_linear=0.072085, deg_quadratic=0.00321308, age_temp_interaction=0.00000000),
    "MEDIUM": TireParams(base_delta=-0.029494, deg_linear=0.000000, deg_quadratic=0.00138054, age_temp_interaction=0.00000000),
    "HARD": TireParams(base_delta=0.187366, deg_linear=0.006512, deg_quadratic=0.00025869, age_temp_interaction=0.00000000),
}

TEMP_REFERENCE_C = 35.320459
TEMP_SENSITIVITY = {
    "SOFT": 0.012182,
    "MEDIUM": 0.012859,
    "HARD": 0.016805,
}

def load_parameters() -> None:
    global TIRE_MODEL, TEMP_REFERENCE_C, TEMP_SENSITIVITY
    config_path = Path(__file__).parent / "model_parameters.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.get("TIRE_MODEL", {}).items():
                TIRE_MODEL[k] = TireParams(
                    v["base_delta"], v["deg_linear"], v["deg_quadratic"], v["age_temp_interaction"]
                )
            TEMP_REFERENCE_C = data.get("TEMP_REFERENCE_C", TEMP_REFERENCE_C)
            sens = data.get("TEMP_SENSITIVITY", {})
            for k, v in sens.items():
                TEMP_SENSITIVITY[k] = v
        except Exception as e:
            print(f"Warning: Failed to load external parameters: {e}", file=sys.stderr)

load_parameters()


def _default_model_params() -> ModelParams:
    return ModelParams(
        tire_model={
            "SOFT": TireParams(
                base_delta=TIRE_MODEL["SOFT"].base_delta,
                deg_linear=TIRE_MODEL["SOFT"].deg_linear,
                deg_quadratic=TIRE_MODEL["SOFT"].deg_quadratic,
                age_temp_interaction=TIRE_MODEL["SOFT"].age_temp_interaction,
            ),
            "MEDIUM": TireParams(
                base_delta=TIRE_MODEL["MEDIUM"].base_delta,
                deg_linear=TIRE_MODEL["MEDIUM"].deg_linear,
                deg_quadratic=TIRE_MODEL["MEDIUM"].deg_quadratic,
                age_temp_interaction=TIRE_MODEL["MEDIUM"].age_temp_interaction,
            ),
            "HARD": TireParams(
                base_delta=TIRE_MODEL["HARD"].base_delta,
                deg_linear=TIRE_MODEL["HARD"].deg_linear,
                deg_quadratic=TIRE_MODEL["HARD"].deg_quadratic,
                age_temp_interaction=TIRE_MODEL["HARD"].age_temp_interaction,
            ),
        },
        temp_reference_c=TEMP_REFERENCE_C,
        temp_sensitivity={
            "SOFT": TEMP_SENSITIVITY["SOFT"],
            "MEDIUM": TEMP_SENSITIVITY["MEDIUM"],
            "HARD": TEMP_SENSITIVITY["HARD"],
        },
    )


def _build_pit_map(pit_stops: List[dict]) -> Dict[int, str]:
    return {int(stop["lap"]): stop["to_tire"] for stop in pit_stops}


def _lap_time(base_lap_time: float, tire: str, tire_age: int, track_temp: float) -> float:
    p = TIRE_MODEL[tire]
    temp_delta = track_temp - TEMP_REFERENCE_C
    temp_effect = TEMP_SENSITIVITY[tire] * temp_delta
    degradation = p.deg_linear * tire_age + p.deg_quadratic * (tire_age * tire_age)
    interaction = p.age_temp_interaction * tire_age * temp_delta
    return base_lap_time + p.base_delta + degradation + temp_effect + interaction


def _lap_time_with_params(
    base_lap_time: float,
    tire: str,
    tire_age: int,
    track_temp: float,
    params: ModelParams,
) -> float:
    p = params.tire_model[tire]
    temp_delta = track_temp - params.temp_reference_c
    temp_effect = params.temp_sensitivity[tire] * temp_delta
    degradation = p.deg_linear * tire_age + p.deg_quadratic * (tire_age * tire_age)
    interaction = p.age_temp_interaction * tire_age * temp_delta
    return base_lap_time + p.base_delta + degradation + temp_effect + interaction


def _simulate_driver(race_config: dict, strategy: dict) -> float:
    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    current_tire = strategy["starting_tire"]
    pit_map = _build_pit_map(strategy.get("pit_stops", []))

    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        total_time += _lap_time(base_lap_time, current_tire, tire_age, track_temp)

        if lap in pit_map:
            total_time += pit_lane_time
            current_tire = pit_map[lap]
            tire_age = 0

    return total_time


def _simulate_driver_with_params(race_config: dict, strategy: dict, params: ModelParams) -> float:
    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    current_tire = strategy["starting_tire"]
    pit_map = _build_pit_map(strategy.get("pit_stops", []))

    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        total_time += _lap_time_with_params(
            base_lap_time, current_tire, tire_age, track_temp, params
        )

        if lap in pit_map:
            total_time += pit_lane_time
            current_tire = pit_map[lap]
            tire_age = 0

    return total_time


def simulate_race(race_config: dict, strategies: dict) -> List[str]:
    # strategies are keyed pos1..pos20 but race outcome depends on total times only.
    totals: List[Tuple[str, float]] = []

    for key in sorted(strategies.keys(), key=lambda s: int(s[3:])):
        strategy = strategies[key]
        driver_id = strategy["driver_id"]
        race_time = _simulate_driver(race_config, strategy)
        totals.append((driver_id, race_time))

    # Secondary driver_id key guarantees deterministic ordering.
    totals.sort(key=lambda item: (item[1], item[0]))
    return [driver_id for driver_id, _ in totals]


def _simulate_race_with_params(race_config: dict, strategies: dict, params: ModelParams) -> List[str]:
    totals: List[Tuple[str, float]] = []

    for key in sorted(strategies.keys(), key=lambda s: int(s[3:])):
        strategy = strategies[key]
        driver_id = strategy["driver_id"]
        race_time = _simulate_driver_with_params(race_config, strategy, params)
        totals.append((driver_id, race_time))

    totals.sort(key=lambda item: (item[1], item[0]))
    return [driver_id for driver_id, _ in totals]


def _validate_output(finishing_positions: List[str]) -> None:
    if len(finishing_positions) != 20:
        raise ValueError("finishing_positions must contain exactly 20 driver IDs")

    if len(set(finishing_positions)) != 20:
        raise ValueError("finishing_positions must not contain duplicates")

    expected = {f"D{i:03d}" for i in range(1, 21)}
    actual = set(finishing_positions)
    if actual != expected:
        missing = sorted(expected - actual)
        extras = sorted(actual - expected)
        raise ValueError(
            "finishing_positions must contain all drivers D001-D020 exactly once; "
            f"missing={missing}, extras={extras}"
        )

    invalid = [driver for driver in finishing_positions if not re.fullmatch(r"D\d{3}", driver)]
    if invalid:
        raise ValueError(f"driver IDs must match D### format, invalid={invalid}")




def _solve_from_stdin() -> None:
    test_case = json.load(sys.stdin)

    output = predict_from_test_case(test_case)
    print(json.dumps(output))


def predict_from_test_case(test_case: dict) -> dict:
    race_id = test_case["race_id"]
    race_config = test_case["race_config"]
    strategies = test_case["strategies"]

    finishing_positions = simulate_race(race_config, strategies)

    _validate_output(finishing_positions)

    return {
        "race_id": race_id,
        "finishing_positions": finishing_positions,
    }


def main(argv: List[str] | None = None) -> None:
    _solve_from_stdin()


if __name__ == "__main__":
    main()
