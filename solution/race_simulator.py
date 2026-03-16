#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path
import argparse

class TireParams:
    def __init__(self, base_delta, deg_linear, deg_quadratic, age_temp_interaction, threshold=0):
        self.base_delta = base_delta
        self.deg_linear = deg_linear
        self.deg_quadratic = deg_quadratic
        self.age_temp_interaction = age_temp_interaction
        self.threshold = threshold

def load_parameters():
    global TIRE_MODEL, TEMP_REFERENCE_C, TEMP_SENSITIVITY
    config_path = Path(__file__).parent / "model_parameters.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.get("TIRE_MODEL", {}).items():
                TIRE_MODEL[k] = TireParams(
                    v["base_delta"], v["deg_linear"], v["deg_quadratic"], v["age_temp_interaction"],
                    v.get("threshold", 0)
                )
            TEMP_REFERENCE_C = data.get("TEMP_REFERENCE_C", TEMP_REFERENCE_C)
            sens = data.get("TEMP_SENSITIVITY", {})
            for k, v in sens.items():
                TEMP_SENSITIVITY[k] = v
        except Exception as e:
            print(f"Warning: Failed to load external parameters: {e}", file=sys.stderr)

def build_pit_map(pit_stops):
    return {int(stop["lap"]): stop["to_tire"] for stop in pit_stops}

def lap_time(base_lap_time, tire, tire_age, track_temp):
    p = TIRE_MODEL[tire]
    temp_delta = track_temp - TEMP_REFERENCE_C
    temp_effect = TEMP_SENSITIVITY[tire] * temp_delta
    
    effective_age = max(0, tire_age - p.threshold)
    degradation = p.deg_linear * effective_age + p.deg_quadratic * (effective_age * effective_age)
    interaction = p.age_temp_interaction * effective_age * temp_delta
    
    return base_lap_time + p.base_delta + degradation + temp_effect + interaction

def simulate_driver(race_config, strategy):
    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    current_tire = strategy["starting_tire"]
    pit_map = build_pit_map(strategy.get("pit_stops", []))

    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        total_time += lap_time(base_lap_time, current_tire, tire_age, track_temp)

        if lap in pit_map:
            total_time += pit_lane_time
            current_tire = pit_map[lap]
            tire_age = 0

    return total_time

def simulate_race(race_config, strategies):
    # strategies are keyed pos1..pos20 but race outcome depends on total times only.
    totals = []

    for key in sorted(strategies.keys(), key=lambda s: int(s[3:])):
        strategy = strategies[key]
        driver_id = strategy["driver_id"]
        race_time = simulate_driver(race_config, strategy)
        totals.append((driver_id, race_time))

    # Secondary driver_id key guarantees deterministic ordering.
    totals.sort(key=lambda item: (item[1], item[0]))
    return [driver_id for driver_id, _ in totals]


def validate_output(finishing_positions):
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

def predict_from_test_case(test_case):
    race_id = test_case["race_id"]
    race_config = test_case["race_config"]
    strategies = test_case["strategies"]

    finishing_positions = simulate_race(race_config, strategies)

    validate_output(finishing_positions)

    return {
        "race_id": race_id,
        "finishing_positions": finishing_positions,
    }

def solve_from_stdin():
    test_case = json.load(sys.stdin)
    return predict_from_test_case(test_case)    
    
# Baseline coefficients. Tune these from historical races.

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run F1 Race Simulator")
    parser.add_argument("--output", "-o", type=str, help="Optional: Path to save the output JSON file")

    # Parse known args so unexpected runner flags do not crash execution.
    if argv is None:
        args, _ = parser.parse_known_args()
    else:
        args, _ = parser.parse_known_args(argv)

    try:
        output_data = solve_from_stdin()
    except Exception as e:
        print(f"Error reading from stdin: {e}", file=sys.stderr)
        sys.exit(1)

    json_str = json.dumps(output_data)

    # Always print to stdout because test_runner.sh expects it.
    print(json_str)

    # If user provided an output file, save it there too.
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"Successfully saved output to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to save output to {args.output}: {e}", file=sys.stderr)



TIRE_MODEL = {
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

load_parameters()

if __name__ == "__main__":
    main()
