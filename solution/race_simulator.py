#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path

# Default tire params - these get overridden by model_parameters.json if it exists
PARAMS = {
    "SOFT": {
        "offset": 2.958962002059359,  # how much faster SOFT is vs base
        "cliff":  10,                  # laps before degradation kicks in
        "deg":    0.3938052242423563,  # degradation per lap (linear)
        "deg2":   0.0,                 # degradation per lap (quadratic extra)
    },
    "MEDIUM": {
        "offset": 3.9262504324101357,
        "cliff":  20,
        "deg":    0.2005977212786465,
        "deg2":   0.0,
    },
    "HARD": {
        "offset": 4.7244861975727845,
        "cliff":  30,
        "deg":    0.10319025698674321,
        "deg2":   0.0,
    },
    "temp_coef":     0.112095,  # hotter track = tires degrade faster
    "fuel_effect":   0.00010257806706596489, # car gets lighter as fuel burns
    "temp_base_coef": 0.0,      # direct effect of temp on base lap time
}


def load_parameters():
    # Try to load calibrated params - pick the one with the best score
    global PARAMS
    candidates = [
        Path(__file__).parent / "model_parameters.json",
        Path(__file__).parent / "model_parameters_checkpoint.json",
    ]
    best_data = None
    best_score = float("-inf")

    for p in candidates:
        if not p.exists():
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            score = data.get("BEST_SCORE", float("-inf"))
            if score > best_score:
                best_score = score
                best_data = data
        except Exception as e:
            print(f"Warning: could not load {p.name}: {e}", file=sys.stderr)

    if best_data and "PARAMS" in best_data:
        PARAMS.update(best_data["PARAMS"])


def simulate_driver(race_config, strategy):
    # Pull race info
    total_laps   = int(race_config["total_laps"])
    base_time    = float(race_config["base_lap_time"])
    pit_time     = float(race_config["pit_lane_time"])
    temp         = float(race_config["track_temp"])

    current_tire = strategy["starting_tire"]
    pit_map = {int(s["lap"]): s["to_tire"] for s in strategy.get("pit_stops", [])}

    tire_age   = 0
    total_time = 0.0
    last_lap_t = 0.0

    # Pre-calculate temperature effects once (same across all laps)
    temp_deg_mult    = 1.0 + temp * PARAMS["temp_coef"]
    temp_base_effect = temp * PARAMS["temp_base_coef"]

    for lap in range(1, total_laps + 1):
        tire_age += 1  # tire age starts at 1 on first lap of a stint
        tire = PARAMS[current_tire]

        # Build this lap's time: base + compound offset + temp effect - fuel save
        fuel_saving = PARAMS["fuel_effect"] * lap
        lap_t = base_time + tire["offset"] + temp_base_effect - fuel_saving

        # Add degradation only after the cliff (grace period)
        over_cliff = max(0, tire_age - tire["cliff"])
        if over_cliff > 0:
            lap_t += over_cliff * tire["deg"] * temp_deg_mult
            lap_t += over_cliff * over_cliff * tire["deg2"] * temp_deg_mult

        total_time += lap_t
        last_lap_t  = lap_t

        # Pit stop at end of this lap
        if lap in pit_map:
            total_time  += pit_time
            current_tire = pit_map[lap]
            tire_age     = 0

    return total_time, last_lap_t


def simulate_race(race_config, strategies):
    results = []
    for strat in strategies.values():
        total, last_lap = simulate_driver(race_config, strat)
        results.append((total, last_lap, strat["driver_id"]))

    # Sort by total race time, then last lap time, then driver ID (keeps it deterministic)
    results.sort(key=lambda x: (x[0], x[1], x[2]))
    return [r[2] for r in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, help="Optional path to save output JSON")
    args, _ = parser.parse_known_args()

    try:
        test_case = json.load(sys.stdin)
    except Exception as e:
        print(f"Error reading stdin: {e}", file=sys.stderr)
        sys.exit(1)

    finishing = simulate_race(test_case["race_config"], test_case["strategies"])
    output    = {"race_id": test_case["race_id"], "finishing_positions": finishing}
    json_str  = json.dumps(output)
    print(json_str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)


load_parameters()

if __name__ == "__main__":
    main()
