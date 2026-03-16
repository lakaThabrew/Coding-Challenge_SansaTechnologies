#!/usr/bin/env python3
"""
Stand-alone Calibration script.
This runs the optimization algorithm to find the best constants and saves them
to `model_parameters.json` and records a log in `calibration_log.txt`
"""

import argparse
import glob
import json
import math
import random
import datetime
import sys
from pathlib import Path

from race_simulator import (
    TireParams,
    TIRE_MODEL,
    TEMP_REFERENCE_C,
    TEMP_SENSITIVITY,
    build_pit_map,
)

class ModelParams:
    def __init__(self, tire_model, temp_reference_c, temp_sensitivity):
        self.tire_model = tire_model
        self.temp_reference_c = temp_reference_c
        self.temp_sensitivity = temp_sensitivity

    def clone(self):
        return ModelParams(
            tire_model={
                k: TireParams(v.base_delta, v.deg_linear, v.deg_quadratic, v.age_temp_interaction)
                for k, v in self.tire_model.items()
            },
            temp_reference_c=self.temp_reference_c,
            temp_sensitivity=dict(self.temp_sensitivity),
        )

def default_model_params():
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

def lap_time_with_params(
    base_lap_time,
    tire,
    tire_age,
    track_temp,
    params,
):
    p = params.tire_model[tire]
    temp_delta = track_temp - params.temp_reference_c
    temp_effect = params.temp_sensitivity[tire] * temp_delta
    degradation = p.deg_linear * tire_age + p.deg_quadratic * (tire_age * tire_age)
    interaction = p.age_temp_interaction * tire_age * temp_delta
    return base_lap_time + p.base_delta + degradation + temp_effect + interaction

def simulate_driver_with_params(race_config, strategy, params):
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
        total_time += lap_time_with_params(
            base_lap_time, current_tire, tire_age, track_temp, params
        )

        if lap in pit_map:
            total_time += pit_lane_time
            current_tire = pit_map[lap]
            tire_age = 0

    return total_time

def simulate_race_with_params(race_config, strategies, params):
    totals = []

    for key in sorted(strategies.keys(), key=lambda s: int(s[3:])):
        strategy = strategies[key]
        driver_id = strategy["driver_id"]
        race_time = simulate_driver_with_params(race_config, strategy, params)
        totals.append((driver_id, race_time))

    totals.sort(key=lambda item: (item[1], item[0]))
    return [driver_id for driver_id, _ in totals]


def rank_score(predicted, expected):
    # Blend pairwise ordering + Spearman + exact-match bonus.
    n = len(expected)
    pred_pos = {driver: i for i, driver in enumerate(predicted)}
    exp_pos = {driver: i for i, driver in enumerate(expected)}

    total_pairs = n * (n - 1) // 2
    pairwise_correct = 0
    for i in range(n):
        di = expected[i]
        for j in range(i + 1, n):
            dj = expected[j]
            if pred_pos[di] < pred_pos[dj]:
                pairwise_correct += 1
    pairwise = pairwise_correct / total_pairs

    ssd = 0
    for driver in exp_pos:
        ssd += (pred_pos[driver] - exp_pos[driver]) ** 2

    rho = 1.0 - (6.0 * ssd) / (n * (n * n - 1))
    bonus = 0.15 if predicted == expected else 0.0
    return 0.65 * pairwise + 0.35 * ((rho + 1.0) / 2.0) + bonus


def evaluate(races, params):
    total = 0.0
    for race in races:
        predicted = simulate_race_with_params(race["race_config"], race["strategies"], params)
        total += rank_score(predicted, race["finishing_positions"])
    return total / max(len(races), 1)


def clamp_params(params):
    clamped = params.clone()

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    for tire in ("SOFT", "MEDIUM", "HARD"):
        p = clamped.tire_model[tire]
        clamped.tire_model[tire] = TireParams(
            base_delta=clamp(p.base_delta, -3.0, 3.0),
            deg_linear=clamp(p.deg_linear, 0.0, 0.20),
            deg_quadratic=clamp(p.deg_quadratic, 0.0, 0.01),
            age_temp_interaction=clamp(p.age_temp_interaction, -0.02, 0.02),
        )
        clamped.temp_sensitivity[tire] = clamp(clamped.temp_sensitivity[tire], -0.05, 0.05)

    clamped.temp_reference_c = clamp(clamped.temp_reference_c, 0.0, 60.0)
    return clamped


def mutate(params, rng, scale):
    p = params.clone()

    tire = rng.choice(["SOFT", "MEDIUM", "HARD"])
    what = rng.choice(["base", "lin", "quad", "cross", "temp", "temp_ref"])

    if what == "temp_ref":
        p.temp_reference_c += rng.gauss(0.0, 2.0 * scale)
    elif what == "temp":
        p.temp_sensitivity[tire] += rng.gauss(0.0, 0.004 * scale)
    else:
        tp = p.tire_model[tire]
        if what == "base":
            tp = TireParams(
                tp.base_delta + rng.gauss(0.0, 0.25 * scale),
                tp.deg_linear,
                tp.deg_quadratic,
                tp.age_temp_interaction,
            )
        elif what == "lin":
            tp = TireParams(
                tp.base_delta,
                tp.deg_linear + rng.gauss(0.0, 0.010 * scale),
                tp.deg_quadratic,
                tp.age_temp_interaction,
            )
        elif what == "cross":
            tp = TireParams(
                tp.base_delta,
                tp.deg_linear,
                tp.deg_quadratic,
                tp.age_temp_interaction + rng.gauss(0.0, 0.0015 * scale),
            )
        else:
            tp = TireParams(
                tp.base_delta,
                tp.deg_linear,
                tp.deg_quadratic + rng.gauss(0.0, 0.00030 * scale),
                tp.age_temp_interaction,
            )
        p.tire_model[tire] = tp

    return clamp_params(p)


def reservoir_sample_historical(rng, sample_size, data_glob):
    selected = []
    seen = 0

    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No historical files found with pattern: {data_glob}")

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            races = json.load(f)

        for race in races:
            seen += 1
            if len(selected) < sample_size:
                selected.append(race)
            else:
                j = rng.randrange(seen)
                if j < sample_size:
                    selected[j] = race

    return selected


def optimize(races, initial, iterations, seed):
    rng = random.Random(seed)
    current = clamp_params(initial)
    current_score = evaluate(races, current)

    best = current.clone()
    best_score = current_score

    for i in range(1, iterations + 1):
        progress = i / max(iterations, 1)
        scale = 1.5 - 1.2 * progress
        candidate = mutate(current, rng, scale)
        candidate_score = evaluate(races, candidate)

        if candidate_score >= current_score:
            current = candidate
            current_score = candidate_score
        else:
            temp = max(0.001, 0.04 * (1.0 - progress))
            accept_prob = math.exp((candidate_score - current_score) / temp)
            if rng.random() < accept_prob:
                current = candidate
                current_score = candidate_score

        if current_score > best_score:
            best = current.clone()
            best_score = current_score

        if i % 100 == 0 or i == iterations:
            print(f"iter={i:5d} current={current_score:.6f} best={best_score:.6f}")

    return best, best_score


def save_calibration_results(best, score, samples, path_json, path_log):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the JSON structure
    out_dict = {
        "TIMESTAMP": now_str,
        "BEST_SCORE": score,
        "SAMPLES_USED": samples,
        "TIRE_MODEL": {},
        "TEMP_REFERENCE_C": best.temp_reference_c,
        "TEMP_SENSITIVITY": dict(best.temp_sensitivity)
    }
    
    for k, v in best.tire_model.items():
        out_dict["TIRE_MODEL"][k] = {
            "base_delta": v.base_delta,
            "deg_linear": v.deg_linear,
            "deg_quadratic": v.deg_quadratic,
            "age_temp_interaction": v.age_temp_interaction
        }
        
    # Write JSON
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=4)
        
    # Append to log
    log_entry = (
        f"Calibration Run: {now_str}\n"
        f"Samples used: {samples}\n"
        f"Best score achieved: {score:.6f}\n"
        f"Saved to: {path_json.name}\n"
        f"----------------------------------------------------\n"
    )
    with open(path_log, "a", encoding="utf-8") as f:
        f.write(log_entry)


parser = argparse.ArgumentParser(description="Calibrate race simulator coefficients")
parser.add_argument("--sample-size", type=int, default=800, help="Number of historical races to sample")
parser.add_argument("--iterations", type=int, default=2000, help="Optimization iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--dry-run", action="store_true", help="Do not save files")
parser.add_argument(
    "--data-glob",
    default="data/historical_races/races_*.json",
    help="Glob pattern for historical race files (from repo root)",
)
args = parser.parse_args(sys.argv[1:])

root = Path(__file__).resolve().parent.parent
json_path = Path(__file__).resolve().parent / "model_parameters.json"
log_path = Path(__file__).resolve().parent / "calibration_log.txt"

print("Sampling historical races...")
rng = random.Random(args.seed)
races = reservoir_sample_historical(
    rng=rng,
    sample_size=args.sample_size,
    data_glob=str(root / args.data_glob),
)
print(f"Sampled races: {len(races)}")

initial = default_model_params()
baseline_score = evaluate(races, initial)
print(f"Baseline score: {baseline_score:.6f}")

best, best_score = optimize(
    races=races,
    initial=initial,
    iterations=args.iterations,
    seed=args.seed,
)
print(f"Best score: {best_score:.6f}")

if args.dry_run:
    print("\nDry run enabled: no files saved.")
else:
    save_calibration_results(best, best_score, len(races), json_path, log_path)
    print(f"\nSaved updated parameters to: {json_path}")
    print(f"Appended log to: {log_path}")
