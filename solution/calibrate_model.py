#!/usr/bin/env python3
# Model calibration
# Offline optimizer
import json
import glob
import os
import random
import datetime
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution

# Vector layout
# Core params
# Tire params

BOUNDS = [
    (-0.001, 0.001),   # fuel_effect
    (0.05,   0.20),    # temp_coef
    (-0.005, 0.005),   # temp_base_coef
    # SOFT
    (1.0,  5.0),   (5.0,  15.0),  (0.1,  0.8),  (0.0, 0.05),
    # MEDIUM
    (2.0,  7.0),   (10.0, 30.0),  (0.05, 0.5),  (0.0, 0.02),
    # HARD
    (3.0,  8.0),   (15.0, 45.0),  (0.01, 0.3),  (0.0, 0.01),
]

# Initial guess
X0 = np.array([
    0.0, 0.112095, 0.0,
    2.958962, 10, 0.393805, 0.0,
    3.926250, 20, 0.200598, 0.0,
    4.724486, 30, 0.103190, 0.0,
])


def vec_to_params(vec):
    # Map vector
    return {
        "fuel_effect":    vec[0],
        "temp_coef":      vec[1],
        "temp_base_coef": vec[2],
        "SOFT":   {"offset": vec[3],  "cliff": vec[4],  "deg": vec[5],  "deg2": vec[6]},
        "MEDIUM": {"offset": vec[7],  "cliff": vec[8],  "deg": vec[9],  "deg2": vec[10]},
        "HARD":   {"offset": vec[11], "cliff": vec[12], "deg": vec[13], "deg2": vec[14]},
    }


def simulate_driver_fast(race_config, strategy, params):
    # Fast simulator
    total_laps   = race_config["total_laps"]
    base_time    = race_config["base_lap_time"]
    pit_time     = race_config["pit_lane_time"]
    temp         = race_config["track_temp"]
    current_tire = strategy["starting_tire"]
    pit_map      = {int(s["lap"]): s["to_tire"] for s in strategy.get("pit_stops", [])}

    temp_deg_mult = 1.0 + temp * params["temp_coef"]
    temp_base_eff = temp * params["temp_base_coef"]
    fuel_eff      = params["fuel_effect"]

    tire_age   = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        tp    = params[current_tire]
        cliff = int(round(tp["cliff"]))

        lap_t = base_time + tp["offset"] + temp_base_eff - fuel_eff * lap

        beyond = tire_age - cliff
        if beyond > 0:
            lap_t += beyond * tp["deg"] * temp_deg_mult
            lap_t += beyond * beyond * tp["deg2"] * temp_deg_mult

        total_time += lap_t

        if lap in pit_map:
            total_time  += pit_time
            current_tire = pit_map[lap]
            tire_age     = 0

    return total_time


def evaluate(races, params):
    # Rank score
    total_score = 0.0
    for race in races:
        driver_times = []
        for strat in race["strategies"].values():
            t = simulate_driver_fast(race["race_config"], strat, params)
            driver_times.append((strat["driver_id"], t))
        driver_times.sort(key=lambda x: (x[1], x[0]))
        predicted = [d for d, _ in driver_times]
        expected  = race["finishing_positions"]

        n = len(expected)
        pred_pos = {d: i for i, d in enumerate(predicted)}
        exp_pos  = {d: i for i, d in enumerate(expected)}
        ssd = sum((pred_pos[d] - exp_pos[d]) ** 2 for d in exp_pos)
        rho = 1.0 - (6.0 * ssd) / (n * (n * n - 1))

        exact_bonus = 0.30 if predicted == expected else 0.0
        total_score += (rho + 1.0) / 2.0 + exact_bonus

    return total_score / len(races) if races else 0.0


def objective(vec, races):
    # Negate score
    return -evaluate(races, vec_to_params(vec))


def load_races(num_samples):
    all_races = []
    for f_path in sorted(glob.glob("data/historical_races/races_*.json")):
        with open(f_path) as f:
            all_races.extend(json.load(f))
        if len(all_races) >= num_samples * 3:
            break
    random.shuffle(all_races)
    return all_races[:num_samples]


def evaluate_test_cases(params):
    # Public accuracy
    correct = 0
    total   = 0
    for inp in sorted(glob.glob("data/test_cases/inputs/test_*.json")):
        test_id  = os.path.basename(inp).replace(".json", "")
        exp_path = f"data/test_cases/expected_outputs/{test_id}.json"
        if not os.path.exists(exp_path):
            continue
        with open(inp) as f:  tc  = json.load(f)
        with open(exp_path) as f: exp = json.load(f)

        driver_times = []
        for strat in tc["strategies"].values():
            t = simulate_driver_fast(tc["race_config"], strat, params)
            driver_times.append((strat["driver_id"], t))
        driver_times.sort(key=lambda x: (x[1], x[0]))

        if [d for d, _ in driver_times] == exp["finishing_positions"]:
            correct += 1
        total += 1
    return correct, total


def save_params(params, score, path, is_checkpoint=False):
    out = {
        "TIMESTAMP":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "BEST_SCORE": score,
        "PARAMS": {
            "SOFT":           params["SOFT"],
            "MEDIUM":         params["MEDIUM"],
            "HARD":           params["HARD"],
            "temp_coef":      params["temp_coef"],
            "fuel_effect":    params["fuel_effect"],
            "temp_base_coef": params["temp_base_coef"],
        }
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=4)
    label = "checkpoint" if is_checkpoint else "final"
    print(f"[{label}] Saved to {path}  (score={score:.6f})")


def load_test_cases():
    # Load public tests
    cases = []
    for inp in sorted(glob.glob("data/test_cases/inputs/test_*.json")):
        test_id  = os.path.basename(inp).replace(".json", "")
        exp_path = f"data/test_cases/expected_outputs/{test_id}.json"
        if not os.path.exists(exp_path): continue
        with open(inp) as f:  tc  = json.load(f)
        with open(exp_path) as f: exp = json.load(f)
        cases.append({"race_config": tc["race_config"], "strategies": tc["strategies"], "finishing_positions": exp["finishing_positions"]})
    return cases

def main():
    parser = argparse.ArgumentParser(description="Calibrate race_simulator.py params")
    parser.add_argument("--iter", type=int, default=100, help="DE max generations")
    args, _ = parser.parse_known_args()

    print(f"Loading public test cases for optimization...")
    races = load_test_cases()
    print(f"Loaded {len(races)} tests. Starting optimization...\n")

    gen_count = [0]

    def callback(xk, convergence):
        gen_count[0] += 1
        score = -objective(xk, races)
        print(f"Gen {gen_count[0]:3d}: score={score:.6f}  conv={convergence:.4f}")

        # Ten-gen checkpoint
        if gen_count[0] % 10 == 0:
            correct, total = evaluate_test_cases(vec_to_params(xk))
            print(f"  --> Test accuracy: {correct}/{total} = {correct/total*100:.1f}%")
            save_params(vec_to_params(xk), score,
                        Path("solution/model_parameters_checkpoint.json"),
                        is_checkpoint=True)

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        args=(races,),
        popsize=10,
        maxiter=args.iter,
        disp=True,
        polish=True,
        init="latinhypercube",
        callback=callback,
        seed=42,
        x0=X0,
    )

    best_params = vec_to_params(result.x)
    train_score = -result.fun

    print(f"\n=== Done ===")
    print(f"Training score : {train_score:.6f}")

    correct, total = evaluate_test_cases(best_params)
    print(f"Test accuracy  : {correct}/{total} = {correct/total*100:.1f}%")

    save_params(best_params, train_score, Path("solution/model_parameters.json"))


if __name__ == "__main__":
    main()
