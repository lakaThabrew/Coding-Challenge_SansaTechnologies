#!/usr/bin/env python3
"""
Advanced Calibration Script for Box Box Box F1 Simulator.
Uses Differential Evolution and Cross-Validation for maximum accuracy.
"""

import argparse
import glob
import json
import math
import random
import datetime
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution
from collections import defaultdict

from race_simulator import (
    TireParams,
    TIRE_MODEL,
    TEMP_REFERENCE_C,
    TEMP_SENSITIVITY,
    build_pit_map,
)

# --------------------------------------------------------------------------------
# Model Structure
# --------------------------------------------------------------------------------

class ModelParams:
    def __init__(self, tire_model, temp_reference_c, temp_sensitivity):
        self.tire_model = tire_model
        self.temp_reference_c = temp_reference_c
        self.temp_sensitivity = temp_sensitivity

    def to_vector(self):
        """Convert params to a flat vector for the optimizer."""
        vec = [self.temp_reference_c]
        for tire in ["SOFT", "MEDIUM", "HARD"]:
            p = self.tire_model[tire]
            vec.extend([
                p.base_delta,
                p.deg_linear,
                p.deg_quadratic,
                p.age_temp_interaction,
                p.threshold,
                self.temp_sensitivity[tire]
            ])
        return np.array(vec)

    @staticmethod
    def from_vector(vec):
        """Create params from a flat vector."""
        temp_ref = vec[0]
        tire_model = {}
        temp_sens = {}
        
        idx = 1
        for tire in ["SOFT", "MEDIUM", "HARD"]:
            base_delta = vec[idx]
            deg_linear = vec[idx+1]
            deg_quadratic = vec[idx+2]
            interaction = vec[idx+3]
            threshold = vec[idx+4]
            sensitivity = vec[idx+5]
            
            tire_model[tire] = TireParams(
                base_delta, deg_linear, deg_quadratic, interaction, threshold
            )
            temp_sens[tire] = sensitivity
            idx += 6
            
        return ModelParams(tire_model, temp_ref, temp_sens)

    @staticmethod
    def get_bounds():
        """Define search bounds for each parameter."""
        # [temp_ref, (base, lin, quad, cross, thresh, sens) x 3]
        bounds = [(10.0, 50.0)]  # Temp Ref
        
        # SOFT
        bounds.extend([
            (-2.0, 1.0),    # base_delta
            (0.0, 0.2),     # deg_linear
            (0.0, 0.01),    # deg_quadratic
            (-0.02, 0.02),  # interaction
            (0.0, 10.0),    # threshold (SOFT usually low)
            (-0.05, 0.05)   # sensitivity
        ])
        
        # MEDIUM
        bounds.extend([
            (-1.0, 1.5),    # base_delta
            (0.0, 0.15),    # deg_linear
            (0.0, 0.005),   # deg_quadratic
            (-0.01, 0.01),  # interaction
            (0.0, 20.0),    # threshold
            (-0.05, 0.05)   # sensitivity
        ])
        
        # HARD
        bounds.extend([
            (0.0, 3.0),     # base_delta
            (0.0, 0.1),     # deg_linear
            (0.0, 0.002),   # deg_quadratic
            (-0.005, 0.005),# interaction
            (0.0, 30.0),    # threshold
            (-0.05, 0.05)   # sensitivity
        ])
        
        return bounds

# --------------------------------------------------------------------------------
# Simulation (Optimized)
# --------------------------------------------------------------------------------

def simulate_driver_fast(race_config, strategy, params):
    total_laps = race_config["total_laps"]
    base_lap_time = race_config["base_lap_time"]
    pit_lane_time = race_config["pit_lane_time"]
    track_temp = race_config["track_temp"]

    current_tire = strategy["starting_tire"]
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda x: x["lap"])
    
    total_time = 0.0
    last_pit_lap = 0
    
    pit_laps = [s["lap"] for s in pit_stops] + [total_laps]
    next_tires = [s["to_tire"] for s in pit_stops]
    
    for i, end_lap in enumerate(pit_laps):
        stint_laps = end_lap - last_pit_lap
        if stint_laps > 0:
            p = params.tire_model[current_tire]
            temp_delta = track_temp - params.temp_reference_c
            temp_effect = params.temp_sensitivity[current_tire] * temp_delta
            
            # Fixed cost per stint
            total_time += stint_laps * (base_lap_time + p.base_delta + temp_effect)
            
            # Degradation starts after threshold laps
            K = max(0, stint_laps - p.threshold)
            if K > 0:
                # Sum of k from 1 to K
                sum_k = K * (K + 1) / 2
                # Sum of k^2 from 1 to K
                sum_k2 = K * (K + 1) * (2 * K + 1) / 6
                
                lin_part = p.deg_linear + p.age_temp_interaction * temp_delta
                total_time += sum_k * lin_part + sum_k2 * p.deg_quadratic
                
        if i < len(next_tires):
            total_time += pit_lane_time
            current_tire = next_tires[i]
            last_pit_lap = end_lap
            
    return total_time

def simulate_race_times_and_order(race_config, strategies, params):
    """Return predicted order and driver total times for a race."""
    driver_times = []
    for pos_key in strategies:
        strat = strategies[pos_key]
        race_time = simulate_driver_fast(race_config, strat, params)
        driver_times.append((strat["driver_id"], race_time, strat))

    driver_times.sort(key=lambda x: (x[1], x[0]))
    predicted = [d for d, _, _ in driver_times]
    time_map = {d: t for d, t, _ in driver_times}
    strat_map = {item["driver_id"]: item for item in strategies.values()}
    return predicted, time_map, strat_map

def evaluate_score(races, params, verbose=False):
    total = 0.0
    for race in races:
        race_config = race["race_config"]
        strategies = race["strategies"]
        expected = race["finishing_positions"]
        
        driver_times = []
        for pos_key in strategies:
            strat = strategies[pos_key]
            race_time = simulate_driver_fast(race_config, strat, params)
            driver_times.append((strat["driver_id"], race_time))
            
        driver_times.sort(key=lambda x: (x[1], x[0]))
        predicted = [d for d, _ in driver_times]
        
        n = len(expected)
        pred_pos = {d: i for i, d in enumerate(predicted)}
        exp_pos = {d: i for i, d in enumerate(expected)}
        
        ssd = sum((pred_pos[d] - exp_pos[d])**2 for d in exp_pos)
        rho = 1.0 - (6.0 * ssd) / (n * (n * n - 1))
        
        bonus = 0.15 if predicted == expected else 0.0
        total += (rho + 1.0) / 2.0 + bonus
    
    score = total / len(races) if races else 0
    if verbose:
        print(f"Avg Score: {score:.6f}")
    return score

def _temperature_weight(track_temp):
    # Emphasize temperature extremes so model generalizes at cold/hot tracks.
    if track_temp < 20.0 or track_temp >= 40.0:
        return 1.35
    return 1.0

def _pairwise_top10_score(expected, predicted):
    """Pairwise ordering accuracy among expected top-10 drivers."""
    top = expected[:10]
    pred_pos = {d: i for i, d in enumerate(predicted)}
    total_pairs = 0
    correct_pairs = 0

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            total_pairs += 1
            if pred_pos[top[i]] < pred_pos[top[j]]:
                correct_pairs += 1

    return (correct_pairs / total_pairs) if total_pairs else 0.0

def _crossover_accuracy(expected, time_map, strat_map):
    """
    Compare best SOFT->HARD vs MEDIUM->HARD one-stop archetypes.
    Returns 1.0 if model predicts same winner as expected order, else 0.0.
    """
    soft_candidates = []
    medium_candidates = []
    expected_pos = {d: i for i, d in enumerate(expected)}

    for driver_id, strat in strat_map.items():
        pit_stops = strat.get("pit_stops", [])
        if len(pit_stops) == 1 and pit_stops[0]["to_tire"] == "HARD":
            item = (driver_id, time_map[driver_id])
            if strat["starting_tire"] == "SOFT":
                soft_candidates.append(item)
            elif strat["starting_tire"] == "MEDIUM":
                medium_candidates.append(item)

    if not soft_candidates or not medium_candidates:
        return 0.5

    best_soft = min(soft_candidates, key=lambda x: x[1])[0]
    best_medium = min(medium_candidates, key=lambda x: x[1])[0]

    model_winner_soft = time_map[best_soft] < time_map[best_medium]
    expected_winner_soft = expected_pos[best_soft] < expected_pos[best_medium]
    return 1.0 if model_winner_soft == expected_winner_soft else 0.0

def _strategy_bias_penalty(expected, predicted, strat_map):
    """Penalize known systematic ranking bias by strategy archetype."""
    exp_pos = {d: i for i, d in enumerate(expected)}
    pred_pos = {d: i for i, d in enumerate(predicted)}

    soft_very_short = []
    medium_mid = []
    hard_early = []

    for driver_id, strat in strat_map.items():
        pit_stops = strat.get("pit_stops", [])
        first_stint = pit_stops[0]["lap"] if pit_stops else 999
        delta = pred_pos[driver_id] - exp_pos[driver_id]
        start = strat["starting_tire"]

        if start == "SOFT" and first_stint <= 7:
            # Negative delta means over-promoted; penalize only that direction.
            soft_very_short.append(max(0.0, -delta))
        elif start == "MEDIUM" and 11 <= first_stint <= 20:
            # Positive delta means under-promoted.
            medium_mid.append(max(0.0, delta))
        elif start == "HARD" and first_stint <= 15:
            # Positive delta means under-promoted.
            hard_early.append(max(0.0, delta))

    def avg(arr):
        return (sum(arr) / len(arr)) if arr else 0.0

    return avg(soft_very_short) + avg(medium_mid) + avg(hard_early)

def parameter_regularization(params):
    """Regularize towards physically plausible behavior from diagnostics."""
    reg = 0.0

    soft = params.tire_model["SOFT"]
    medium = params.tire_model["MEDIUM"]
    hard = params.tire_model["HARD"]

    # Reduce SOFT short-stint over-performance (encourage earlier degradation onset).
    reg += 0.0025 * max(0.0, soft.threshold - 6.0) ** 2

    # Flatten MEDIUM mid-stint penalty (discourage overly steep degradation terms).
    reg += 0.06 * max(0.0, medium.deg_linear - 0.09) ** 2
    reg += 12.0 * max(0.0, medium.deg_quadratic - 0.0025) ** 2

    # Rebalance HARD early stint penalty (avoid excessive baseline handicap).
    reg += 0.02 * max(0.0, hard.base_delta - 1.0) ** 2

    # Keep temperature sensitivity in a stable range to avoid overfitting.
    for tire in ["SOFT", "MEDIUM", "HARD"]:
        reg += 0.01 * max(0.0, abs(params.temp_sensitivity[tire]) - 0.04) ** 2

    return reg

def evaluate_objective_score(races, params, verbose=False):
    """
    Composite objective:
    - Base Spearman + exact-match bonus
    - Pairwise top-10 ordering score
    - SOFT-vs-MEDIUM crossover accuracy
    - Strategy-bias penalty from known failure modes
    - Temperature bucket weighting
    """
    weighted_total = 0.0
    total_weight = 0.0

    for race in races:
        race_config = race["race_config"]
        strategies = race["strategies"]
        expected = race["finishing_positions"]

        predicted, time_map, strat_map = simulate_race_times_and_order(race_config, strategies, params)

        n = len(expected)
        pred_pos = {d: i for i, d in enumerate(predicted)}
        exp_pos = {d: i for i, d in enumerate(expected)}

        ssd = sum((pred_pos[d] - exp_pos[d]) ** 2 for d in exp_pos)
        rho = 1.0 - (6.0 * ssd) / (n * (n * n - 1))
        base = (rho + 1.0) / 2.0
        bonus = 0.15 if predicted == expected else 0.0

        pairwise_top10 = _pairwise_top10_score(expected, predicted)
        crossover = _crossover_accuracy(expected, time_map, strat_map)
        bias_penalty = _strategy_bias_penalty(expected, predicted, strat_map)

        race_score = base + bonus + 0.22 * pairwise_top10 + 0.08 * crossover - 0.015 * bias_penalty

        w = _temperature_weight(float(race_config["track_temp"]))
        weighted_total += w * race_score
        total_weight += w

    score = (weighted_total / total_weight) if total_weight else 0.0
    score -= parameter_regularization(params)

    if verbose:
        print(f"Composite Score: {score:.6f}")
    return score

# --------------------------------------------------------------------------------
# Main Process
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--pop", type=int, default=15)
    parser.add_argument("--gen", type=int, default=50)
    args = parser.parse_args()

    print(f"[{datetime.datetime.now()}] Loading historical data...")
    data_files = glob.glob("data/historical_races/races_*.json")
    all_races = []
    # Load a few files to get enough samples
    for f_path in sorted(data_files):
        with open(f_path, "r") as f:
            all_races.extend(json.load(f))
        if len(all_races) >= args.samples * 2: break

    random.shuffle(all_races)
    train_races = all_races[:args.samples]
    val_races = all_races[args.samples:args.samples + 500]
    
    print(f"Training on {len(train_races)} races, validating on {len(val_races)} races.")

    def objective(vec):
        params = ModelParams.from_vector(vec)
        score = evaluate_objective_score(train_races, params)
        return 1.0 - (score / 1.45)

    print(f"[{datetime.datetime.now()}] Starting Differential Evolution...")
    
    def checkpoint_callback(xk, convergence):
        best_p = ModelParams.from_vector(xk)
        score = evaluate_objective_score(train_races, best_p)
        print(f"[{datetime.datetime.now()}] Checkpoint: Best Composite Score = {score:.6f} (conv={convergence:.4f})")
        # Save intermediate result
        save_calibration_results(best_p, score, len(train_races), Path("solution/model_parameters_checkpoint.json"), Path("solution/calibration_log.txt"), is_checkpoint=True)

    result = differential_evolution(
        objective,
        bounds=ModelParams.get_bounds(),
        popsize=args.pop,
        maxiter=args.gen,
        disp=True,
        polish=True,
        callback=checkpoint_callback,
        workers=1
    )
    
    best_params = ModelParams.from_vector(result.x)
    train_score = evaluate_objective_score(train_races, best_params)
    val_score = evaluate_objective_score(val_races, best_params)
    
    print(f"\nFinal Results:")
    print(f"Train Score: {train_score:.6f}")
    print(f"Val Score:   {val_score:.6f}")
    
    save_calibration_results(best_params, val_score, len(train_races), Path("solution/model_parameters.json"), Path("solution/calibration_log.txt"))
    
def save_calibration_results(best, score, samples, path_json, path_log, is_checkpoint=False):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    out_dict = {
        "TIMESTAMP": now_str,
        "BEST_SCORE": score,
        "SAMPLES_USED": samples,
        "TIRE_MODEL": {},
        "TEMP_REFERENCE_C": best.temp_reference_c,
        "TEMP_SENSITIVITY": best.temp_sensitivity
    }
    
    for k, v in best.tire_model.items():
        out_dict["TIRE_MODEL"][k] = {
            "base_delta": v.base_delta,
            "deg_linear": v.deg_linear,
            "deg_quadratic": v.deg_quadratic,
            "age_temp_interaction": v.age_temp_interaction,
            "threshold": int(v.threshold)
        }
    
    with open(path_json, "w") as f:
        json.dump(out_dict, f, indent=4)
        
    if not is_checkpoint:
        log_entry = (
            f"Calibration Run: {now_str}\n"
            f"Samples used: {samples}\n"
            f"Best score achieved: {score:.6f}\n"
            f"Saved to: {path_json.name}\n"
            f"----------------------------------------------------\n"
        )
        with open(path_log, "a") as f:
            f.write(log_entry)
        
    print(f"Saved optimized parameters to {path_json}")

if __name__ == "__main__":
    main()
