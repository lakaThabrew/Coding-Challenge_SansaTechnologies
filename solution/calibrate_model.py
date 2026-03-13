#!/usr/bin/env python3
"""
Stand-alone Calibration script.
This runs the optimization algorithm to find the best constants and saves them
to `model_parameters.json` and records a log in `calibration_log.txt`
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import datetime
from pathlib import Path
from typing import List, Tuple

# Import the base simulation structures from our main app
from race_simulator import (
    ModelParams,
    TireParams,
    _default_model_params,
    _simulate_race_with_params,
)

def _rank_score(predicted: List[str], expected: List[str]) -> float:
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


def _evaluate(races: List[dict], params: ModelParams) -> float:
    total = 0.0
    for race in races:
        predicted = _simulate_race_with_params(race["race_config"], race["strategies"], params)
        total += _rank_score(predicted, race["finishing_positions"])
    return total / max(len(races), 1)


def _clamp_params(params: ModelParams) -> ModelParams:
    clamped = params.clone()

    def clamp(v: float, lo: float, hi: float) -> float:
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


def _mutate(params: ModelParams, rng: random.Random, scale: float) -> ModelParams:
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

    return _clamp_params(p)


def _reservoir_sample_historical(
    rng: random.Random, sample_size: int, data_glob: str
) -> List[dict]:
    selected: List[dict] = []
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


def _optimize(
    races: List[dict], initial: ModelParams, iterations: int, seed: int
) -> Tuple[ModelParams, float]:
    rng = random.Random(seed)
    current = _clamp_params(initial)
    current_score = _evaluate(races, current)

    best = current.clone()
    best_score = current_score

    for i in range(1, iterations + 1):
        progress = i / max(iterations, 1)
        scale = 1.5 - 1.2 * progress
        candidate = _mutate(current, rng, scale)
        candidate_score = _evaluate(races, candidate)

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


def save_calibration_results(best: ModelParams, score: float, samples: int, path_json: Path, path_log: Path):
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
        f"----------------------------------------------------\n"
        f"Calibration Run: {now_str}\n"
        f"Samples used: {samples}\n"
        f"Best score achieved: {score:.6f}\n"
        f"Saved to: {path_json.name}\n"
        f"----------------------------------------------------\n"
    )
    with open(path_log, "a", encoding="utf-8") as f:
        f.write(log_entry)


def main(argv: List[str] | None = None) -> None:
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
    import sys
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    root = Path(__file__).resolve().parent.parent
    json_path = Path(__file__).resolve().parent / "model_parameters.json"
    log_path = Path(__file__).resolve().parent / "calibration_log.txt"

    print("Sampling historical races...")
    rng = random.Random(args.seed)
    races = _reservoir_sample_historical(
        rng=rng,
        sample_size=args.sample_size,
        data_glob=str(root / args.data_glob),
    )
    print(f"Sampled races: {len(races)}")

    initial = _default_model_params()
    baseline_score = _evaluate(races, initial)
    print(f"Baseline score: {baseline_score:.6f}")

    best, best_score = _optimize(
        races=races,
        initial=initial,
        iterations=args.iterations,
        seed=args.seed,
    )
    print(f"Best score: {best_score:.6f}")

    if args.dry_run:
        print("\nDry run enabled: no files saved.")
        return

    save_calibration_results(best, best_score, len(races), json_path, log_path)
    print(f"\nSaved updated parameters to: {json_path}")
    print(f"Appended log to: {log_path}")

if __name__ == "__main__":
    main()
