# F1 Race Simulator

Predicts F1 finishing positions given pit stop strategies, tire choices, and track conditions.

---

## Files

| File | Purpose |
|---|---|
| `race_simulator.py` | Main script — reads a race JSON from stdin, outputs finishing order |
| `calibrate_model.py` | Offline training — tunes model params using 30k historical races |
| `model_parameters.json` | Best parameters found by calibration (auto-loaded at runtime) |
| `model_parameters_checkpoint.json` | Saved mid-run checkpoint from calibration |
| `run_command.txt` | Tells the test runner how to invoke the solution |
| `requirements.txt` | Python dependencies |

---

## How to Run

**Single test case:**
```bash
cat data/test_cases/inputs/test_001.json | python solution/race_simulator.py
```

**Full test suite (100 cases):**
```bash
bash test_runner.sh
```

**Calibrate new parameters:**
```bash
python solution/calibrate_model.py --samples 2000 --pop 20 --gen 100
```

---

## How the Model Works

Each driver is simulated lap by lap. Their total race time is:

```
lap_time = base_lap_time
         + tire_offset          ← compound speed (SOFT fastest, HARD slowest)
         - fuel_saving           ← car gets lighter as fuel burns
         + degradation           ← kicks in after the cliff (grace laps)
```

**Degradation** only starts after a per-compound grace period (`cliff`):
- SOFT: ~10 laps before it goes off
- MEDIUM: ~20 laps
- HARD: ~30 laps

Track temperature also scales the degradation rate — hotter = faster wear.

After all laps, drivers are sorted by total time. Ties broken by driver ID.

---

## Calibration

`calibrate_model.py` runs **Differential Evolution** (scipy) over 3000 historical races to find the best values for all tire parameters. It prints test case accuracy every 10 generations and saves checkpoints automatically.

---

*Built for the Sansa Technologies F1 Coding Challenge.*
