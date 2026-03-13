# F1 Race Simulator Execution Guide

This guide explains how to properly run the Race Simulator and its components to predict race outcomes and calibrate the mathematical model.

## 1. Running the Simulator (Prediction Mode)

The simulator (`solution/race_simulator.py`) calculates the total completion time for each lap by analyzing the tire models, pit stops, degradation coefficients, track temperature, and mathematical interactions in a linear-quadratic fashion to determine the exact finishing order.

### To predict a single race:

You can pass a single JSON test case into the script via standard input. It will output the predicted finishing positions in JSON format.

**Using PowerShell (Windows):**
```powershell
Get-Content data\test_cases\inputs\test_001.json | python solution\race_simulator.py
```

**Using Bash / Git Bash:**
```bash
python solution/race_simulator.py < data/test_cases/inputs/test_001.json
```

---

## 2. Running the Full Test Suite

The repository includes a bash script to test your solution against all 100 benchmark test cases natively to evaluate accuracy.

Make sure you have Git Bash installed (Windows) or are using a standard Linux/Mac terminal. 

```bash
bash test_runner.sh
```
*Note: This script automatically reads the command defined in `solution/run_command.txt` and uses it to perform tests sequentially.*

---

## 3. Calibrating the Model (Training Mode)

Our theoretical model contains default baseline limits, but exact tire degradation scales vary depending on the track. The `solution/calibrate_model.py` script automatically parses the historical data provided (`data/historical_races`) to reverse-engineer these exact constants using simulated annealing.

### To run a full calibration cycle:

```powershell
python solution\calibrate_model.py
```

### To run a smaller calibration cycle (for faster testing):

```powershell
python solution\calibrate_model.py --iterations 500 --sample-size 400
```

### What happens when you run the calibration:

1. It intelligently loads historical JSONs up to the prescribed sample size.
2. The optimizer mathematically mutates and searches for the optimal variables corresponding to tire age and track temperatures against the documented race history.
3. Upon completion, it dynamically updates two new files:
    - **`solution/model_parameters.json`**: Saves the newfound mathematical configuration. The main predictor natively uses these updated constants if available.
    - **`solution/calibration_log.txt`**: Appends historical logs detailing exactly what the script scored and timestamps of when it happened.
