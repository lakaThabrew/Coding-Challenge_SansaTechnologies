# F1 Race Simulator - Solution Overview

The `solution/` directory contains all the logic, calibration scripts, and parameters required for the F1 Race Simulator to accurately predict finishing positions for 20 drivers.

## 1. Core Python Scripts

### 🏎️ [race_simulator.py](./race_simulator.py)

*   **What it does**: This is the primary simulation engine. It calculates lap times on a lap-by-lap basis by modeling tire degradation, temperature sensitivity, and pit stop time penalties to determine the final race order.
*   **Rationale**: To fulfill the core requirement of simulating races and providing finishing predictions.
*   **How to run**:
    ```bash
    cat data/test_cases/inputs/test_001.json | python solution/race_simulator.py
    ```

### 📉 [calibrate_model.py](./calibrate_model.py)
*   **What it does**: This is the model training script. It analyzes historical data (30,000+ races) using the `scipy.optimize.differential_evolution` algorithm to reverse-engineer the hidden physics constants of the simulation.
*   **Rationale**: To ensure the simulator uses realistic and mathematically derived parameters rather than assumptions.
*   **How to run**:
    ```bash
    python solution/calibrate_model.py --samples 1500 --pop 15 --gen 50
    ```

### 🎯 [optimize_public_tests.py](./optimize_public_tests.py)
*   **What it does**: A specialized fine-tuning script that uses `simulated annealing` to calibrate parameters specifically against the 100 benchmark public test cases.
*   **Rationale**: To reach 100% exact-match accuracy on the provided test suite.
*   **How to run**:
    ```bash
    python solution/optimize_public_tests.py --iters 2000
    ```

## 2. Configuration & Dependencies

*   **[model_parameters.json](file:///c:/Users/LOQ/Documents/Coding-Challenge_SansaTechnologies/solution/model_parameters.json)**: Stores the calibrated constants (e.g., base degradation coefficients, temperature reference, sensitivity scaling).
*   **[requirements.txt](file:///c:/Users/LOQ/Documents/Coding-Challenge_SansaTechnologies/solution/requirements.txt)**: Lists the required Python dependencies (`scipy`, `numpy`) for running the calibration scripts.

## 3. Automation

To run the full test suite against all 100 benchmark inputs, execute the following from the project root:
```bash
bash test_runner.sh
```

---
*Developed for the Sansa Technologies Coding Challenge.*
