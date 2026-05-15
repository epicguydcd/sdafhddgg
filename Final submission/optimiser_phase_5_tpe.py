import optuna
import time
import json
from actualBaselineBot import run_bot

# ======================================================================
# FOCUSED PHASE 5 — Tight bounds, ~18 params, track zones included
# 
# The 4-phase run got us to 81.648s but Phase 4 (45 params) stalled.
# This focused run uses tight bounds around the Phase 3 best and
# specifically targets the track zones which were NEVER tuned.
# ======================================================================

# Current best from Phases 1-3 (81.648s)
BEST_PARAMS = {
    'STEER_KP': 9.377630658922872,
    'STEER_KD': 6.118343617748074,
    'CENTERING_GAIN': 0.026278308118437656,
    'BASE_SPEED_MULT': 6.474152947715284,
    'DISTANCE_SCALER': 13.82164300327937,
    'CORNER_EXIT_PUNCH': 64.64075532272774,
    'MID_CORNER_PENALTY': 6.150272999167717,
    'WALL_FEAR_MULT': 3.2751166162632352,
    'LOOKAHEAD_BLEND_THRESHOLD': 0.35154406487629886,
    'LOOKAHEAD_WEIGHT': 0.8531047146811941,
    'BRAKE_KP': 0.15605010408964332,
    'BRAKE_KD': 0.006388142485283601,
    'BRAKE_TRIGGER_MARGIN': 4.70761478681899,
    'TRAIL_BRAKE_FACTOR': 0.2118580015594262,
    'TRAIL_BRAKE_THROTTLE': 0.02922251950867933,
    'PARTIAL_THROTTLE_MARGIN': 8.9744913788975,
    'PARTIAL_THROTTLE_VALUE': 0.3456381761963969,
    'UPSHIFT_RPM': 18292,
    'DOWNSHIFT_RPM': 12168,
    'SHIFT_SLIP_DETECTION': 1.2137441115484418,
    'SHIFT_WAIT_STEPS': 7,
    'APEX_ACTIVATION_DIST': 83.68514299602667,
    'TRACK_EDGE_DIFF_TRIGGER': 4.8527564193828185,
    'APEX_BIAS_CAP': 0.5550670658104218,
    'APEX_DIVISOR': 127.25517515739612,
    'HIGH_SPEED_STEER_DAMP_SPEED': 98.80593407981057,
    'HIGH_SPEED_STEER_DAMP_BASE': 1.5319476290747356,
    'HIGH_SPEED_STEER_DAMP_DIVISOR': 3.614048403481607,
    # Zone params — THESE WERE NEVER TUNED, using original defaults
    'RAINEY_START': 2271.86,
    'RAINEY_END': 2401.47,
    'RAINEY_SPEED_LIMIT': 171.64,
    'RAINEY_TURN_IN_POS': -0.77,
    'CORKSCREW_TURNIN_KP': 3.5,
    'CORKSCREW_TURNIN_KD': 0.5,
    'TURN_11_START': 3164.74,
    'TURN_11_END': 3250.66,
    'TURN_11_SPEED_LIMIT': 120.93,
    # TCS/ABS (tuned in Phase 3)
    'ABS_SLIP_THRESHOLD': 2.633923660948679,
    'ABS_KP': 0.16684576340101862,
    'ABS_KI': 0.02320510063445782,
    'ABS_KD': 0.09619036105559349,
    'ABS_INTEGRAL_CAP': 13.015814010987384,
    'TCS_THRESHOLD': 16.84427138318147,
    'TCS_KP': 0.02729977440616593,
    'TCS_KI': 0.030819251330101004,
    'TCS_KD': 0.002826807961750267,
    'TCS_INTEGRAL_CAP': 4.084129194822758,
    'TCS_INTEGRAL_DECAY': 0.9598322013488279,
}


def objective(trial):
    p = dict(BEST_PARAMS)  # Start from current best, only override explored params
    
    # --- Core speed (tight bounds ±20% around best) ---
    p['STEER_KP'] = trial.suggest_float('STEER_KP', 7.0, 12.0)
    p['STEER_KD'] = trial.suggest_float('STEER_KD', 4.0, 9.0)
    p['BASE_SPEED_MULT'] = trial.suggest_float('BASE_SPEED_MULT', 5.5, 7.5)
    p['DISTANCE_SCALER'] = trial.suggest_float('DISTANCE_SCALER', 11.0, 17.0)
    p['CORNER_EXIT_PUNCH'] = trial.suggest_float('CORNER_EXIT_PUNCH', 50.0, 70.0)
    
    # --- Braking + trail-braking ---
    p['BRAKE_KP'] = trial.suggest_float('BRAKE_KP', 0.10, 0.22)
    p['BRAKE_TRIGGER_MARGIN'] = trial.suggest_float('BRAKE_TRIGGER_MARGIN', 2.0, 7.0)
    p['TRAIL_BRAKE_FACTOR'] = trial.suggest_float('TRAIL_BRAKE_FACTOR', 0.10, 0.35)
    p['TRAIL_BRAKE_THROTTLE'] = trial.suggest_float('TRAIL_BRAKE_THROTTLE', 0.0, 0.15)
    
    # --- Look-ahead ---
    p['LOOKAHEAD_BLEND_THRESHOLD'] = trial.suggest_float('LOOKAHEAD_BLEND_THRESHOLD', 0.2, 0.6)
    p['LOOKAHEAD_WEIGHT'] = trial.suggest_float('LOOKAHEAD_WEIGHT', 0.7, 0.95)
    
    # --- TRACK ZONES (NEVER TUNED — widest exploration) ---
    # Corkscrew zone
    p['RAINEY_START'] = trial.suggest_float('RAINEY_START', 2220.0, 2320.0)
    p['RAINEY_END'] = trial.suggest_float('RAINEY_END', 2370.0, 2450.0)
    p['RAINEY_SPEED_LIMIT'] = trial.suggest_float('RAINEY_SPEED_LIMIT', 140.0, 190.0)
    p['RAINEY_TURN_IN_POS'] = trial.suggest_float('RAINEY_TURN_IN_POS', -0.95, -0.5)
    p['CORKSCREW_TURNIN_KP'] = trial.suggest_float('CORKSCREW_TURNIN_KP', 1.0, 7.0)
    p['CORKSCREW_TURNIN_KD'] = trial.suggest_float('CORKSCREW_TURNIN_KD', 0.1, 2.5)
    
    # Turn 11 zone
    p['TURN_11_START'] = trial.suggest_float('TURN_11_START', 3120.0, 3210.0)
    p['TURN_11_END'] = trial.suggest_float('TURN_11_END', 3220.0, 3300.0)
    p['TURN_11_SPEED_LIMIT'] = trial.suggest_float('TURN_11_SPEED_LIMIT', 100.0, 145.0)

    try:
        current_best = study.best_value
    except ValueError:
        current_best = 10000.0

    try:
        lap_time = run_bot(p, current_best)
    except Exception as e:
        print(f"  Trial failed: {e}")
        lap_time = 10000.0

    time.sleep(1.0)
    return lap_time


if __name__ == "__main__":
    print("=" * 60)
    print("  FOCUSED PHASE 5 — Track Zones + Core Speed")
    print("  Baseline: 81.648s | Target: sub-80s")
    print("  Params: 20 (tight bounds) | Trials: 300")
    print("=" * 60)

    study = optuna.create_study(
        direction='minimize',
        study_name="torcs_phase5_focused",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15)
    )

    # Enqueue current best as trial 0
    baseline_trial = {
        'STEER_KP': 9.377630658922872,
        'STEER_KD': 6.118343617748074,
        'BASE_SPEED_MULT': 6.474152947715284,
        'DISTANCE_SCALER': 13.82164300327937,
        'CORNER_EXIT_PUNCH': 64.64075532272774,
        'BRAKE_KP': 0.15605010408964332,
        'BRAKE_TRIGGER_MARGIN': 4.70761478681899,
        'TRAIL_BRAKE_FACTOR': 0.2118580015594262,
        'TRAIL_BRAKE_THROTTLE': 0.02922251950867933,
        'LOOKAHEAD_BLEND_THRESHOLD': 0.35154406487629886,
        'LOOKAHEAD_WEIGHT': 0.8531047146811941,
        'RAINEY_START': 2271.86,
        'RAINEY_END': 2401.47,
        'RAINEY_SPEED_LIMIT': 171.64,
        'RAINEY_TURN_IN_POS': -0.77,
        'CORKSCREW_TURNIN_KP': 3.5,
        'CORKSCREW_TURNIN_KD': 0.5,
        'TURN_11_START': 3164.74,
        'TURN_11_END': 3250.66,
        'TURN_11_SPEED_LIMIT': 120.93,
    }
    study.enqueue_trial(baseline_trial)

    study.optimize(objective, n_trials=300)

    print("\n" + "=" * 60)
    print("  PHASE 5 COMPLETE!")
    print(f"  Best Lap Time: {study.best_value:.3f}s")
    print("=" * 60)

    # Build complete param set with best trial's values
    final_params = dict(BEST_PARAMS)
    final_params.update(study.best_params)
    
    with open('phase5_best.json', 'w') as f:
        json.dump({'params': final_params, 'time': study.best_value}, f, indent=2)

    print(f"\nBest params saved to: phase5_best.json")
    print("\n# --- Copy these into bot.py ---")
    for k, v in sorted(final_params.items()):
        print(f"{k} = {v}")
