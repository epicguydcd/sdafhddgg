import optuna
import time
import json
from actualBaselineBot import run_bot

# ======================================================================
# PHASE 6 — Optimize new features: Racing Line + Engine Braking + Speed Penalty
# Plus re-tune core speed/braking params with tight bounds
# ======================================================================

BEST_PARAMS = {
    'STEER_KP': 7.5500778899557766, 'STEER_KD': 8.661765678765278,
    'CENTERING_GAIN': 0.026278308118437656,
    'BASE_SPEED_MULT': 6.111111653883914, 'DISTANCE_SCALER': 15.515669038337016,
    'CORNER_EXIT_PUNCH': 51.921844937822, 'MID_CORNER_PENALTY': 6.150272999167717,
    'WALL_FEAR_MULT': 3.2751166162632352,
    'LOOKAHEAD_BLEND_THRESHOLD': 0.42320248198089244,
    'LOOKAHEAD_WEIGHT': 0.9315238778576724,
    'SPEED_PENALTY_REFERENCE': 150.0,
    'BRAKE_KP': 0.18882196845205348, 'BRAKE_KD': 0.006388142485283601,
    'BRAKE_TRIGGER_MARGIN': 5.226901691885811,
    'TRAIL_BRAKE_FACTOR': 0.2520526941300226,
    'TRAIL_BRAKE_THROTTLE': 0.04847852012829285,
    'PARTIAL_THROTTLE_MARGIN': 8.9744913788975,
    'PARTIAL_THROTTLE_VALUE': 0.3456381761963969,
    'UPSHIFT_RPM': 18292, 'DOWNSHIFT_RPM': 12168,
    'SHIFT_SLIP_DETECTION': 1.2137441115484418, 'SHIFT_WAIT_STEPS': 7,
    'ENGINE_BRAKE_RPM_BOOST': 1000,
    'APEX_ACTIVATION_DIST': 83.68514299602667,
    'TRACK_EDGE_DIFF_TRIGGER': 4.8527564193828185,
    'APEX_BIAS_CAP': 0.5550670658104218,
    'APEX_DIVISOR': 127.25517515739612,
    'HIGH_SPEED_STEER_DAMP_SPEED': 98.80593407981057,
    'HIGH_SPEED_STEER_DAMP_BASE': 1.5319476290747356,
    'HIGH_SPEED_STEER_DAMP_DIVISOR': 3.614048403481607,
    'RACING_LINE_ACTIVATION_DIST': 120.0, 'RACING_LINE_DIR_DIVISOR': 100.0,
    'RACING_LINE_DELTA_THRESHOLD': 0.5,
    'RACING_LINE_ENTRY_OFFSET': 0.3, 'RACING_LINE_EXIT_OFFSET': 0.2,
    'RAINEY_START': 2317.4825425883373, 'RAINEY_END': 2405.439847659891,
    'RAINEY_SPEED_LIMIT': 178.77431624521114,
    'RAINEY_TURN_IN_POS': -0.6349772147835389,
    'CORKSCREW_TURNIN_KP': 5.38400599491688,
    'CORKSCREW_TURNIN_KD': 2.3155993911732278,
    'TURN_11_START': 3205.479207020344, 'TURN_11_END': 3286.8866376283645,
    'TURN_11_SPEED_LIMIT': 139.0711151028296,
    'ABS_SLIP_THRESHOLD': 2.633923660948679, 'ABS_KP': 0.16684576340101862,
    'ABS_KI': 0.02320510063445782, 'ABS_KD': 0.09619036105559349,
    'ABS_INTEGRAL_CAP': 13.015814010987384,
    'TCS_THRESHOLD': 16.84427138318147, 'TCS_KP': 0.02729977440616593,
    'TCS_KI': 0.030819251330101004, 'TCS_KD': 0.002826807961750267,
    'TCS_INTEGRAL_CAP': 4.084129194822758, 'TCS_INTEGRAL_DECAY': 0.9598322013488279,
}


def objective(trial):
    p = dict(BEST_PARAMS)
    
    # --- NEW FEATURES (wide bounds — never tuned) ---
    # Racing Line
    p['RACING_LINE_ACTIVATION_DIST'] = trial.suggest_float('RACING_LINE_ACTIVATION_DIST', 60.0, 180.0)
    p['RACING_LINE_DIR_DIVISOR'] = trial.suggest_float('RACING_LINE_DIR_DIVISOR', 40.0, 200.0)
    p['RACING_LINE_DELTA_THRESHOLD'] = trial.suggest_float('RACING_LINE_DELTA_THRESHOLD', 0.1, 2.0)
    p['RACING_LINE_ENTRY_OFFSET'] = trial.suggest_float('RACING_LINE_ENTRY_OFFSET', 0.05, 0.6)
    p['RACING_LINE_EXIT_OFFSET'] = trial.suggest_float('RACING_LINE_EXIT_OFFSET', 0.05, 0.5)
    
    # Engine Braking
    p['ENGINE_BRAKE_RPM_BOOST'] = trial.suggest_int('ENGINE_BRAKE_RPM_BOOST', 200, 3000)
    
    # Speed-Scaled Penalty
    p['SPEED_PENALTY_REFERENCE'] = trial.suggest_float('SPEED_PENALTY_REFERENCE', 80.0, 250.0)
    
    # --- Core speed/braking (tight bounds) ---
    p['STEER_KP'] = trial.suggest_float('STEER_KP', 5.5, 10.0)
    p['STEER_KD'] = trial.suggest_float('STEER_KD', 5.0, 12.0)
    p['BASE_SPEED_MULT'] = trial.suggest_float('BASE_SPEED_MULT', 5.5, 7.0)
    p['DISTANCE_SCALER'] = trial.suggest_float('DISTANCE_SCALER', 12.0, 19.0)
    p['CORNER_EXIT_PUNCH'] = trial.suggest_float('CORNER_EXIT_PUNCH', 40.0, 68.0)
    p['BRAKE_KP'] = trial.suggest_float('BRAKE_KP', 0.12, 0.25)
    p['BRAKE_TRIGGER_MARGIN'] = trial.suggest_float('BRAKE_TRIGGER_MARGIN', 3.0, 7.5)
    p['TRAIL_BRAKE_FACTOR'] = trial.suggest_float('TRAIL_BRAKE_FACTOR', 0.12, 0.38)
    p['TRAIL_BRAKE_THROTTLE'] = trial.suggest_float('TRAIL_BRAKE_THROTTLE', 0.0, 0.15)
    p['LOOKAHEAD_BLEND_THRESHOLD'] = trial.suggest_float('LOOKAHEAD_BLEND_THRESHOLD', 0.25, 0.6)
    p['LOOKAHEAD_WEIGHT'] = trial.suggest_float('LOOKAHEAD_WEIGHT', 0.8, 0.98)
    p['MID_CORNER_PENALTY'] = trial.suggest_float('MID_CORNER_PENALTY', 3.0, 10.0)
    
    # --- Track zones (tight bounds around Phase 5 best) ---
    p['RAINEY_SPEED_LIMIT'] = trial.suggest_float('RAINEY_SPEED_LIMIT', 160.0, 195.0)
    p['CORKSCREW_TURNIN_KP'] = trial.suggest_float('CORKSCREW_TURNIN_KP', 3.0, 8.0)
    p['CORKSCREW_TURNIN_KD'] = trial.suggest_float('CORKSCREW_TURNIN_KD', 0.5, 4.0)
    p['TURN_11_SPEED_LIMIT'] = trial.suggest_float('TURN_11_SPEED_LIMIT', 120.0, 155.0)

    try: current_best = study.best_value
    except ValueError: current_best = 10000.0
    try: lap_time = run_bot(p, current_best)
    except Exception as e: print(f"  Trial failed: {e}"); lap_time = 10000.0
    time.sleep(1.0)
    return lap_time


if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 6 — Racing Line + Engine Braking + Speed Penalty")
    print("  Baseline: 80.912s | Target: sub-78s")
    print("  Params: 23 | Trials: 300")
    print("=" * 60)

    study = optuna.create_study(
        direction='minimize', study_name="torcs_phase6_racing_line",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15))

    baseline_trial = {
        'RACING_LINE_ACTIVATION_DIST': 120.0, 'RACING_LINE_DIR_DIVISOR': 100.0,
        'RACING_LINE_DELTA_THRESHOLD': 0.5,
        'RACING_LINE_ENTRY_OFFSET': 0.3, 'RACING_LINE_EXIT_OFFSET': 0.2,
        'ENGINE_BRAKE_RPM_BOOST': 1000, 'SPEED_PENALTY_REFERENCE': 150.0,
        'STEER_KP': 7.55, 'STEER_KD': 8.66,
        'BASE_SPEED_MULT': 6.111, 'DISTANCE_SCALER': 15.516,
        'CORNER_EXIT_PUNCH': 51.922,
        'BRAKE_KP': 0.189, 'BRAKE_TRIGGER_MARGIN': 5.227,
        'TRAIL_BRAKE_FACTOR': 0.252, 'TRAIL_BRAKE_THROTTLE': 0.048,
        'LOOKAHEAD_BLEND_THRESHOLD': 0.423, 'LOOKAHEAD_WEIGHT': 0.932,
        'MID_CORNER_PENALTY': 6.15,
        'RAINEY_SPEED_LIMIT': 178.774,
        'CORKSCREW_TURNIN_KP': 5.384, 'CORKSCREW_TURNIN_KD': 2.316,
        'TURN_11_SPEED_LIMIT': 139.071,
    }
    study.enqueue_trial(baseline_trial)
    study.optimize(objective, n_trials=300)

    print(f"\n  PHASE 6 Best: {study.best_value:.3f}s")
    final_params = dict(BEST_PARAMS)
    final_params.update(study.best_params)
    with open('phase6_best.json', 'w') as f:
        json.dump({'params': final_params, 'time': study.best_value}, f, indent=2)
    print(f"  Saved to: phase6_best.json")
    print("\n# --- Copy these into bot.py ---")
    for k, v in sorted(final_params.items()): print(f"{k} = {v}")
