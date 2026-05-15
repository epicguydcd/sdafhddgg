import optuna
import time
import json
from actualBaselineBot import run_bot

# ======================================================================
# STAGED OPTUNA OPTIMIZER
# 
# 4-phase optimization: steering → speed/brakes → TCS/ABS → full unlock
# Each phase freezes all other params at current best and only explores
# the target subsystem. This finds better results than exploring all
# 45+ params simultaneously.
# ======================================================================

# --- Reasonable starting values matching the new PD/PID architecture ---
BASELINE_PARAMS = {
    # Steering PD
    'STEER_KP': 8.854,
    'STEER_KD': 2.0,
    'CENTERING_GAIN': 0.0075,
    # Speed control
    'BASE_SPEED_MULT': 5.24,
    'DISTANCE_SCALER': 19.85,
    'CORNER_EXIT_PUNCH': 53.96,
    'MID_CORNER_PENALTY': 9.31,
    'WALL_FEAR_MULT': 3.37,
    'LOOKAHEAD_BLEND_THRESHOLD': 0.7,
    'LOOKAHEAD_WEIGHT': 0.6,
    # Braking PD + trail-braking
    'BRAKE_KP': 0.052,
    'BRAKE_KD': 0.02,
    'BRAKE_TRIGGER_MARGIN': 2.0,
    'TRAIL_BRAKE_FACTOR': 0.15,
    'TRAIL_BRAKE_THROTTLE': 0.1,
    'PARTIAL_THROTTLE_MARGIN': 6.76,
    'PARTIAL_THROTTLE_VALUE': 0.59,
    # Shifting
    'UPSHIFT_RPM': 17961,
    'DOWNSHIFT_RPM': 11443,
    'SHIFT_SLIP_DETECTION': 1.115,
    'SHIFT_WAIT_STEPS': 6,
    # Apex
    'APEX_ACTIVATION_DIST': 125.88,
    'TRACK_EDGE_DIFF_TRIGGER': 4.71,
    'APEX_BIAS_CAP': 0.457,
    'APEX_DIVISOR': 189.15,
    'HIGH_SPEED_STEER_DAMP_SPEED': 110.0,
    'HIGH_SPEED_STEER_DAMP_BASE': 1.336,
    'HIGH_SPEED_STEER_DAMP_DIVISOR': 2.324,
    # Corkscrew zone
    'RAINEY_START': 2271.86,
    'RAINEY_END': 2401.47,
    'RAINEY_SPEED_LIMIT': 171.64,
    'RAINEY_TURN_IN_POS': -0.770,
    'CORKSCREW_TURNIN_KP': 3.5,
    'CORKSCREW_TURNIN_KD': 0.5,
    # Turn 11 zone
    'TURN_11_START': 3164.74,
    'TURN_11_END': 3250.66,
    'TURN_11_SPEED_LIMIT': 120.93,
    # ABS PID
    'ABS_SLIP_THRESHOLD': 2.986,
    'ABS_KP': 0.2,
    'ABS_KI': 0.01,
    'ABS_KD': 0.02,
    'ABS_INTEGRAL_CAP': 5.0,
    # TCS PID
    'TCS_THRESHOLD': 11.99,
    'TCS_KP': 0.038,
    'TCS_KI': 0.005,
    'TCS_KD': 0.01,
    'TCS_INTEGRAL_CAP': 5.0,
    'TCS_INTEGRAL_DECAY': 0.9,
}


def get_phase_params(trial, phase, frozen):
    """Build full param dict: unfrozen params from trial, rest from frozen."""
    p = dict(frozen)  # Start with all frozen values
    
    if phase == 1:  # STEERING
        p['STEER_KP'] = trial.suggest_float('STEER_KP', 4.0, 15.0)
        p['STEER_KD'] = trial.suggest_float('STEER_KD', 0.3, 10.0)
        p['CENTERING_GAIN'] = trial.suggest_float('CENTERING_GAIN', 0.0, 0.05)
        p['APEX_ACTIVATION_DIST'] = trial.suggest_float('APEX_ACTIVATION_DIST', 80.0, 220.0)
        p['TRACK_EDGE_DIFF_TRIGGER'] = trial.suggest_float('TRACK_EDGE_DIFF_TRIGGER', 2.0, 10.0)
        p['APEX_BIAS_CAP'] = trial.suggest_float('APEX_BIAS_CAP', 0.1, 0.6)
        p['APEX_DIVISOR'] = trial.suggest_float('APEX_DIVISOR', 120.0, 300.0)
        p['HIGH_SPEED_STEER_DAMP_SPEED'] = trial.suggest_float('HIGH_SPEED_STEER_DAMP_SPEED', 70.0, 140.0)
        p['HIGH_SPEED_STEER_DAMP_BASE'] = trial.suggest_float('HIGH_SPEED_STEER_DAMP_BASE', 0.8, 2.0)
        p['HIGH_SPEED_STEER_DAMP_DIVISOR'] = trial.suggest_float('HIGH_SPEED_STEER_DAMP_DIVISOR', 1.5, 6.0)

    elif phase == 2:  # SPEED CONTROL + BRAKING
        p['BASE_SPEED_MULT'] = trial.suggest_float('BASE_SPEED_MULT', 4.0, 7.5)
        p['DISTANCE_SCALER'] = trial.suggest_float('DISTANCE_SCALER', 12.0, 25.0)
        p['CORNER_EXIT_PUNCH'] = trial.suggest_float('CORNER_EXIT_PUNCH', 25.0, 70.0)
        p['MID_CORNER_PENALTY'] = trial.suggest_float('MID_CORNER_PENALTY', 4.0, 15.0)
        p['WALL_FEAR_MULT'] = trial.suggest_float('WALL_FEAR_MULT', 1.5, 6.0)
        p['LOOKAHEAD_BLEND_THRESHOLD'] = trial.suggest_float('LOOKAHEAD_BLEND_THRESHOLD', 0.3, 0.95)
        p['LOOKAHEAD_WEIGHT'] = trial.suggest_float('LOOKAHEAD_WEIGHT', 0.2, 0.9)
        p['BRAKE_KP'] = trial.suggest_float('BRAKE_KP', 0.02, 0.2)
        p['BRAKE_KD'] = trial.suggest_float('BRAKE_KD', 0.005, 0.15)
        p['BRAKE_TRIGGER_MARGIN'] = trial.suggest_float('BRAKE_TRIGGER_MARGIN', 0.5, 8.0)
        p['TRAIL_BRAKE_FACTOR'] = trial.suggest_float('TRAIL_BRAKE_FACTOR', 0.02, 0.4)
        p['TRAIL_BRAKE_THROTTLE'] = trial.suggest_float('TRAIL_BRAKE_THROTTLE', 0.0, 0.4)
        p['PARTIAL_THROTTLE_MARGIN'] = trial.suggest_float('PARTIAL_THROTTLE_MARGIN', 2.0, 10.0)
        p['PARTIAL_THROTTLE_VALUE'] = trial.suggest_float('PARTIAL_THROTTLE_VALUE', 0.3, 0.95)

    elif phase == 3:  # TCS + ABS + SHIFTING
        p['ABS_SLIP_THRESHOLD'] = trial.suggest_float('ABS_SLIP_THRESHOLD', 1.0, 5.0)
        p['ABS_KP'] = trial.suggest_float('ABS_KP', 0.05, 0.5)
        p['ABS_KI'] = trial.suggest_float('ABS_KI', 0.001, 0.08)
        p['ABS_KD'] = trial.suggest_float('ABS_KD', 0.001, 0.15)
        p['ABS_INTEGRAL_CAP'] = trial.suggest_float('ABS_INTEGRAL_CAP', 1.0, 15.0)
        p['TCS_THRESHOLD'] = trial.suggest_float('TCS_THRESHOLD', 4.0, 18.0)
        p['TCS_KP'] = trial.suggest_float('TCS_KP', 0.01, 0.15)
        p['TCS_KI'] = trial.suggest_float('TCS_KI', 0.001, 0.06)
        p['TCS_KD'] = trial.suggest_float('TCS_KD', 0.001, 0.08)
        p['TCS_INTEGRAL_CAP'] = trial.suggest_float('TCS_INTEGRAL_CAP', 1.0, 15.0)
        p['TCS_INTEGRAL_DECAY'] = trial.suggest_float('TCS_INTEGRAL_DECAY', 0.75, 0.995)
        p['UPSHIFT_RPM'] = trial.suggest_int('UPSHIFT_RPM', 16000, 18500)
        p['DOWNSHIFT_RPM'] = trial.suggest_int('DOWNSHIFT_RPM', 8000, 13000)
        p['SHIFT_SLIP_DETECTION'] = trial.suggest_float('SHIFT_SLIP_DETECTION', 1.05, 1.8)
        p['SHIFT_WAIT_STEPS'] = trial.suggest_int('SHIFT_WAIT_STEPS', 3, 10)

    elif phase == 4:  # FULL UNLOCK (tighter bounds from phases 1-3)
        p['STEER_KP'] = trial.suggest_float('STEER_KP', 4.0, 15.0)
        p['STEER_KD'] = trial.suggest_float('STEER_KD', 0.3, 10.0)
        p['CENTERING_GAIN'] = trial.suggest_float('CENTERING_GAIN', 0.0, 0.05)
        p['BASE_SPEED_MULT'] = trial.suggest_float('BASE_SPEED_MULT', 4.0, 7.5)
        p['DISTANCE_SCALER'] = trial.suggest_float('DISTANCE_SCALER', 12.0, 25.0)
        p['CORNER_EXIT_PUNCH'] = trial.suggest_float('CORNER_EXIT_PUNCH', 25.0, 70.0)
        p['MID_CORNER_PENALTY'] = trial.suggest_float('MID_CORNER_PENALTY', 4.0, 15.0)
        p['WALL_FEAR_MULT'] = trial.suggest_float('WALL_FEAR_MULT', 1.5, 6.0)
        p['LOOKAHEAD_BLEND_THRESHOLD'] = trial.suggest_float('LOOKAHEAD_BLEND_THRESHOLD', 0.3, 0.95)
        p['LOOKAHEAD_WEIGHT'] = trial.suggest_float('LOOKAHEAD_WEIGHT', 0.2, 0.9)
        p['BRAKE_KP'] = trial.suggest_float('BRAKE_KP', 0.02, 0.2)
        p['BRAKE_KD'] = trial.suggest_float('BRAKE_KD', 0.005, 0.15)
        p['BRAKE_TRIGGER_MARGIN'] = trial.suggest_float('BRAKE_TRIGGER_MARGIN', 0.5, 8.0)
        p['TRAIL_BRAKE_FACTOR'] = trial.suggest_float('TRAIL_BRAKE_FACTOR', 0.02, 0.4)
        p['TRAIL_BRAKE_THROTTLE'] = trial.suggest_float('TRAIL_BRAKE_THROTTLE', 0.0, 0.4)
        p['PARTIAL_THROTTLE_MARGIN'] = trial.suggest_float('PARTIAL_THROTTLE_MARGIN', 2.0, 10.0)
        p['PARTIAL_THROTTLE_VALUE'] = trial.suggest_float('PARTIAL_THROTTLE_VALUE', 0.3, 0.95)
        p['ABS_SLIP_THRESHOLD'] = trial.suggest_float('ABS_SLIP_THRESHOLD', 1.0, 5.0)
        p['ABS_KP'] = trial.suggest_float('ABS_KP', 0.05, 0.5)
        p['ABS_KI'] = trial.suggest_float('ABS_KI', 0.001, 0.08)
        p['ABS_KD'] = trial.suggest_float('ABS_KD', 0.001, 0.15)
        p['ABS_INTEGRAL_CAP'] = trial.suggest_float('ABS_INTEGRAL_CAP', 1.0, 15.0)
        p['TCS_THRESHOLD'] = trial.suggest_float('TCS_THRESHOLD', 4.0, 18.0)
        p['TCS_KP'] = trial.suggest_float('TCS_KP', 0.01, 0.15)
        p['TCS_KI'] = trial.suggest_float('TCS_KI', 0.001, 0.06)
        p['TCS_KD'] = trial.suggest_float('TCS_KD', 0.001, 0.08)
        p['TCS_INTEGRAL_CAP'] = trial.suggest_float('TCS_INTEGRAL_CAP', 1.0, 15.0)
        p['TCS_INTEGRAL_DECAY'] = trial.suggest_float('TCS_INTEGRAL_DECAY', 0.75, 0.995)
        p['UPSHIFT_RPM'] = trial.suggest_int('UPSHIFT_RPM', 16000, 18500)
        p['DOWNSHIFT_RPM'] = trial.suggest_int('DOWNSHIFT_RPM', 8000, 13000)
        p['SHIFT_SLIP_DETECTION'] = trial.suggest_float('SHIFT_SLIP_DETECTION', 1.05, 1.8)
        p['SHIFT_WAIT_STEPS'] = trial.suggest_int('SHIFT_WAIT_STEPS', 3, 10)
        p['APEX_ACTIVATION_DIST'] = trial.suggest_float('APEX_ACTIVATION_DIST', 80.0, 220.0)
        p['TRACK_EDGE_DIFF_TRIGGER'] = trial.suggest_float('TRACK_EDGE_DIFF_TRIGGER', 2.0, 10.0)
        p['APEX_BIAS_CAP'] = trial.suggest_float('APEX_BIAS_CAP', 0.1, 0.6)
        p['APEX_DIVISOR'] = trial.suggest_float('APEX_DIVISOR', 120.0, 300.0)
        p['HIGH_SPEED_STEER_DAMP_SPEED'] = trial.suggest_float('HIGH_SPEED_STEER_DAMP_SPEED', 70.0, 140.0)
        p['HIGH_SPEED_STEER_DAMP_BASE'] = trial.suggest_float('HIGH_SPEED_STEER_DAMP_BASE', 0.8, 2.0)
        p['HIGH_SPEED_STEER_DAMP_DIVISOR'] = trial.suggest_float('HIGH_SPEED_STEER_DAMP_DIVISOR', 1.5, 6.0)
        # Track zones
        p['RAINEY_START'] = trial.suggest_float('RAINEY_START', 2220.0, 2320.0)
        p['RAINEY_END'] = trial.suggest_float('RAINEY_END', 2370.0, 2450.0)
        p['RAINEY_SPEED_LIMIT'] = trial.suggest_float('RAINEY_SPEED_LIMIT', 140.0, 190.0)
        p['RAINEY_TURN_IN_POS'] = trial.suggest_float('RAINEY_TURN_IN_POS', -0.95, -0.4)
        p['CORKSCREW_TURNIN_KP'] = trial.suggest_float('CORKSCREW_TURNIN_KP', 1.0, 8.0)
        p['CORKSCREW_TURNIN_KD'] = trial.suggest_float('CORKSCREW_TURNIN_KD', 0.1, 3.0)
        p['TURN_11_START'] = trial.suggest_float('TURN_11_START', 3120.0, 3210.0)
        p['TURN_11_END'] = trial.suggest_float('TURN_11_END', 3220.0, 3300.0)
        p['TURN_11_SPEED_LIMIT'] = trial.suggest_float('TURN_11_SPEED_LIMIT', 95.0, 140.0)

    return p


# Maps each phase to the parameter names it explores
PHASE_PARAM_KEYS = {
    1: ['STEER_KP', 'STEER_KD', 'CENTERING_GAIN', 'APEX_ACTIVATION_DIST',
        'TRACK_EDGE_DIFF_TRIGGER', 'APEX_BIAS_CAP', 'APEX_DIVISOR',
        'HIGH_SPEED_STEER_DAMP_SPEED', 'HIGH_SPEED_STEER_DAMP_BASE',
        'HIGH_SPEED_STEER_DAMP_DIVISOR'],
    2: ['BASE_SPEED_MULT', 'DISTANCE_SCALER', 'CORNER_EXIT_PUNCH',
        'MID_CORNER_PENALTY', 'WALL_FEAR_MULT', 'LOOKAHEAD_BLEND_THRESHOLD',
        'LOOKAHEAD_WEIGHT', 'BRAKE_KP', 'BRAKE_KD', 'BRAKE_TRIGGER_MARGIN',
        'TRAIL_BRAKE_FACTOR', 'TRAIL_BRAKE_THROTTLE', 'PARTIAL_THROTTLE_MARGIN',
        'PARTIAL_THROTTLE_VALUE'],
    3: ['ABS_SLIP_THRESHOLD', 'ABS_KP', 'ABS_KI', 'ABS_KD', 'ABS_INTEGRAL_CAP',
        'TCS_THRESHOLD', 'TCS_KP', 'TCS_KI', 'TCS_KD', 'TCS_INTEGRAL_CAP',
        'TCS_INTEGRAL_DECAY', 'UPSHIFT_RPM', 'DOWNSHIFT_RPM',
        'SHIFT_SLIP_DETECTION', 'SHIFT_WAIT_STEPS'],
    4: ['STEER_KP', 'STEER_KD', 'CENTERING_GAIN', 'BASE_SPEED_MULT',
        'DISTANCE_SCALER', 'CORNER_EXIT_PUNCH', 'MID_CORNER_PENALTY',
        'WALL_FEAR_MULT', 'LOOKAHEAD_BLEND_THRESHOLD', 'LOOKAHEAD_WEIGHT',
        'BRAKE_KP', 'BRAKE_KD', 'BRAKE_TRIGGER_MARGIN', 'TRAIL_BRAKE_FACTOR',
        'TRAIL_BRAKE_THROTTLE', 'PARTIAL_THROTTLE_MARGIN', 'PARTIAL_THROTTLE_VALUE',
        'ABS_SLIP_THRESHOLD', 'ABS_KP', 'ABS_KI', 'ABS_KD', 'ABS_INTEGRAL_CAP',
        'TCS_THRESHOLD', 'TCS_KP', 'TCS_KI', 'TCS_KD', 'TCS_INTEGRAL_CAP',
        'TCS_INTEGRAL_DECAY', 'UPSHIFT_RPM', 'DOWNSHIFT_RPM',
        'SHIFT_SLIP_DETECTION', 'SHIFT_WAIT_STEPS',
        'APEX_ACTIVATION_DIST', 'TRACK_EDGE_DIFF_TRIGGER', 'APEX_BIAS_CAP',
        'APEX_DIVISOR', 'HIGH_SPEED_STEER_DAMP_SPEED', 'HIGH_SPEED_STEER_DAMP_BASE',
        'HIGH_SPEED_STEER_DAMP_DIVISOR',
        'RAINEY_START', 'RAINEY_END', 'RAINEY_SPEED_LIMIT', 'RAINEY_TURN_IN_POS',
        'CORKSCREW_TURNIN_KP', 'CORKSCREW_TURNIN_KD',
        'TURN_11_START', 'TURN_11_END', 'TURN_11_SPEED_LIMIT'],
}


def run_phase(phase_num, phase_name, n_trials, frozen_params):
    """Run one optimization phase."""
    print(f"\n{'='*60}")
    print(f"  PHASE {phase_num}: {phase_name}")
    print(f"  Trials: {n_trials}")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        direction='minimize',
        study_name=f"torcs_phase{phase_num}_{phase_name.lower().replace(' ', '_')}",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15)
    )

    # Enqueue baseline as first trial so trial 1 uses known-good values
    baseline_trial = {k: frozen_params[k] for k in PHASE_PARAM_KEYS[phase_num]}
    study.enqueue_trial(baseline_trial)

    def objective(trial):
        try:
            current_best = study.best_value
        except ValueError:
            current_best = 10000.0

        p = get_phase_params(trial, phase_num, frozen_params)

        try:
            lap_time = run_bot(p, current_best)
        except Exception as e:
            print(f"  Trial failed: {e}")
            lap_time = 10000.0

        time.sleep(1.0)
        return lap_time

    study.optimize(objective, n_trials=n_trials)

    print(f"\n  Phase {phase_num} Best: {study.best_value:.3f}s")
    
    # Merge best trial params into frozen for next phase
    best_params = dict(frozen_params)
    best_params.update(study.best_params)
    return best_params, study.best_value


if __name__ == "__main__":
    print("=" * 60)
    print("  STAGED OPTUNA OPTIMIZER — PD/PID Architecture")
    print("  Target: Sub 1:20 on Corkscrew")
    print("=" * 60)

    current_params = dict(BASELINE_PARAMS)

    # Phase 1: Steering PD
    current_params, best_time = run_phase(1, "STEERING PD", 150, current_params)
    print(f"\n  >>> After Phase 1: {best_time:.3f}s")
    
    # Save intermediate results
    with open('phase1_best.json', 'w') as f:
        json.dump({'params': current_params, 'time': best_time}, f, indent=2)

    # Phase 2: Speed Control + Braking
    current_params, best_time = run_phase(2, "SPEED + BRAKING", 150, current_params)
    print(f"\n  >>> After Phase 2: {best_time:.3f}s")
    
    with open('phase2_best.json', 'w') as f:
        json.dump({'params': current_params, 'time': best_time}, f, indent=2)

    # Phase 3: TCS + ABS + Shifting  
    current_params, best_time = run_phase(3, "TCS + ABS + SHIFTING", 150, current_params)
    print(f"\n  >>> After Phase 3: {best_time:.3f}s")
    
    with open('phase3_best.json', 'w') as f:
        json.dump({'params': current_params, 'time': best_time}, f, indent=2)

    # Phase 4: Full unlock with all params
    current_params, best_time = run_phase(4, "FULL UNLOCK", 350, current_params)
    print(f"\n  >>> After Phase 4 (FINAL): {best_time:.3f}s")
    
    with open('final_best.json', 'w') as f:
        json.dump({'params': current_params, 'time': best_time}, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE!")
    print(f"  Final Best Lap Time: {best_time:.3f}s")
    print(f"  Best params saved to: final_best.json")
    print("=" * 60)
    
    # Print params formatted for bot.py
    print("\n# --- Copy these into bot.py ---")
    for k, v in sorted(current_params.items()):
        if isinstance(v, int):
            print(f"{k} = {v}")
        else:
            print(f"{k} = {v}")