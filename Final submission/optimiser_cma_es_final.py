import optuna, time, json
from actualBaselineBot import run_bot

# EXACT V7 params (80.370s standalone) — full precision, no rounding
V7 = {
    'STEER_KP': 8.616023978509647, 'STEER_KD': 10.253216712881247,
    'CENTERING_GAIN': 0.026278308118437656,
    'BASE_SPEED_MULT': 6.752179590978522, 'DISTANCE_SCALER': 13.718775973419172,
    'CORNER_EXIT_PUNCH': 51.40190501229689,
    'MID_CORNER_PENALTY': 5.347371771934745, 'WALL_FEAR_MULT': 3.2751166162632352,
    'LOOKAHEAD_BLEND_THRESHOLD': 0.32117116163132, 'LOOKAHEAD_WEIGHT': 0.8570481661379072,
    'SPEED_PENALTY_REFERENCE': 150.27501272576808,
    'BRAKE_KP': 0.14252306983126253, 'BRAKE_KD': 0.006388142485283601,
    'BRAKE_TRIGGER_MARGIN': 6.394364845963751, 'TRAIL_BRAKE_FACTOR': 0.3198734861241366,
    'TRAIL_BRAKE_THROTTLE': 0.06144089070800798,
    'PARTIAL_THROTTLE_MARGIN': 8.9744913788975, 'PARTIAL_THROTTLE_VALUE': 0.3456381761963969,
    'UPSHIFT_RPM': 18292, 'DOWNSHIFT_RPM': 12168,
    'SHIFT_SLIP_DETECTION': 1.2137441115484418, 'SHIFT_WAIT_STEPS': 7,
    'ENGINE_BRAKE_RPM_BOOST': 2030,
    'APEX_ACTIVATION_DIST': 83.68514299602667, 'TRACK_EDGE_DIFF_TRIGGER': 4.8527564193828185,
    'APEX_BIAS_CAP': 0.5550670658104218, 'APEX_DIVISOR': 127.25517515739612,
    'HIGH_SPEED_STEER_DAMP_SPEED': 98.80593407981057,
    'HIGH_SPEED_STEER_DAMP_BASE': 1.5319476290747356,
    'HIGH_SPEED_STEER_DAMP_DIVISOR': 3.614048403481607,
    'RACING_LINE_ACTIVATION_DIST': 66.37194832614182,
    'RACING_LINE_DIR_DIVISOR': 69.70453097185721,
    'RACING_LINE_DELTA_THRESHOLD': 1.8714262413577176,
    'RACING_LINE_ENTRY_OFFSET': 0.5177472726556489,
    'RACING_LINE_EXIT_OFFSET': 0.36257196069303044,
    'RAINEY_START': 2317.4825425883373, 'RAINEY_END': 2405.439847659891,
    'RAINEY_SPEED_LIMIT': 183.3812658363146, 'RAINEY_TURN_IN_POS': -0.6349772147835389,
    'CORKSCREW_TURNIN_KP': 6.044983076395464, 'CORKSCREW_TURNIN_KD': 2.7389851442772617,
    'TURN_11_START': 3205.479207020344, 'TURN_11_END': 3286.8866376283645,
    'TURN_11_SPEED_LIMIT': 128.60194649283616,
    'KAMM_STEER_SCALE': 0.0,           # Disabled — v7 has no Kamm
    'MID_CORNER_SHIFT_THRESHOLD': 2.0,  # Disabled — v7 has no shift inhibition
    'ABS_SLIP_THRESHOLD': 2.633923660948679, 'ABS_KP': 0.16684576340101862,
    'ABS_KI': 0.02320510063445782, 'ABS_KD': 0.09619036105559349,
    'ABS_INTEGRAL_CAP': 13.015814010987384,
    'TCS_THRESHOLD': 16.84427138318147, 'TCS_KP': 0.02729977440616593,
    'TCS_KI': 0.030819251330101004, 'TCS_KD': 0.002826807961750267,
    'TCS_INTEGRAL_CAP': 4.084129194822758, 'TCS_INTEGRAL_DECAY': 0.9598322013488279,
}

def tight(val, pct):
    """Return (lo, hi) as val +/- pct%"""
    d = abs(val) * pct / 100.0
    return (val - d, val + d)

def objective(trial):
    p = dict(V7)
    # --- Core speed (most impact on lap time) ---
    lo,hi = tight(V7['BASE_SPEED_MULT'], 10)
    p['BASE_SPEED_MULT'] = trial.suggest_float('BSM', lo, hi)
    lo,hi = tight(V7['DISTANCE_SCALER'], 10)
    p['DISTANCE_SCALER'] = trial.suggest_float('DSC', lo, hi)
    lo,hi = tight(V7['CORNER_EXIT_PUNCH'], 15)
    p['CORNER_EXIT_PUNCH'] = trial.suggest_float('CEP', lo, hi)
    lo,hi = tight(V7['MID_CORNER_PENALTY'], 15)
    p['MID_CORNER_PENALTY'] = trial.suggest_float('MCP', lo, hi)
    lo,hi = tight(V7['LOOKAHEAD_WEIGHT'], 8)
    p['LOOKAHEAD_WEIGHT'] = trial.suggest_float('LAW', lo, hi)
    # --- Braking ---
    lo,hi = tight(V7['BRAKE_KP'], 15)
    p['BRAKE_KP'] = trial.suggest_float('BKP', lo, hi)
    lo,hi = tight(V7['BRAKE_TRIGGER_MARGIN'], 15)
    p['BRAKE_TRIGGER_MARGIN'] = trial.suggest_float('BTM', lo, hi)
    lo,hi = tight(V7['TRAIL_BRAKE_FACTOR'], 15)
    p['TRAIL_BRAKE_FACTOR'] = trial.suggest_float('TBF', lo, hi)
    # --- Steering ---
    lo,hi = tight(V7['STEER_KP'], 8)
    p['STEER_KP'] = trial.suggest_float('SKP', lo, hi)
    lo,hi = tight(V7['STEER_KD'], 10)
    p['STEER_KD'] = trial.suggest_float('SKD', lo, hi)
    # --- Track zones ---
    lo,hi = tight(V7['RAINEY_SPEED_LIMIT'], 8)
    p['RAINEY_SPEED_LIMIT'] = trial.suggest_float('RSL', lo, hi)
    lo,hi = tight(V7['TURN_11_SPEED_LIMIT'], 10)
    p['TURN_11_SPEED_LIMIT'] = trial.suggest_float('T11SL', lo, hi)
    # --- Partial throttle ---
    lo,hi = tight(V7['PARTIAL_THROTTLE_MARGIN'], 15)
    p['PARTIAL_THROTTLE_MARGIN'] = trial.suggest_float('PTM', lo, hi)
    lo,hi = tight(V7['PARTIAL_THROTTLE_VALUE'], 15)
    p['PARTIAL_THROTTLE_VALUE'] = trial.suggest_float('PTV', lo, hi)

    try: best = study.best_value
    except ValueError: best = 10000.0
    try: return run_bot(p, best)
    except Exception as e: print(f"  Trial failed: {e}"); return 10000.0
    finally: time.sleep(1.0)

if __name__ == "__main__":
    print("=" * 60)
    print("  V7 CMA-ES Refinement (tight bounds, exact params)")
    print("  Baseline: 80.370s | Target: sub-80s")
    print("  Params: 14 | Trials: 200")
    print("  Sampler: CMA-ES (local optimization)")
    print("=" * 60)

    x0 = {
        'BSM': V7['BASE_SPEED_MULT'], 'DSC': V7['DISTANCE_SCALER'],
        'CEP': V7['CORNER_EXIT_PUNCH'], 'MCP': V7['MID_CORNER_PENALTY'],
        'LAW': V7['LOOKAHEAD_WEIGHT'],
        'BKP': V7['BRAKE_KP'], 'BTM': V7['BRAKE_TRIGGER_MARGIN'],
        'TBF': V7['TRAIL_BRAKE_FACTOR'],
        'SKP': V7['STEER_KP'], 'SKD': V7['STEER_KD'],
        'RSL': V7['RAINEY_SPEED_LIMIT'], 'T11SL': V7['TURN_11_SPEED_LIMIT'],
        'PTM': V7['PARTIAL_THROTTLE_MARGIN'], 'PTV': V7['PARTIAL_THROTTLE_VALUE'],
    }
    study = optuna.create_study(direction='minimize', study_name="torcs_v7_cma",
        sampler=optuna.samplers.CmaEsSampler(x0=x0, sigma0=0.05))
    # Enqueue exact v7 baseline as first trial
    study.enqueue_trial(x0)
    study.optimize(objective, n_trials=200)
    print(f"\n  V7 CMA Best: {study.best_value:.3f}s")
    remap = {'BSM':'BASE_SPEED_MULT','DSC':'DISTANCE_SCALER','CEP':'CORNER_EXIT_PUNCH',
             'MCP':'MID_CORNER_PENALTY','LAW':'LOOKAHEAD_WEIGHT',
             'BKP':'BRAKE_KP','BTM':'BRAKE_TRIGGER_MARGIN','TBF':'TRAIL_BRAKE_FACTOR',
             'SKP':'STEER_KP','SKD':'STEER_KD',
             'RSL':'RAINEY_SPEED_LIMIT','T11SL':'TURN_11_SPEED_LIMIT',
             'PTM':'PARTIAL_THROTTLE_MARGIN','PTV':'PARTIAL_THROTTLE_VALUE'}
    clean = dict(V7)
    for short, long in remap.items():
        if short in study.best_params: clean[long] = study.best_params[short]
    with open('v7_cma_best.json', 'w') as f:
        json.dump({'params': clean, 'time': study.best_value}, f, indent=2)
    print(f"  Saved to: v7_cma_best.json")
    print("\n# --- Copy these into bot_v7 copy ---")
    for k, v in sorted(clean.items()): print(f"{k} = {v}")
