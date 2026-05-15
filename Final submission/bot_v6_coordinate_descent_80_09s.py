from torcs_jm_par import Client
import math 

# ======================================================================
# BOT — Deep CMA-ES + Coordinate Descent optimized (80.09s standing lap | 77.13s flying lap)
# Final parameters from targeted coordinate descent (v16)
# ======================================================================

# --- 1. Steering PD ---
STEER_KP = 8.423663942293075
STEER_KD = 9.922233945688301
CENTERING_GAIN = 0.026165060655430138

# --- 2. Speed Control ---
BASE_SPEED_MULT = 6.76078742274564
DISTANCE_SCALER = 13.228014476129015
CORNER_EXIT_PUNCH = 53.83523815927046
MID_CORNER_PENALTY = 5.20022359455938
WALL_FEAR_MULT = 3.2751166162632352
LOOKAHEAD_BLEND_THRESHOLD = 0.304962872026747
LOOKAHEAD_WEIGHT = 0.8806456092999848
SPEED_PENALTY_REFERENCE = 150.27501272576808

# --- 3. Braking ---
BRAKE_KP = 0.1388478270481817
BRAKE_KD = 0.006388142485283601
BRAKE_TRIGGER_MARGIN = 6.154916459412998
TRAIL_BRAKE_FACTOR = 0.3322778427835175
TRAIL_BRAKE_THROTTLE = 0.06022619641937321
PARTIAL_THROTTLE_MARGIN = 9.169766649730292
PARTIAL_THROTTLE_VALUE = 0.3656797647079652

# --- 4. Shifting ---
UPSHIFT_RPM = 18292
DOWNSHIFT_RPM = 12168
SHIFT_SLIP_DETECTION = 1.2137441115484418
SHIFT_WAIT_STEPS = 7
ENGINE_BRAKE_RPM_BOOST = 2030

# --- 5. Apex & Damping ---
APEX_ACTIVATION_DIST = 83.68514299602667
TRACK_EDGE_DIFF_TRIGGER = 4.8527564193828185
APEX_BIAS_CAP = 0.5631589421040883
APEX_DIVISOR = 127.25517515739612
HIGH_SPEED_STEER_DAMP_SPEED = 98.80593407981057
HIGH_SPEED_STEER_DAMP_BASE = 1.5396882895799664
HIGH_SPEED_STEER_DAMP_DIVISOR = 3.3393013902625652

# --- 6. Racing Line ---
RACING_LINE_ACTIVATION_DIST = 66.37194832614182
RACING_LINE_DIR_DIVISOR = 69.70453097185721
RACING_LINE_DELTA_THRESHOLD = 1.8714262413577176
RACING_LINE_ENTRY_OFFSET = 0.5130629691792599
RACING_LINE_EXIT_OFFSET = 0.36013269745817234

# --- 7. Track Zones (3 of 4) ---
R_S = 2317.4825425883373
R_E = 2405.439847659891
R_S_L = 191.72702611542414
R_T_H_P = -0.6349772147835389
CORKSCREW_TURNIN_KP = 6.0771614587555
CORKSCREW_TURNIN_KD = 2.7482734838493066
T11S = 3205.479207020344
T11E = 3286.8866376283645
T11SL = 125.43681252597926

# --- 8. ABS PID ---
ABS_SLIP_THRESHOLD = 2.633923660948679
ABS_KP = 0.16684576340101862
ABS_KI = 0.02320510063445782
ABS_KD = 0.09619036105559349
ABS_INTEGRAL_CAP = 13.015814010987384

# --- 9. TCS PID ---
TCS_THRESHOLD = 16.84427138318147
TCS_KP = 0.02729977440616593
TCS_KI = 0.030819251330101004
TCS_KD = 0.002826807961750267
TCS_INTEGRAL_CAP = 4.084129194822758
TCS_INTEGRAL_DECAY = 0.9598322013488279

TRACK_FINGERPRINTS = {"cs": (62.02670, 62.01465)}
FINGERPRINT_TOLERANCE = 2.0
GEARS = [-2.0, 0.0, 3.9, 2.9, 2.3, 1.87, 1.68, 1.54, 1.46]
TIRE_RADIUS = 0.315
TRACK_LENGTH = 3608.45


def set_generic_params():
    global STEER_KP, STEER_KD, CENTERING_GAIN, APEX_BIAS_CAP, HIGH_SPEED_STEER_DAMP_BASE
    global BASE_SPEED_MULT, DISTANCE_SCALER, CORNER_EXIT_PUNCH, MID_CORNER_PENALTY
    global WALL_FEAR_MULT, LOOKAHEAD_BLEND_THRESHOLD, LOOKAHEAD_WEIGHT
    global BRAKE_KP, BRAKE_KD, BRAKE_TRIGGER_MARGIN
    global TRAIL_BRAKE_FACTOR, TRAIL_BRAKE_THROTTLE
    global PARTIAL_THROTTLE_MARGIN, PARTIAL_THROTTLE_VALUE
    global ABS_SLIP_THRESHOLD, ABS_KP, ABS_KI, ABS_KD
    global TCS_THRESHOLD, TCS_KP, TCS_KI, TCS_KD
    global RACING_LINE_ENTRY_OFFSET, RACING_LINE_EXIT_OFFSET
    STEER_KP=8.0; STEER_KD=7.0; CENTERING_GAIN=0.020
    APEX_BIAS_CAP=0.35; HIGH_SPEED_STEER_DAMP_BASE=1.50
    BASE_SPEED_MULT=6.0; DISTANCE_SCALER=15.0
    CORNER_EXIT_PUNCH=45.0; MID_CORNER_PENALTY=7.0
    WALL_FEAR_MULT=3.5; LOOKAHEAD_BLEND_THRESHOLD=0.45; LOOKAHEAD_WEIGHT=0.75
    BRAKE_KP=0.10; BRAKE_KD=0.008; BRAKE_TRIGGER_MARGIN=4.5
    TRAIL_BRAKE_FACTOR=0.20; TRAIL_BRAKE_THROTTLE=0.05
    PARTIAL_THROTTLE_MARGIN=8.0; PARTIAL_THROTTLE_VALUE=0.40
    ABS_SLIP_THRESHOLD=2.6; ABS_KP=0.17; ABS_KI=0.02; ABS_KD=0.08
    TCS_THRESHOLD=14.0; TCS_KP=0.03; TCS_KI=0.02; TCS_KD=0.005
    RACING_LINE_ENTRY_OFFSET=0.30; RACING_LINE_EXIT_OFFSET=0.20


def reset_state():
    for attr in ['prev_error', 'prev_center_dist']:
        if hasattr(calculate_steering, attr): delattr(calculate_steering, attr)
    for attr in ['prev_dist', 'prev_speed_error']:
        if hasattr(speed_control, attr): delattr(speed_control, attr)
    for attr in ['slip_integral', 'prev_slip']:
        if hasattr(traction_control, attr): delattr(traction_control, attr)
    for attr in ['slip_integral', 'prev_slip']:
        if hasattr(apply_abs, attr): delattr(apply_abs, attr)
    for attr in ['last_shift_step']:
        if hasattr(shift_stable, attr): delattr(shift_stable, attr)
    for attr in ['track_profiled', 'current_track', 'fp1_hist', 'fp2_hist',
                 'fp1', 'fp2', 'prev_turnin_error']:
        if hasattr(drive, attr): delattr(drive, attr)


def shift_stable(S, R, step):
    if not hasattr(shift_stable, "last_shift_step"):
        shift_stable.last_shift_step = -100
    current_gear = int(R['gear'])
    current_rpm = S['rpm']
    if current_gear < 1:
        shift_stable.last_shift_step = step; return 1
    if (step - shift_stable.last_shift_step) < SHIFT_WAIT_STEPS:
        if R['brake'] > 0.5 and current_gear > 1 and current_rpm < DOWNSHIFT_RPM + ENGINE_BRAKE_RPM_BOOST:
            shift_stable.last_shift_step = step; return current_gear - 1
        return current_gear
    speed_ms = S['speedX'] / 3.6
    avg_ws = (S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) / 2.0
    is_slipping = (avg_ws * TIRE_RADIUS) > (speed_ms * SHIFT_SLIP_DETECTION)
    if current_gear < 7:
        if current_rpm > UPSHIFT_RPM and not is_slipping:
            shift_stable.last_shift_step = step; return current_gear + 1
    if current_gear > 1:
        if current_rpm < DOWNSHIFT_RPM:
            shift_stable.last_shift_step = step; return current_gear - 1
    return current_gear


def calculate_steering(S, current_track):
    if not hasattr(calculate_steering, 'prev_error'):
        calculate_steering.prev_error = 0.0
        calculate_steering.prev_center_dist = 200.0
    if current_track == "cs" and S['distRaced'] < 60: return 0.01
    elif current_track != "cs" and S['distRaced'] < 10: return 0.01

    center_dist = S['track'][9]
    left_open = S['track'][13] + S['track'][14] + S['track'][15]
    right_open = S['track'][3] + S['track'][4] + S['track'][5]

    delta_center = center_dist - calculate_steering.prev_center_dist
    calculate_steering.prev_center_dist = center_dist
    racing_line_target = 0.0
    if center_dist < RACING_LINE_ACTIVATION_DIST:
        ci = 1.0 - (center_dist / RACING_LINE_ACTIVATION_DIST)
        dn = max(-1.0, min(1.0, (left_open - right_open) / RACING_LINE_DIR_DIVISOR))
        if delta_center < -RACING_LINE_DELTA_THRESHOLD:
            racing_line_target = dn * RACING_LINE_ENTRY_OFFSET * ci
        elif delta_center > RACING_LINE_DELTA_THRESHOLD:
            racing_line_target = dn * RACING_LINE_EXIT_OFFSET * ci

    apex_bias = 0.0
    if center_dist < APEX_ACTIVATION_DIST:
        diff = left_open - right_open
        if abs(diff) > TRACK_EDGE_DIFF_TRIGGER:
            apex_bias = -APEX_BIAS_CAP * (diff / APEX_DIVISOR)
            apex_bias = max(-APEX_BIAS_CAP, min(APEX_BIAS_CAP, apex_bias))

    target_pos = (S['trackPos'] - racing_line_target) * CENTERING_GAIN
    error = S['angle'] - target_pos + apex_bias
    d_error = error - calculate_steering.prev_error
    calculate_steering.prev_error = error
    steer = STEER_KP * error + STEER_KD * d_error

    if S['speedX'] > HIGH_SPEED_STEER_DAMP_SPEED:
        scale = HIGH_SPEED_STEER_DAMP_BASE + (
            math.sqrt(S['speedX'] - HIGH_SPEED_STEER_DAMP_SPEED) / HIGH_SPEED_STEER_DAMP_DIVISOR)
        steer /= scale
    return max(-1.0, min(1.0, steer))


def apply_abs(S, current_brake):
    if current_brake <= 0.0: return 0.0
    speed_ms = S['speedX'] / 3.6
    if speed_ms < 5.0: return current_brake
    if not hasattr(apply_abs, 'slip_integral'):
        apply_abs.slip_integral = 0.0; apply_abs.prev_slip = 0.0
    slip = speed_ms - (sum(S['wheelSpinVel']) / 4.0 * TIRE_RADIUS)
    if slip > ABS_SLIP_THRESHOLD:
        se = slip - ABS_SLIP_THRESHOLD
        apply_abs.slip_integral = min(apply_abs.slip_integral + se, ABS_INTEGRAL_CAP)
        ds = se - apply_abs.prev_slip; apply_abs.prev_slip = se
        return max(0.0, current_brake - (ABS_KP*se + ABS_KI*apply_abs.slip_integral + ABS_KD*ds))
    else:
        apply_abs.slip_integral *= 0.9; apply_abs.prev_slip = 0.0
        return current_brake


def speed_control(S, R, flying_lap=False):
    fwd = max(S['track'][8], S['track'][9], S['track'][10])
    cw = min(S['track'][3], S['track'][4], S['track'][5],
             S['track'][13], S['track'][14], S['track'][15])
    if cw < fwd * LOOKAHEAD_BLEND_THRESHOLD and fwd > 10:
        ed = fwd * LOOKAHEAD_WEIGHT + cw * (1.0 - LOOKAHEAD_WEIGHT)
    else: ed = fwd
    sc = min(S['track'][4], S['track'][14])
    if sc < 20: ed = min(ed, sc * WALL_FEAR_MULT)
    if ed == -1 or ed > 200: ed = 200.0
    elif ed <= 0: ed = 1.0
    if not hasattr(speed_control, "prev_dist"):
        speed_control.prev_dist = ed; speed_control.prev_speed_error = 0.0
    dd = ed - speed_control.prev_dist; speed_control.prev_dist = ed
    ts = math.sqrt(ed * DISTANCE_SCALER) * BASE_SPEED_MULT
    if dd > 0.5 and ed > 30.0: ts += CORNER_EXIT_PUNCH
    else:
        sf = max(0.5, S['speedX'] / SPEED_PENALTY_REFERENCE)
        ts -= abs(R['steer']) * MID_CORNER_PENALTY * sf
    cs = S['speedX']; a = 0.0; b = 0.0
    se = cs - ts; dse = se - speed_control.prev_speed_error; speed_control.prev_speed_error = se
    if se > BRAKE_TRIGGER_MARGIN:
        b = max(0.0, min(1.0, BRAKE_KP * se + BRAKE_KD * dse)); a = 0.0
    elif se > 0:
        b = max(0.0, min(1.0, TRAIL_BRAKE_FACTOR * se)); a = TRAIL_BRAKE_THROTTLE
    else:
        a = 1.0
        if cs > ts - PARTIAL_THROTTLE_MARGIN: a = PARTIAL_THROTTLE_VALUE
        b = 0.0
    if cs < 10 and ed > 10 and b < 0.1: a = 1.0; b = 0.0
    if ed >= 200: a = 1.0; b = 0.0
    return a, b


def traction_control(S, R):
    if not hasattr(traction_control, 'slip_integral'):
        traction_control.slip_integral = 0.0; traction_control.prev_slip = 0.0
    accel = R['accel']
    slip = (S['wheelSpinVel'][2]+S['wheelSpinVel'][3]) - (S['wheelSpinVel'][0]+S['wheelSpinVel'][1])
    if slip > TCS_THRESHOLD:
        se = slip - TCS_THRESHOLD
        traction_control.slip_integral = max(0.0, min(traction_control.slip_integral + se, TCS_INTEGRAL_CAP))
        ds = se - traction_control.prev_slip
        accel = max(0.0, accel - (TCS_KP*se + TCS_KI*traction_control.slip_integral + TCS_KD*ds))
    else: traction_control.slip_integral *= TCS_INTEGRAL_DECAY
    traction_control.prev_slip = max(0.0, slip - TCS_THRESHOLD)
    return accel


def drive(c, step_counter):
    S = c.S.d; R = c.R.d
    if not hasattr(drive, "track_profiled"):
        drive.track_profiled = False; drive.current_track = "cs"
        drive.fp1_hist = []; drive.fp2_hist = []; drive.prev_turnin_error = 0.0
    if not drive.track_profiled:
        if 5 <= step_counter <= 15:
            if abs(S['track'][9] - 62.02670) > 4.0:
                drive.current_track = "unknown"; drive.track_profiled = True; set_generic_params()
        if not drive.track_profiled:
            if 5 <= step_counter <= 25: drive.fp1_hist.append(S['track'][9])
            elif step_counter == 26: drive.fp1 = sum(drive.fp1_hist)/len(drive.fp1_hist)
            if 30 <= step_counter <= 50: drive.fp2_hist.append(S['track'][9])
            elif step_counter == 51:
                drive.fp2 = sum(drive.fp2_hist)/len(drive.fp2_hist)
                mt = None
                for tn, (e1,e2) in TRACK_FINGERPRINTS.items():
                    if e1<0: continue
                    if abs(drive.fp1-e1)<FINGERPRINT_TOLERANCE and abs(drive.fp2-e2)<FINGERPRINT_TOLERANCE:
                        mt = tn; break
                if mt: drive.current_track = mt
                else: drive.current_track = "unknown"; set_generic_params()
                drive.track_profiled = True

    R['steer'] = calculate_steering(S, drive.current_track)
    current_lap = int(S['distRaced'] / TRACK_LENGTH)
    is_flying = False
    a, b = speed_control(S, R, flying_lap=is_flying)
    R['brake'] = apply_abs(S, b); R['accel'] = a
    R['accel'] = traction_control(S, R)
    cd = S['distFromStart'] % TRACK_LENGTH
    if drive.current_track == "cs":
        corkscrew_limit = R_S_L 
        t11_limit = T11SL 
        if R_S < cd < R_E:
            if S['speedX'] > corkscrew_limit: R['accel'] = 0.0; R['brake'] = 1.0
            if S['trackPos'] < R_T_H_P:
                te = R_T_H_P - S['trackPos']; dt = te - drive.prev_turnin_error
                drive.prev_turnin_error = te
                R['steer'] += CORKSCREW_TURNIN_KP * te + CORKSCREW_TURNIN_KD * dt
            else: drive.prev_turnin_error = 0.0
        else: drive.prev_turnin_error = 0.0
        if T11S < cd < T11E:
            if S['speedX'] > t11_limit: R['accel'] = 0.0; R['brake'] = 1.0
        # Full throttle from T11 end to finish
        if cd > T11E and S['distRaced'] > 1000: R['accel'] = 1.0; R['brake'] = 0.0
    R['gear'] = shift_stable(S, R, step_counter)
    # Launch: full throttle for 50 steps, but only lock gear 1 for first 15
    if step_counter < 50: R['accel'] = 1.0; R['brake'] = 0.0; R['clutch'] = 0.0
    if step_counter < 15: R['gear'] = 1
    R['meta'] = 0
    if not hasattr(drive, 'last_telem_pos'): drive.last_telem_pos = -500



if __name__ == "__main__":
    C = Client(p=3001)
    total_steps = C.maxSteps
    for sl in range(C.maxSteps, 0, -1):
        sc = total_steps - sl
        C.get_servers_input(); drive(C, sc); C.respond_to_server()
    C.shutdown()
