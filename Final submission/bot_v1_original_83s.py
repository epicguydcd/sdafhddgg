from torcs_jm_par import Client
import math 

# ======================================================================
# ================= MASTER BOT CONFIGURATION FILE ======================
# ======================================================================

# --- 1. Steering PD Controller ---
STEER_KP = 7.5500778899557766
STEER_KD = 8.661765678765278
CENTERING_GAIN = 0.026278308118437656

# --- 2. Speed Control + Look-Ahead ---
BASE_SPEED_MULT = 6.111111653883914
DISTANCE_SCALER = 15.515669038337016
CORNER_EXIT_PUNCH = 51.921844937822
MID_CORNER_PENALTY = 6.150272999167717
WALL_FEAR_MULT = 3.2751166162632352
LOOKAHEAD_BLEND_THRESHOLD = 0.42320248198089244
LOOKAHEAD_WEIGHT = 0.9315238778576724
SPEED_PENALTY_REFERENCE = 150.0       # Speed-scaled penalty: penalty scales with speed/this

# --- 3. Braking PD + Trail-Braking ---
BRAKE_KP = 0.18882196845205348
BRAKE_KD = 0.006388142485283601
BRAKE_TRIGGER_MARGIN = 5.226901691885811
TRAIL_BRAKE_FACTOR = 0.2520526941300226
TRAIL_BRAKE_THROTTLE = 0.04847852012829285
PARTIAL_THROTTLE_MARGIN = 8.9744913788975
PARTIAL_THROTTLE_VALUE = 0.3456381761963969

# --- 4. Shifting Dynamics ---
UPSHIFT_RPM = 18292
DOWNSHIFT_RPM = 12168
SHIFT_SLIP_DETECTION = 1.2137441115484418
SHIFT_WAIT_STEPS = 7
ENGINE_BRAKE_RPM_BOOST = 1000        # Extra RPM headroom for downshift during heavy braking

# --- 5. Steering Apex & High-Speed Damping ---
APEX_ACTIVATION_DIST = 83.68514299602667
TRACK_EDGE_DIFF_TRIGGER = 4.8527564193828185
APEX_BIAS_CAP = 0.5550670658104218
APEX_DIVISOR = 127.25517515739612
HIGH_SPEED_STEER_DAMP_SPEED = 98.80593407981057
HIGH_SPEED_STEER_DAMP_BASE = 1.5319476290747356
HIGH_SPEED_STEER_DAMP_DIVISOR = 3.614048403481607

# --- 6. Out-In-Out Racing Line ---
RACING_LINE_ACTIVATION_DIST = 120.0   # Activate when forward dist < this
RACING_LINE_DIR_DIVISOR = 100.0        # Normalizer for corner direction
RACING_LINE_DELTA_THRESHOLD = 0.5      # Min delta_dist to detect entry/exit
RACING_LINE_ENTRY_OFFSET = 0.3         # How far outside on corner entry
RACING_LINE_EXIT_OFFSET = 0.2          # How far outside on corner exit

# --- 7. Track-Specific Zones (Laguna Seca — 3 of max 4) ---
R_S = 2317.4825425883373
R_E = 2405.439847659891
R_S_L = 178.77431624521114
R_T_H_P = -0.6349772147835389
CORKSCREW_TURNIN_KP = 5.38400599491688
CORKSCREW_TURNIN_KD = 2.3155993911732278
T11S = 3205.479207020344
T11E = 3286.8866376283645
T11SL = 139.0711151028296

# --- 8. ABS PID Controller ---
ABS_SLIP_THRESHOLD = 2.633923660948679
ABS_KP = 0.16684576340101862
ABS_KI = 0.02320510063445782
ABS_KD = 0.09619036105559349
ABS_INTEGRAL_CAP = 13.015814010987384

# --- 9. TCS PID Controller ---
TCS_THRESHOLD = 16.84427138318147
TCS_KP = 0.02729977440616593
TCS_KI = 0.030819251330101004
TCS_KD = 0.002826807961750267
TCS_INTEGRAL_CAP = 4.084129194822758
TCS_INTEGRAL_DECAY = 0.9598322013488279


TRACK_FINGERPRINTS = {
    "cs": (62.02670, 62.01465),
}
FINGERPRINT_TOLERANCE = 2.0

# --- Game Engine Constants ---
GEARS = [-2.0, 0.0, 3.9, 2.9, 2.3, 1.87, 1.68, 1.54, 1.46]
TIRE_RADIUS = 0.315
TRACK_LENGTH = 3608.45

# ======================================================================
# ======================= GENERIC FALLBACK LOGIC =======================
# ======================================================================

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
    
    STEER_KP = 8.0;  STEER_KD = 1.5;  CENTERING_GAIN = 0.015
    APEX_BIAS_CAP = 0.20;  HIGH_SPEED_STEER_DAMP_BASE = 1.45
    BASE_SPEED_MULT = 5.0;  DISTANCE_SCALER = 18.5
    CORNER_EXIT_PUNCH = 50.0;  MID_CORNER_PENALTY = 10.5
    WALL_FEAR_MULT = 4.0;  LOOKAHEAD_BLEND_THRESHOLD = 0.7;  LOOKAHEAD_WEIGHT = 0.6
    BRAKE_KP = 0.06;  BRAKE_KD = 0.02;  BRAKE_TRIGGER_MARGIN = 3.0
    TRAIL_BRAKE_FACTOR = 0.1;  TRAIL_BRAKE_THROTTLE = 0.05
    PARTIAL_THROTTLE_MARGIN = 7.0;  PARTIAL_THROTTLE_VALUE = 0.6
    ABS_SLIP_THRESHOLD = 2.5;  ABS_KP = 0.2;  ABS_KI = 0.01;  ABS_KD = 0.02
    TCS_THRESHOLD = 9.7;  TCS_KP = 0.04;  TCS_KI = 0.005;  TCS_KD = 0.01
    RACING_LINE_ENTRY_OFFSET = 0.15;  RACING_LINE_EXIT_OFFSET = 0.1

# ======================================================================
# =========================== BOT LOGIC ================================
# ======================================================================

def reset_state():
    """Reset all persistent PID state between runs."""
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
        shift_stable.last_shift_step = step
        return 1

    if (step - shift_stable.last_shift_step) < SHIFT_WAIT_STEPS:
        # AGGRESSIVE ENGINE BRAKING: skip wait during heavy braking
        if R['brake'] > 0.5 and current_gear > 1 and current_rpm < DOWNSHIFT_RPM + ENGINE_BRAKE_RPM_BOOST:
            shift_stable.last_shift_step = step
            return current_gear - 1
        return current_gear

    speed_ms = S['speedX'] / 3.6
    avg_wheel_speed = (S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) / 2.0
    is_slipping = (avg_wheel_speed * TIRE_RADIUS) > (speed_ms * SHIFT_SLIP_DETECTION)

    if current_gear < 7:
        if current_rpm > UPSHIFT_RPM and not is_slipping:
            shift_stable.last_shift_step = step
            return current_gear + 1
        
    if current_gear > 1:
        if current_rpm < DOWNSHIFT_RPM:
            shift_stable.last_shift_step = step
            return current_gear - 1

    return current_gear


def calculate_steering(S, current_track):
    """PD steering with out-in-out racing line."""
    if not hasattr(calculate_steering, 'prev_error'):
        calculate_steering.prev_error = 0.0
        calculate_steering.prev_center_dist = 200.0

    # Launch stabilization
    if current_track == "cs" and S['distRaced'] < 60:
        return 0.01
    elif current_track != "cs" and S['distRaced'] < 10:
        return 0.01

    center_dist = S['track'][9]
    left_open = S['track'][13] + S['track'][14] + S['track'][15]
    right_open = S['track'][3] + S['track'][4] + S['track'][5]

    # --- OUT-IN-OUT RACING LINE ---
    delta_center = center_dist - calculate_steering.prev_center_dist
    calculate_steering.prev_center_dist = center_dist
    
    racing_line_target = 0.0
    if center_dist < RACING_LINE_ACTIVATION_DIST:
        corner_intensity = 1.0 - (center_dist / RACING_LINE_ACTIVATION_DIST)
        corner_direction = left_open - right_open
        dir_normalized = max(-1.0, min(1.0, corner_direction / RACING_LINE_DIR_DIVISOR))
        
        if delta_center < -RACING_LINE_DELTA_THRESHOLD:
            # Corner ENTRY: position on OUTSIDE
            racing_line_target = dir_normalized * RACING_LINE_ENTRY_OFFSET * corner_intensity
        elif delta_center > RACING_LINE_DELTA_THRESHOLD:
            # Corner EXIT: drift to OUTSIDE
            racing_line_target = dir_normalized * RACING_LINE_EXIT_OFFSET * corner_intensity
        # At apex: racing_line_target = 0, let apex_bias handle inside positioning

    # --- Apex bias (existing — handles inside positioning at apex) ---
    apex_bias = 0.0
    if center_dist < APEX_ACTIVATION_DIST:
        diff = left_open - right_open
        if abs(diff) > TRACK_EDGE_DIFF_TRIGGER:
            apex_bias = -APEX_BIAS_CAP * (diff / APEX_DIVISOR)
            apex_bias = max(-APEX_BIAS_CAP, min(APEX_BIAS_CAP, apex_bias))

    # PD controller with racing line offset
    target_pos = (S['trackPos'] - racing_line_target) * CENTERING_GAIN
    error = S['angle'] - target_pos + apex_bias
    d_error = error - calculate_steering.prev_error
    calculate_steering.prev_error = error

    steer = STEER_KP * error + STEER_KD * d_error

    # High-speed damping
    speed_kmh = S['speedX']
    if speed_kmh > HIGH_SPEED_STEER_DAMP_SPEED:
        scale = HIGH_SPEED_STEER_DAMP_BASE + (
            math.sqrt(speed_kmh - HIGH_SPEED_STEER_DAMP_SPEED)
            / HIGH_SPEED_STEER_DAMP_DIVISOR
        )
        steer /= scale
    
    return max(-1.0, min(1.0, steer))


def apply_abs(S, current_brake):
    """PID ABS — smooth brake modulation."""
    if current_brake <= 0.0:
        return 0.0
    speed_ms = S['speedX'] / 3.6
    if speed_ms < 5.0:
        return current_brake

    if not hasattr(apply_abs, 'slip_integral'):
        apply_abs.slip_integral = 0.0
        apply_abs.prev_slip = 0.0

    avg_wheel_rads = sum(S['wheelSpinVel']) / 4.0
    avg_wheel_speed = avg_wheel_rads * TIRE_RADIUS
    slip = speed_ms - avg_wheel_speed
    
    if slip > ABS_SLIP_THRESHOLD:
        slip_error = slip - ABS_SLIP_THRESHOLD
        apply_abs.slip_integral += slip_error
        apply_abs.slip_integral = min(apply_abs.slip_integral, ABS_INTEGRAL_CAP)
        d_slip = slip_error - apply_abs.prev_slip
        apply_abs.prev_slip = slip_error
        reduction = (ABS_KP * slip_error +
                     ABS_KI * apply_abs.slip_integral +
                     ABS_KD * d_slip)
        return max(0.0, current_brake - reduction)
    else:
        apply_abs.slip_integral *= 0.9
        apply_abs.prev_slip = 0.0
        return current_brake


def speed_control(S, R):
    """Look-ahead speed targeting with PD braking, trail-braking, and speed-scaled penalty."""
    fwd_dist = max(S['track'][8], S['track'][9], S['track'][10])

    corner_warning = min(S['track'][3], S['track'][4], S['track'][5],
                         S['track'][13], S['track'][14], S['track'][15])
    
    if corner_warning < fwd_dist * LOOKAHEAD_BLEND_THRESHOLD and fwd_dist > 10:
        effective_dist = (fwd_dist * LOOKAHEAD_WEIGHT +
                          corner_warning * (1.0 - LOOKAHEAD_WEIGHT))
    else:
        effective_dist = fwd_dist

    side_clearance = min(S['track'][4], S['track'][14])
    if side_clearance < 20:
        effective_dist = min(effective_dist, side_clearance * WALL_FEAR_MULT)

    if effective_dist == -1 or effective_dist > 200:
        effective_dist = 200.0
    elif effective_dist <= 0:
        effective_dist = 1.0 

    if not hasattr(speed_control, "prev_dist"):
        speed_control.prev_dist = effective_dist
        speed_control.prev_speed_error = 0.0
        
    delta_dist = effective_dist - speed_control.prev_dist
    speed_control.prev_dist = effective_dist

    target_speed = math.sqrt(effective_dist * DISTANCE_SCALER) * BASE_SPEED_MULT
    
    if delta_dist > 0.5 and effective_dist > 30.0:
        target_speed += CORNER_EXIT_PUNCH
    else:
        # SPEED-SCALED PENALTY: penalize more at higher speeds
        speed_factor = max(0.5, S['speedX'] / SPEED_PENALTY_REFERENCE)
        target_speed -= abs(R['steer']) * MID_CORNER_PENALTY * speed_factor

    current_speed = S['speedX']
    accel = 0.0
    brake = 0.0

    speed_error = current_speed - target_speed
    d_speed_error = speed_error - speed_control.prev_speed_error
    speed_control.prev_speed_error = speed_error

    if speed_error > BRAKE_TRIGGER_MARGIN:
        brake = BRAKE_KP * speed_error + BRAKE_KD * d_speed_error
        brake = max(0.0, min(1.0, brake))
        accel = 0.0
    elif speed_error > 0:
        brake = TRAIL_BRAKE_FACTOR * speed_error
        brake = max(0.0, min(1.0, brake))
        accel = TRAIL_BRAKE_THROTTLE
    else:
        accel = 1.0
        if current_speed > target_speed - PARTIAL_THROTTLE_MARGIN:
            accel = PARTIAL_THROTTLE_VALUE
        brake = 0.0

    if current_speed < 10 and effective_dist > 10 and brake < 0.1:
        accel = 1.0
        brake = 0.0
    
    if effective_dist >= 200:
        accel = 1.0
        brake = 0.0

    return accel, brake

    
def traction_control(S, R):
    """PID traction control."""
    if not hasattr(traction_control, 'slip_integral'):
        traction_control.slip_integral = 0.0
        traction_control.prev_slip = 0.0

    accel = R['accel']
    rear_vel = (S['wheelSpinVel'][2] + S['wheelSpinVel'][3])
    front_vel = (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])
    slip = rear_vel - front_vel
    
    if slip > TCS_THRESHOLD:
        slip_error = slip - TCS_THRESHOLD
        traction_control.slip_integral += slip_error
        traction_control.slip_integral = max(0.0,
            min(traction_control.slip_integral, TCS_INTEGRAL_CAP))
        d_slip = slip_error - traction_control.prev_slip
        reduction = (TCS_KP * slip_error +
                     TCS_KI * traction_control.slip_integral +
                     TCS_KD * d_slip)
        accel -= reduction
        accel = max(0.0, accel)
    else:
        traction_control.slip_integral *= TCS_INTEGRAL_DECAY

    traction_control.prev_slip = max(0.0, slip - TCS_THRESHOLD)
    return accel


def drive(c, step_counter):
    S = c.S.d 
    R = c.R.d 
    
    # FAST DYNAMIC TRACK PROFILER
    if not hasattr(drive, "track_profiled"):
        drive.track_profiled = False
        drive.current_track = "cs"
        drive.fp1_hist = []
        drive.fp2_hist = []
        drive.prev_turnin_error = 0.0

    if not drive.track_profiled:
        if 5 <= step_counter <= 15:
            if abs(S['track'][9] - 62.02670) > 4.0:
                drive.current_track = "unknown"
                drive.track_profiled = True
                set_generic_params()
        
        if not drive.track_profiled:
            if 5 <= step_counter <= 25:
                drive.fp1_hist.append(S['track'][9])
            elif step_counter == 26:
                drive.fp1 = sum(drive.fp1_hist) / len(drive.fp1_hist)

            if 30 <= step_counter <= 50:
                drive.fp2_hist.append(S['track'][9])
            elif step_counter == 51:
                drive.fp2 = sum(drive.fp2_hist) / len(drive.fp2_hist)
                matched_track = None
                for t_name, (expected_fp1, expected_fp2) in TRACK_FINGERPRINTS.items():
                    if expected_fp1 < 0: continue
                    if (abs(drive.fp1 - expected_fp1) < FINGERPRINT_TOLERANCE and
                        abs(drive.fp2 - expected_fp2) < FINGERPRINT_TOLERANCE):
                        matched_track = t_name
                        break
                if matched_track:
                    drive.current_track = matched_track
                else:
                    drive.current_track = "unknown"
                    set_generic_params()
                drive.track_profiled = True

    R['steer'] = calculate_steering(S, drive.current_track)
    base_accel, base_brake = speed_control(S, R)
    R['brake'] = apply_abs(S, base_brake)
    R['accel'] = base_accel
    R['accel'] = traction_control(S, R)
    
    current_dist = S['distFromStart'] % TRACK_LENGTH

    if drive.current_track == "cs":
        # Zone 1: Corkscrew brake + PD turn-in
        if R_S < current_dist < R_E:
            if S['speedX'] > R_S_L:
                R['accel'] = 0.0
                R['brake'] = 1.0
            if S['trackPos'] < R_T_H_P:
                turnin_error = R_T_H_P - S['trackPos']
                d_turnin = turnin_error - drive.prev_turnin_error
                drive.prev_turnin_error = turnin_error
                R['steer'] += CORKSCREW_TURNIN_KP * turnin_error + CORKSCREW_TURNIN_KD * d_turnin
            else:
                drive.prev_turnin_error = 0.0
        else:
            drive.prev_turnin_error = 0.0

        # Zone 2: Turn 11 brake
        if T11S < current_dist < T11E:
            if S['speedX'] > T11SL:
                R['accel'] = 0.0   
                R['brake'] = 1.0

        # Zone 3: Exit straight
        if 3300 < current_dist < 3550:
            R['accel'] = 1.0
            R['brake'] = 0.0

    R['gear'] = shift_stable(S, R, step_counter)
    
    if step_counter < 50:
        R['accel'] = 1.0
        R['brake'] = 0.0
        R['gear'] = 1
        R['clutch'] = 0.0 
    
    R['meta'] = 0 
    return


if __name__ == "__main__":
    C = Client(p=3001)
    total_steps = C.maxSteps
    for steps_left in range(C.maxSteps, 0, -1):
        step_counter = total_steps - steps_left
        C.get_servers_input()
        drive(C, step_counter)
        C.respond_to_server()
    C.shutdown()