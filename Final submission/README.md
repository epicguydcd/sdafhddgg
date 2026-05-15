# TORCS Racing Bot — IBM AI Racing League 2026

## Best Standing Lap Time: **1:20.09** (Laguna Seca Corkscrew)
## Best Flying Lap Time: **1:17.13**

---

## Files Overview

### Final Bot

| File | Lap Time | Description |
|------|----------|-------------|
| **`final_bot_80_09s.py`** | **80.09s ★** | **Final bot — Deep CMA-ES + Coordinate Descent optimised with off-track recovery** |

### Previous Version Bots (Progression)

| # | File | Lap Time | Description |
|---|------|----------|-------------|
| 1 | `bot_v1_original_83s.py` | ~83s | Original bot — first working version with Optuna TPE-tuned parameters |
| 2 | `bot_v2_first_sub81_80_91s.py` | 80.91s | First sub-81s — refined speed/braking params, ABS PID tuned |
| 3 | `bot_v3_racing_line_80_37s.py` | 80.37s | Racing line logic added — dynamic entry/exit offsets |
| 4 | `bot_v4_structural_fix_80_35s.py` | 80.35s | Manual code fixes — launch optimisation + finish straight |
| 5 | `bot_v5_cma_optimised_80_18s.py` | 80.18s | CMA-ES optimised — tight ±10% bounds around v4 params |
| 6 | `bot_v6_coordinate_descent_80_09s.py` | 80.09s | Targeted coordinate descent — final parameter refinement |
| 7 | **`final_bot_80_09s.py`** | **80.09s ★** | **Final — v6 params + off-track recovery system** |

### Optimisers

| File | Method | Produces |
|------|--------|----------|
| `optimiser_phases_1_to_4_tpe.py` | Optuna TPE, wide search | `bot_v1_original_83s.py` |
| `optimiser_phase_5_tpe.py` | Optuna TPE, focused speed/braking | `bot_v2_first_sub81_80_91s.py` |
| `optimiser_phase_6_tpe.py` | Optuna TPE, racing line params | `bot_v3_racing_line_80_37s.py` |
| `optimiser_cma_es_final.py` | Optuna CMA-ES, tight ±10% | `bot_v5_cma_optimised_80_18s.py` |

### Support

| File | Purpose |
|------|---------|
| `torcs_jm_par.py` | TORCS UDP client — handles all communication with the simulator |

---

## Optimisation Pipeline

```
optimiser_phases_1_to_4_tpe.py ──► bot_v1_original_83s.py              (~83s)
              │                            │
              ▼                            ▼ params refined
optimiser_phase_5_tpe.py ────────► bot_v2_first_sub81_80_91s.py        (80.91s)
              │                            │
              ▼                            ▼ params refined
optimiser_phase_6_tpe.py ────────► bot_v3_racing_line_80_37s.py        (80.37s)
                                           │
                                           ▼ manual code changes
                                   bot_v4_structural_fix_80_35s.py     (80.35s)
                                           │
                                           ▼ CMA-ES optimisation
optimiser_cma_es_final.py ───────► bot_v5_cma_optimised_80_18s.py      (80.18s)
                                           │
                                           ▼ coordinate descent
                                   bot_v6_coordinate_descent_80_09s.py (80.09s)
                                           │
                                           ▼ off-track recovery added
                                   final_bot_80_09s.py            ★    (80.09s)
```

---

## How to Run

```bash
# 1. Start TORCS with the Laguna Seca Corkscrew track (port 3001)
# 2. Run the final bot:
python final_bot_80_09s.py

# To re-run CMA-ES optimisation (~45 min for 200 trials):
pip install optuna cmaes
python optimiser_cma_es_final.py
```

---

## Architecture: Reactive PD/PID Controller

Fully reactive — no pre-computed speed maps, no neural networks, no multi-lap learning. Every decision is made in real-time based on 19 track edge sensors (distance measurements at angles from −90° to +90°).

### Core Systems

1. **Steering (PD Controller)** — Calculates steering angle from track position error and its derivative. Includes apex bias for smoother cornering and a racing line system that dynamically shifts the target position on corner entry and exit.

2. **Speed Control (PD Controller)** — Determines target speed from forward sensor distance using `target = sqrt(dist × SCALER) × MULT`. Adds a corner exit punch when sensors detect opening road, and applies a mid-corner penalty scaled by current steering angle.

3. **Braking (PD + ABS PID)** — Triggers braking when speed exceeds target by a configurable margin. Trail braking applies light brake pressure near corners. The ABS PID monitors wheel slip and modulates brake pressure to prevent lock-up.

4. **Traction Control (PID)** — Monitors rear vs front wheel spin differential and reduces throttle during wheelspin to prevent loss of traction on corner exits.

5. **Gear Shifting** — RPM-based upshift/downshift with wheel slip detection to prevent upshifts during wheelspin, and emergency engine braking downshifts during heavy braking.

6. **Off-Track Recovery** — Detects when the car leaves the track surface or gets stuck. Manages reversing maneuvers when stuck, and applies controlled steering and throttle to re-enter the track safely.

---

## Optimisation Approach

### Phase 1–4: Wide Exploration (TPE)
Used Optuna's Tree-structured Parzen Estimator over 300+ trials to find a working parameter set across all controller subsystems. This established the baseline at ~83s.

### Phase 5: Focused Refinement (TPE)
Narrowed the search space to speed and braking parameters. Key gains came from higher target speeds combined with stronger ABS-enabled braking — brake harder but later.

### Phase 6: Racing Line (TPE)
Introduced the racing line system (dynamic entry/exit position offsets) and optimised its parameters. Also tuned the speed-scaled mid-corner penalty.

### Phase 7: Manual Structural Fixes
Two code-level changes without any optimiser:
- **Finish straight override** — prevented false braking on the main straight
- **Smarter launch** — reduced gear-1 lock time from 50 to 15 steps for faster acceleration

### Phase 8: CMA-ES Fine-Tuning
Switched from TPE to CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with tight ±10% bounds. CMA-ES models parameter correlations and makes precise micro-adjustments, avoiding the instability seen with wide TPE searches.

### Phase 9: Coordinate Descent
Final targeted optimisation of individual parameters one at a time, squeezing the last 0.09s.

---

## Dependencies

```
pip install optuna cmaes
```

- Python 3.10+
- TORCS with server running on port 3001
