# Hill Climb: Hand Gestures + Reinforcement Learning

A 2D hill-climb driving game in Python (pygame) that you can play three ways:

| Mode | How it drives |
|------|---------------|
| **1. Play with Hand Gestures** | Your webcam watches your hand (OpenCV + MediaPipe) |
| **2. Play with Keyboard** | Arrow keys / A-D |
| **3. Watch AI Drive** | A **Q-learning** agent that taught itself to drive |

Collect coins and diamonds, grab fuel cans before the tank runs dry, do flips for bonus coins, and don't land on your roof.

## Hand gestures

| Gesture | Action |
|---------|--------|
| ✋ Open palm (3+ fingers up) | **Gas** |
| ✊ Fist | **Brake / reverse** |
| ✌️ Two fingers, or no hand | Coast |

In the air, gas tilts the car back and brake tilts it forward, so you can steer flips and landings.
A live camera preview with your hand skeleton is shown in the corner.

## Setup

```bash
git clone https://github.com/<your-username>/hill-game.git
cd hill-game
pip install -r requirements.txt
copy .env.example .env        # Windows   (macOS/Linux: cp .env.example .env)
python main.py
```

- Python 3.9 – 3.13
- The keyboard and AI modes only need `pygame`. OpenCV and MediaPipe are only for hand gestures.
- The MediaPipe hand model (~8 MB) downloads automatically the first time you pick gesture mode.

## How the AI learns (reinforcement learning)

The agent uses **tabular Q-learning** in pure Python, with no GPU and no deep-learning libraries.

1. **State.** 15 times per second, the agent reads the game and buckets it into a small discrete state:
   car tilt · tilt speed · forward speed · which wheels touch the ground · slope of the hill ahead.
2. **Action.** It picks **gas**, **coast** or **brake**.
3. **Reward.** It gets **+1 per metre** of new ground, small bonuses for coins and fuel,
   **−30 for crashing** (roof touches the ground) and −5 for getting stuck.
4. **Learning.** It keeps a table of Q-values (expected future reward for each action in each state)
   and updates them with **n-step Q-learning**:

   `Q(s,a) ← Q(s,a) + α · (r₀ + γr₁ + … + γⁿ·max Q(sₙ,·) − Q(s,a))`

   Looking 8 decisions ahead lets a crash penalty reach the choice that caused it,
   like flooring it just before a crest.
5. **Exploration.** It starts mostly random (ε-greedy, ε = 1.0 → 0.02) and gradually trusts what it has learned.

In **Watch AI** mode, the panel in the corner shows the agent's live Q-values for each action,
so you can see what it is "thinking".

### Train it yourself

```bash
python train.py                  # uses TRAIN_EPISODES etc. from .env
python train.py --episodes 500   # quick run
python train.py --resume         # keep improving the saved model
```

Every 100 episodes the agent is tested on 5 fixed tracks, and the best version is saved to
`models/q_table.json`. A per-episode log is written to `models/training_log.csv`.
A trained model is already included, so **Watch AI Drive** works straight after cloning.

## Project structure

```
hill game/
├── main.py            # pygame game: menu, rendering, sound, 3 play modes
├── hill_env.py        # physics + game rules (no graphics), shared by the game and the AI
├── agent.py           # Q-learning agent: state buckets, Q-table, save/load
├── train.py           # training loop
├── gestures.py        # webcam hand tracking -> gas / brake / coast
├── config.py          # loads settings from .env
├── .env.example       # all settings with defaults (copy to .env)
├── requirements.txt
├── models/
│   └── q_table.json   # trained AI
└── assets/
    ├── images/        # car, wheel, coin, diamond, fuel, flag
    └── sounds/
```

## Settings (`.env`)

All settings are optional. See `.env.example` for the full list. The most useful ones:

| Variable | Default | What it does |
|----------|---------|--------------|
| `CAMERA_INDEX` | `0` | Webcam to use (try `1` for an external camera) |
| `GESTURE_MIN_CONFIDENCE` | `0.6` | Hand-detection strictness (0–1) |
| `GESTURE_SMOOTHING_FRAMES` | `5` | Steadier vs. faster gesture response |
| `SOUND_ENABLED` | `true` | Turn sound on/off |
| `TRAIN_EPISODES` | `2000` | Training length |
| `LEARNING_RATE` / `DISCOUNT` | `0.1` / `0.98` | Q-learning α and γ |

`.env` is in `.gitignore`. Commit `.env.example`, never your personal `.env`.

## Controls

| Key | Action |
|-----|--------|
| → / D | Gas |
| ← / A | Brake / reverse |
| P / Esc | Pause |
| R | Drive again (after game over) |
| M | Back to menu (paused / game over) |
