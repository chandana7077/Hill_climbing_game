"""Tabular Q-learning agent for the hill climb game (pure Python).

The continuous game state is bucketed into a small discrete state:
    body tilt x tilt speed x forward speed x wheel contacts x slope ahead
For every state the agent stores one Q-value per action (coast, gas, brake),
an estimate of the future reward. It learns them with n-step Q-learning:
    Q(s,a) += lr * (r0 + g*r1 + ... + g^n * max Q(s_n,.) - Q(s,a))
"""
import json
import math
import random
from bisect import bisect
from pathlib import Path

from hill_env import BRAKE, COAST, GAS

ACTION_REPEAT = 4  # the agent picks an action every 4 frames (15 times per second)
N_STEPS = 8        # n-step Q-learning look-ahead (about half a second)
N_ACTIONS = 3
EXPLORE_ACTIONS = (COAST, GAS, BRAKE)
EXPLORE_WEIGHTS = (0.25, 0.55, 0.20)
# When Q-values tie (e.g. an unseen state) prefer gas, then coast, then brake.
TIE_ORDER = (GAS, COAST, BRAKE)

_deg = math.radians
ANGLE_EDGES = [_deg(d) for d in (-60, -35, -15, -5, 5, 15, 35, 60)]
ANG_VEL_EDGES = [-2.0, -0.5, 0.5, 2.0]          # rad/s
SPEED_EDGES = [-0.5, 1.0, 4.0, 8.0, 14.0, 19.0]  # m/s
SLOPE_EDGES = [_deg(d) for d in (-25, -10, 10, 25)]


def state_key(obs):
    contacts = int(obs["rear_contact"]) * 2 + int(obs["front_contact"])
    return (
        f"{bisect(ANGLE_EDGES, obs['angle'])},"
        f"{bisect(ANG_VEL_EDGES, obs['ang_vel'])},"
        f"{bisect(SPEED_EDGES, obs['speed'])},"
        f"{contacts},"
        f"{bisect(SLOPE_EDGES, obs['slope_ahead'])}"
    )


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount=0.98, seed=None):
        self.learning_rate = learning_rate
        self.discount = discount
        self.q = {}
        self.rng = random.Random(seed)

    def q_values(self, key):
        return self.q.get(key, [0.0] * N_ACTIONS)

    def best_action(self, key):
        values = self.q_values(key)
        return max(TIE_ORDER, key=lambda a: values[a])

    def act(self, key, epsilon=0.0):
        """Epsilon-greedy: explore with probability epsilon, else exploit.

        Exploration favours gas: random coasting and braking mostly
        leave the car standing still, which teaches nothing.
        """
        if epsilon > 0 and self.rng.random() < epsilon:
            return self.rng.choices(EXPLORE_ACTIONS, EXPLORE_WEIGHTS)[0]
        return self.best_action(key)

    def update(self, key, action, target):
        """Move Q(key, action) towards a target return."""
        values = self.q.setdefault(key, [0.0] * N_ACTIONS)
        values[action] += self.learning_rate * (target - values[action])

    def n_step_target(self, rewards, next_key, terminal):
        """n-step return: r0 + g*r1 + ... + g^n * max Q(next state).

        Looking several decisions ahead lets a crash penalty reach the
        choice that caused it (e.g. too much gas before a crest).
        """
        target = 0.0
        for r in reversed(rewards):
            target = r + self.discount * target
        if not terminal:
            target += self.discount ** len(rewards) * max(self.q_values(next_key))
        return target

    # ---------------- persistence ----------------
    def save(self, path, meta=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "hill-climb-q-table/1",
            "action_repeat": ACTION_REPEAT,
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "meta": meta or {},
            "q": {k: [round(v, 4) for v in vals] for k, vals in self.q.items()},
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(data.get("learning_rate", 0.1), data.get("discount", 0.98))
        agent.q = {k: list(v) for k, v in data["q"].items()}
        agent.meta = data.get("meta", {})
        return agent
