"""Train the Q-learning agent to drive.

    python train.py                  # settings from .env
    python train.py --episodes 500   # override the episode count
    python train.py --resume         # continue from the saved model

Each episode is played on a new random track. Every EVAL_EVERY episodes the
greedy policy is tested on fixed tracks, and the best one is saved to
MODEL_PATH.
"""
import argparse
import csv
import random
import time
from collections import deque

import config
from agent import ACTION_REPEAT, N_STEPS, QLearningAgent, state_key
from hill_env import HillClimbEnv

EVAL_EVERY = 100
EVAL_SEEDS = (1001, 1002, 1003, 1004, 1005)


def run_episode(env, agent, seed, epsilon=0.0, learn=True):
    obs = env.reset(seed)
    key = state_key(obs)
    total_reward = 0.0
    done = False
    history = deque()  # (state, action, reward) of the last N_STEPS decisions
    while not done:
        action = agent.act(key, epsilon)
        reward = 0.0
        for _ in range(ACTION_REPEAT):
            obs, r, done, info = env.step(action)
            reward += r
            if done:
                break
        key_next = state_key(obs)
        total_reward += reward
        if learn:
            history.append((key, action, reward))
            # Hitting the time limit is not a real ending, so keep bootstrapping.
            terminal = done and info["reason"] != "time"
            if len(history) == N_STEPS:
                _learn_oldest(agent, history, key_next, terminal)
            if done:
                while history:
                    _learn_oldest(agent, history, key_next, terminal)
        key = key_next
    return total_reward


def _learn_oldest(agent, history, key_next, terminal):
    s, a, _ = history[0]
    target = agent.n_step_target([r for _, _, r in history], key_next, terminal)
    agent.update(s, a, target)
    history.popleft()


def evaluate(agent):
    env = HillClimbEnv(max_seconds=config.MAX_EPISODE_SECONDS)
    distances = []
    for seed in EVAL_SEEDS:
        run_episode(env, agent, seed, epsilon=0.0, learn=False)
        distances.append(env.distance)
    return sum(distances) / len(distances)


def main():
    parser = argparse.ArgumentParser(description="Train the hill climb Q-learning agent")
    parser.add_argument("--episodes", type=int, default=config.TRAIN_EPISODES)
    parser.add_argument("--resume", action="store_true", help="continue from MODEL_PATH")
    args = parser.parse_args()

    if args.resume and config.MODEL_PATH.exists():
        agent = QLearningAgent.load(config.MODEL_PATH)
        agent.learning_rate = config.LEARNING_RATE
        agent.discount = config.DISCOUNT
        best_eval = agent.meta.get("eval_distance", 0.0)
        print(f"Resuming from {config.MODEL_PATH} ({len(agent.q)} states, eval {best_eval:.0f} m)")
    else:
        agent = QLearningAgent(config.LEARNING_RATE, config.DISCOUNT, seed=config.TRAIN_SEED)
        best_eval = float("-inf")

    rng = random.Random(config.TRAIN_SEED)
    env = HillClimbEnv(max_seconds=config.MAX_EPISODE_SECONDS)
    config.TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(config.TRAINING_LOG, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["episode", "epsilon", "distance_m", "reward", "end_reason", "coins", "states", "eval_distance_m"])

    print(f"Training for {args.episodes} episodes (Ctrl+C to stop early; the best model is kept)")
    start = time.time()
    recent = []
    try:
        for episode in range(1, args.episodes + 1):
            frac = min(1.0, (episode - 1) / max(1, config.EPSILON_DECAY_EPISODES))
            epsilon = config.EPSILON_START + (config.EPSILON_END - config.EPSILON_START) * frac
            reward = run_episode(env, agent, rng.randrange(1 << 30), epsilon)
            recent.append(env.distance)

            eval_distance = ""
            if episode % EVAL_EVERY == 0 or episode == args.episodes:
                eval_distance = evaluate(agent)
                saved = ""
                if eval_distance > best_eval:
                    best_eval = eval_distance
                    agent.save(config.MODEL_PATH, {"episodes": episode, "eval_distance": round(best_eval, 1)})
                    saved = "  <- saved"
                avg = sum(recent) / len(recent)
                recent.clear()
                print(
                    f"ep {episode:5d} | eps {epsilon:.2f} | avg {avg:6.0f} m | "
                    f"eval {eval_distance:6.0f} m | states {len(agent.q):5d} | "
                    f"{time.time() - start:5.0f}s{saved}"
                )
                eval_distance = round(eval_distance, 1)

            log.writerow([episode, round(epsilon, 3), round(env.distance, 1), round(reward, 1),
                          env.end_reason, env.coins, len(agent.q), eval_distance])
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        log_file.close()

    print(f"Best evaluation distance: {best_eval:.0f} m. Model: {config.MODEL_PATH}")
    print("Watch it drive: python main.py  ->  Watch AI Drive")


if __name__ == "__main__":
    main()
