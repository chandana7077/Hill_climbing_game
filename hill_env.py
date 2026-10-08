"""Hill climb physics and game rules, with no rendering.

The same engine is used by the human game (main.py) and by the
reinforcement-learning trainer (train.py), so the AI learns exactly the game
you play. Units are pixels and seconds; PX_PER_M converts to metres.

The car is two wheels joined by a rigid axle (position-based dynamics).
The body is drawn on top of the axle, and touching the ground with the
roof means a crash.
"""
import math
import random

from pygame.math import Vector2

# ---------------- Simulation ----------------
PX_PER_M = 30.0
FRAME_DT = 1 / 60
SUBSTEPS = 4
DT = FRAME_DT / SUBSTEPS
SOLVER_ITERS = 4

# ---------------- Car ----------------
GRAVITY = 900.0
WHEEL_R = 20.0
WHEELBASE = 100.0
DRIVE_ACCEL = 520.0
BRAKE_DECEL = 1400.0
REVERSE_ACCEL = 420.0
MAX_SPEED = 650.0
MAX_REVERSE = 280.0
ROLL_FRICTION = 0.35          # fraction of rolling speed lost per second
AIR_TORQUE = 350.0            # tilt control while airborne
GROUND_TORQUE = 110.0         # gas lifts the nose a little on the ground
MAX_SPIN = 6.0                # rad/s
CONTACT_SLOP = 1.5
# Roof points (car-local: x forward, y up from the axle line). Touching
# the ground with these is a crash.
CRASH_POINTS = ((-34.0, 70.0), (8.0, 70.0))
CRASH_R = 7.0

# ---------------- Rules ----------------
FUEL_MAX = 100.0
FUEL_DRAIN = 2.5              # per second
OUT_OF_FUEL_STOP_SECONDS = 1.5

COAST, GAS, BRAKE = 0, 1, 2
ACTION_NAMES = ("coast", "gas", "brake")

# ---------------- RL reward ----------------
REWARD_CRASH = -30.0
REWARD_STUCK = -5.0
REWARD_COIN = 0.1
REWARD_FUEL = 2.0


class Terrain:
    """Endless hills: a sum of sine waves that get taller with distance."""

    FLAT_LEN = 500.0
    RAMP_LEN = 700.0
    HARDER_EVERY = 20000.0
    MAX_EXTRA_DIFFICULTY = 0.5

    def __init__(self, seed):
        rng = random.Random(seed)
        self.waves = [
            (amp, freq, rng.uniform(0, math.tau))
            for amp, freq in ((75, 0.0024), (38, 0.0058), (14, 0.016), (4, 0.045))
        ]

    def sample(self, x):
        """Return (ground y, dy/dx) at world x. Screen y points down."""
        t = x - self.FLAT_LEN
        if t <= 0:
            return 0.0, 0.0
        h = dh = 0.0
        for amp, freq, phase in self.waves:
            arg = freq * t + phase
            h += amp * math.sin(arg)
            dh += amp * freq * math.cos(arg)
        if t < self.RAMP_LEN:
            ramp, dramp = t / self.RAMP_LEN, 1 / self.RAMP_LEN
        else:
            ramp, dramp = 1.0, 0.0
        extra = t / self.HARDER_EVERY
        if extra < self.MAX_EXTRA_DIFFICULTY:
            diff, ddiff = 1.0 + extra, 1 / self.HARDER_EVERY
        else:
            diff, ddiff = 1.0 + self.MAX_EXTRA_DIFFICULTY, 0.0
        y = -h * ramp * diff
        slope = -(dh * ramp * diff + h * dramp * diff + h * ramp * ddiff)
        return y, slope

    def y(self, x):
        return self.sample(x)[0]


class Wheel:
    __slots__ = ("pos", "old", "vel", "contact", "normal", "spin", "spin_rate")

    def __init__(self, x, y):
        self.pos = Vector2(x, y)
        self.old = Vector2(x, y)
        self.vel = Vector2()
        self.contact = False
        self.normal = Vector2(0, -1)
        self.spin = 0.0
        self.spin_rate = 0.0


class Car:
    def __init__(self, x, ground_y):
        y = ground_y - WHEEL_R
        self.rear = Wheel(x - WHEELBASE / 2, y)
        self.front = Wheel(x + WHEELBASE / 2, y)
        self.wheels = (self.rear, self.front)

    @property
    def mid(self):
        return (self.rear.pos + self.front.pos) * 0.5

    @property
    def axis(self):
        d = self.front.pos - self.rear.pos
        return d / d.length() if d.length_squared() > 1e-9 else Vector2(1, 0)

    @property
    def up(self):
        a = self.axis
        return Vector2(a.y, -a.x)

    @property
    def angle(self):
        """Body angle in radians, nose-up positive."""
        a = self.axis
        return math.atan2(-a.y, a.x)

    @property
    def ang_vel(self):
        """Angular velocity in rad/s, nose-up positive."""
        return (self.front.vel - self.rear.vel).dot(self.up) / WHEELBASE

    @property
    def velocity(self):
        return (self.rear.vel + self.front.vel) * 0.5

    @property
    def grounded(self):
        return self.rear.contact or self.front.contact

    def local_to_world(self, lx, ly):
        return self.mid + self.axis * lx + self.up * ly


class Item:
    __slots__ = ("kind", "x", "y", "value")

    def __init__(self, kind, x, y, value=1):
        self.kind, self.x, self.y, self.value = kind, x, y, value


class HillClimbEnv:
    """Gym-style environment: reset() -> obs, step(action) -> (obs, reward, done, info).

    One step is one 1/60 s frame. `events` lists what happened during the
    last step (coin, diamond, fuel, flip, air, crash) so the game can play
    sounds.
    """

    START_X = 200.0
    PICKUP_RADIUS = 48.0

    def __init__(self, seed=None, max_seconds=None, stuck_seconds=8.0):
        self.max_seconds = max_seconds
        self.stuck_seconds = stuck_seconds
        self.reset(seed)

    # ---------------- lifecycle ----------------
    def reset(self, seed=None):
        self.seed = random.randrange(1 << 30) if seed is None else seed
        self.rng = random.Random(self.seed * 7919 + 1)
        self.terrain = Terrain(self.seed)
        self.car = Car(self.START_X, self.terrain.y(self.START_X))

        self.time = 0.0
        self.fuel = FUEL_MAX
        self.coins = 0
        self.diamonds = 0
        self.flips = 0
        self.done = False
        self.end_reason = None
        self.events = []

        self.best_x = self.START_X
        self.progress_mark = self.START_X
        self.last_progress_time = 0.0
        self.stopped_time = 0.0
        self.air_rotation = 0.0
        self.air_time = 0.0

        self.items = []
        self.next_coin_x = 900.0
        self.next_fuel_x = 180 * PX_PER_M
        self.fuel_spawned = 0
        self._spawn_items(self.START_X + 2000)
        return self.observe()

    @property
    def distance(self):
        """Best distance reached, in metres."""
        return (self.best_x - self.START_X) / PX_PER_M

    def step(self, action):
        if self.done:
            return self.observe(), 0.0, True, {"events": [], "reason": self.end_reason}

        self.events = []
        gas = action == GAS and self.fuel > 0
        brake = action == BRAKE
        was_grounded = self.car.grounded

        for _ in range(SUBSTEPS):
            self._substep(gas, brake)
        self.time += FRAME_DT
        self.fuel = max(0.0, self.fuel - FUEL_DRAIN * FRAME_DT)

        reward = self._update_progress()
        reward += self._update_air(was_grounded)
        reward += self._collect_items()
        reward += self._check_end()
        return self.observe(), reward, self.done, {"events": self.events, "reason": self.end_reason}

    def observe(self):
        car = self.car
        mx = car.mid.x
        return {
            "angle": car.angle,
            "ang_vel": car.ang_vel,
            "speed": car.velocity.x / PX_PER_M,
            "rear_contact": car.rear.contact,
            "front_contact": car.front.contact,
            "slope_here": math.atan(-self.terrain.sample(mx)[1]),
            "slope_ahead": math.atan(-self.terrain.sample(mx + 4 * PX_PER_M)[1]),
            "fuel": self.fuel / FUEL_MAX,
        }

    # ---------------- physics ----------------
    def _substep(self, gas, brake):
        car = self.car
        for w in car.wheels:
            w.old.update(w.pos)
            w.vel.y += GRAVITY * DT
            w.pos += w.vel * DT
            w.contact = False

        for _ in range(SOLVER_ITERS):
            self._solve_axle()
            for w in car.wheels:
                self._collide(w)

        for w in car.wheels:
            w.vel = (w.pos - w.old) / DT

        grounded = car.grounded
        for w in car.wheels:
            if not w.contact:
                if gas:
                    w.spin_rate = min(w.spin_rate + 40 * DT, MAX_SPEED / WHEEL_R)
                w.spin += w.spin_rate * DT
                continue
            n = w.normal
            t = Vector2(-n.y, n.x)  # along the ground, pointing forward
            vn = w.vel.dot(n)
            if vn > 0:  # damp bounce off the ground (soft suspension feel)
                w.vel -= n * (vn * 0.35)
            vt = w.vel.dot(t)
            if gas:
                if vt < MAX_SPEED:
                    w.vel += t * (DRIVE_ACCEL * DT)
            elif brake:
                if vt > 10:
                    w.vel -= t * min(vt, BRAKE_DECEL * DT)
                elif vt > -MAX_REVERSE:
                    w.vel -= t * (REVERSE_ACCEL * DT)
            else:
                w.vel -= t * (vt * min(1.0, ROLL_FRICTION * DT))
            w.spin_rate = w.vel.dot(t) / WHEEL_R
            w.spin += w.spin_rate * DT

        # Tilt: gas pitches the nose up, brake pitches it down (strong in the air).
        torque = 0.0
        if gas:
            torque = GROUND_TORQUE if grounded else AIR_TORQUE
        elif brake:
            torque = -(GROUND_TORQUE if grounded else AIR_TORQUE)
        up = car.up
        if torque:
            car.front.vel += up * (torque * DT)
            car.rear.vel -= up * (torque * DT)

        rel = (car.front.vel - car.rear.vel).dot(up)
        limit = MAX_SPIN * WHEELBASE
        if abs(rel) > limit:
            excess = (rel - math.copysign(limit, rel)) * 0.5
            car.front.vel -= up * excess
            car.rear.vel += up * excess

    def _solve_axle(self):
        rear, front = self.car.rear, self.car.front
        delta = front.pos - rear.pos
        d = delta.length()
        if d > 1e-6:
            corr = delta * ((d - WHEELBASE) / d * 0.5)
            rear.pos += corr
            front.pos -= corr

    def _collide(self, w):
        gy, slope = self.terrain.sample(w.pos.x)
        inv = 1.0 / math.sqrt(1.0 + slope * slope)
        dist = (gy - w.pos.y) * inv
        pen = WHEEL_R - dist
        if pen > -CONTACT_SLOP:
            n = Vector2(slope * inv, -inv)
            w.contact = True
            w.normal = n
            if pen > 0:
                w.pos += n * pen

    # ---------------- rules ----------------
    def _update_progress(self):
        x = self.car.mid.x
        reward = 0.0
        if x > self.best_x:
            reward = (x - self.best_x) / PX_PER_M
            self.best_x = x
        if x > self.progress_mark + PX_PER_M:
            self.progress_mark = x
            self.last_progress_time = self.time
        self._spawn_items(x + 2000)
        self.items = [it for it in self.items if it.x > x - 800]
        return reward

    def _update_air(self, was_grounded):
        car = self.car
        if not car.grounded:
            self.air_rotation += car.ang_vel * FRAME_DT
            self.air_time += FRAME_DT
            return 0.0
        if not was_grounded:
            flips = int((abs(self.air_rotation) + 0.6) / math.tau)
            if flips:
                self.flips += flips
                self.coins += 5 * flips
                self.events.append("flip")
            elif self.air_time > 0.8:
                self.events.append("air")
        self.air_rotation = 0.0
        self.air_time = 0.0
        return 0.0

    def _collect_items(self):
        center = self.car.local_to_world(0, 30)
        reward = 0.0
        kept = []
        for it in self.items:
            if (it.x - center.x) ** 2 + (it.y - center.y) ** 2 > self.PICKUP_RADIUS ** 2:
                kept.append(it)
                continue
            self.events.append(it.kind)
            if it.kind == "coin":
                self.coins += it.value
                reward += REWARD_COIN
            elif it.kind == "diamond":
                self.diamonds += 1
                reward += REWARD_COIN * 5
            elif it.kind == "fuel":
                self.fuel = FUEL_MAX
                reward += REWARD_FUEL
        self.items = kept
        return reward

    def _check_end(self):
        car = self.car
        for lx, ly in CRASH_POINTS:
            p = car.local_to_world(lx, ly)
            gy, slope = self.terrain.sample(p.x)
            if (gy - p.y) / math.sqrt(1 + slope * slope) < CRASH_R:
                self.events.append("crash")
                return self._end("crash", REWARD_CRASH)

        speed = car.velocity.length() / PX_PER_M
        if self.fuel <= 0:
            self.stopped_time = self.stopped_time + FRAME_DT if speed < 0.4 else 0.0
            if self.stopped_time > OUT_OF_FUEL_STOP_SECONDS:
                return self._end("fuel", 0.0)
        if self.stuck_seconds and self.time - self.last_progress_time > self.stuck_seconds:
            return self._end("stuck", REWARD_STUCK)
        if self.max_seconds and self.time >= self.max_seconds:
            return self._end("time", 0.0)
        return 0.0

    def _end(self, reason, reward):
        self.done = True
        self.end_reason = reason
        return reward

    def _spawn_items(self, until_x):
        while self.next_coin_x < until_x:
            if self.rng.random() < 0.12:
                x = self.next_coin_x
                self.items.append(Item("diamond", x, self.terrain.y(x) - 45))
                self.next_coin_x += self.rng.uniform(500, 1000)
                continue
            count = self.rng.randint(4, 8)
            value = 1 + int((self.next_coin_x - self.START_X) / PX_PER_M // 300)
            for i in range(count):
                x = self.next_coin_x + i * 40
                self.items.append(Item("coin", x, self.terrain.y(x) - 42, value))
            self.next_coin_x += count * 40 + self.rng.uniform(400, 900)
        while self.next_fuel_x < until_x:
            x = self.next_fuel_x
            self.items.append(Item("fuel", x, self.terrain.y(x) - 45))
            self.fuel_spawned += 1
            self.next_fuel_x += (180 + 40 * self.fuel_spawned) * PX_PER_M
