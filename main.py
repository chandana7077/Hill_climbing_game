"""Hill Climb: drive with your keyboard, your hand, or watch the AI.

    python main.py

Menu: 1 = Hand Gestures, 2 = Keyboard, 3 = Watch AI Drive
Keyboard: RIGHT/D = gas, LEFT/A = brake, P/ESC = pause
"""
import json
import math
import sys

import pygame

import config
from agent import ACTION_REPEAT, QLearningAgent, state_key
from gestures import PREVIEW_SIZE, GestureController, GestureError
from hill_env import (
    BRAKE, COAST, FUEL_MAX, GAS, PX_PER_M, WHEEL_R, WHEELBASE, HillClimbEnv,
)

W, H = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
IMAGES = config.BASE_DIR / "assets" / "images"
SOUNDS = config.BASE_DIR / "assets" / "sounds"

MODE_GESTURE, MODE_KEYBOARD, MODE_AI = "gesture", "keyboard", "ai"
MENU_ITEMS = (
    (MODE_GESTURE, "Play with Hand Gestures"),
    (MODE_KEYBOARD, "Play with Keyboard"),
    (MODE_AI, "Watch AI Drive"),
)
MODE_TITLES = {MODE_GESTURE: "HAND", MODE_KEYBOARD: "KEYBOARD", MODE_AI: "AI"}
END_MESSAGES = {"crash": "DRIVER DOWN!", "fuel": "OUT OF FUEL", "stuck": "STUCK!", "time": "TIME UP"}

# Car sprite geometry, measured on assets/images/car.png (1024x1024).
CAR_CROP = pygame.Rect(59, 265, 850, 423)
CAR_REAR_WHEEL = (252, 640)
CAR_FRONT_WHEEL = (729, 640)
WHEEL_CROP = pygame.Rect(512 - 330, 456 - 330, 660, 660)

SKY_TOP, SKY_BOTTOM = (92, 170, 235), (200, 232, 250)
HILL_FAR, HILL_NEAR = (150, 200, 170), (110, 170, 120)
GRASS, GRASS_LIGHT = (70, 160, 50), (120, 200, 70)
DIRT, DIRT_DARK = (150, 100, 60), (115, 75, 45)
WHITE, BLACK, PANEL = (255, 255, 255), (20, 20, 20), (0, 0, 0, 140)
GREEN, YELLOW, RED, BLUE = (70, 200, 80), (245, 200, 50), (225, 60, 50), (60, 140, 230)


# ---------------- assets ----------------
def load_image(name, crop=None, size=None, height=None):
    img = pygame.image.load(str(IMAGES / name)).convert_alpha()
    img = img.subsurface(crop or img.get_bounding_rect(min_alpha=20)).copy()
    if height:
        size = (round(img.get_width() * height / img.get_height()), height)
    return pygame.transform.smoothscale(img, size) if size else img


class Sounds:
    def __init__(self):
        self.enabled = False
        self.sounds = {}
        self.engine_on = False
        if not config.SOUND_ENABLED:
            return
        try:
            pygame.mixer.init()
        except pygame.error as exc:
            print(f"[sound] disabled: {exc}")
            return
        self.enabled = True
        for name in ("coin", "diamond", "fuel", "engine", "crash", "flip_bonus", "jump"):
            try:
                snd = pygame.mixer.Sound(str(SOUNDS / f"{name}.mpeg"))
                snd.set_volume(config.SOUND_VOLUME * (0.4 if name == "engine" else 1.0))
                self.sounds[name] = snd
            except (pygame.error, FileNotFoundError) as exc:
                print(f"[sound] could not load {name}: {exc}")
        self.engine_channel = pygame.mixer.Channel(0) if pygame.mixer.get_num_channels() else None

    def play(self, name):
        snd = self.sounds.get(name)
        if snd:
            snd.play()

    def engine(self, on):
        if not self.enabled or "engine" not in self.sounds or on == self.engine_on:
            return
        self.engine_on = on
        if on:
            self.engine_channel.play(self.sounds["engine"], loops=-1)
        else:
            self.engine_channel.stop()


def load_save():
    try:
        data = json.loads(config.SAVE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("best", {})
    data.setdefault("total_coins", 0)
    data.setdefault("total_diamonds", 0)
    return data


def write_save(data):
    try:
        config.SAVE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[save] could not write {config.SAVE_FILE}: {exc}")


# ---------------- game ----------------
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Hill Climb - Hand Gestures + Reinforcement Learning")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("arial", 54, bold=True)
        self.font_mid = pygame.font.SysFont("arial", 30, bold=True)
        self.font = pygame.font.SysFont("arial", 20, bold=True)
        self.font_small = pygame.font.SysFont("arial", 15)
        self.sounds = Sounds()
        self.save = load_save()
        self._load_sprites()
        self.sky = self._make_sky()

        self.state = "menu"
        self.menu_index = 0
        self.message = ""
        self.mode = None
        self.env = None
        self.gestures = None
        self.agent = None
        self.ai_action = COAST
        self.ai_q = (0.0, 0.0, 0.0)
        self.frame = 0
        self.popups = []
        self.over_timer = 0.0
        self.new_record = False

    def _load_sprites(self):
        scale = WHEELBASE / (CAR_FRONT_WHEEL[0] - CAR_REAR_WHEEL[0])
        car = load_image("car.png", crop=CAR_CROP)
        self.car_img = pygame.transform.smoothscale(
            car, (round(CAR_CROP.w * scale), round(CAR_CROP.h * scale)))
        # Offset from the axle midpoint to the sprite centre, in car-local units (x fwd, y up).
        axle_x = (CAR_REAR_WHEEL[0] + CAR_FRONT_WHEEL[0]) / 2 - CAR_CROP.x
        axle_y = CAR_REAR_WHEEL[1] - CAR_CROP.y
        self.car_offset = ((CAR_CROP.w / 2 - axle_x) * scale, (axle_y - CAR_CROP.h / 2) * scale)
        d = round(WHEEL_R * 2 + 2)
        self.wheel_img = load_image("car_wheel.png", crop=WHEEL_CROP, size=(d, d))
        self.coin_img = load_image("coin.png", size=(28, 28))
        self.diamond_img = load_image("diamond.png", height=28)
        self.fuel_img = load_image("fuel.png", height=40)
        self.fuel_icon = load_image("fuel.png", height=26)
        self.coin_icon = load_image("coin.png", size=(24, 24))
        self.flag_img = load_image("flag.png", height=64)

    def _make_sky(self):
        sky = pygame.Surface((W, H))
        for y in range(H):
            t = y / H
            sky.fill([round(a + (b - a) * t) for a, b in zip(SKY_TOP, SKY_BOTTOM)], (0, y, W, 1))
        return sky

    # ---------------- flow ----------------
    def run(self):
        while True:
            dt = self.clock.tick(config.FPS) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                self.handle_event(event)
            if self.state == "play":
                self.update_play()
            elif self.state == "over":
                self.over_timer += dt
                if self.mode == MODE_AI and self.over_timer > 3:
                    self.start_run()
            self.draw()
            pygame.display.flip()

    def quit(self):
        self.sounds.engine(False)
        if self.gestures:
            self.gestures.stop()
        write_save(self.save)
        pygame.quit()
        sys.exit()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            key = event.key
            if self.state == "menu":
                if key in (pygame.K_UP, pygame.K_w):
                    self.menu_index = (self.menu_index - 1) % len(MENU_ITEMS)
                elif key in (pygame.K_DOWN, pygame.K_s):
                    self.menu_index = (self.menu_index + 1) % len(MENU_ITEMS)
                elif key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.select_mode(MENU_ITEMS[self.menu_index][0])
                elif pygame.K_1 <= key <= pygame.K_3:
                    self.select_mode(MENU_ITEMS[key - pygame.K_1][0])
                elif key == pygame.K_ESCAPE:
                    self.quit()
            elif self.state == "play":
                if key in (pygame.K_ESCAPE, pygame.K_p):
                    self.state = "paused"
                    self.sounds.engine(False)
            elif self.state == "paused":
                if key in (pygame.K_ESCAPE, pygame.K_p, pygame.K_RETURN):
                    self.state = "play"
                elif key in (pygame.K_m, pygame.K_q):
                    self.to_menu()
            elif self.state == "over":
                if key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE):
                    self.start_run()
                elif key in (pygame.K_ESCAPE, pygame.K_m):
                    self.to_menu()
        elif event.type == pygame.MOUSEMOTION and self.state == "menu":
            for i, rect in enumerate(self.menu_rects()):
                if rect.collidepoint(event.pos):
                    self.menu_index = i
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.state == "menu":
            for i, rect in enumerate(self.menu_rects()):
                if rect.collidepoint(event.pos):
                    self.select_mode(MENU_ITEMS[i][0])

    def select_mode(self, mode):
        self.message = ""
        if mode == MODE_AI:
            if not config.MODEL_PATH.exists():
                self.message = "No trained AI yet. Run:  python train.py"
                return
            try:
                self.agent = QLearningAgent.load(config.MODEL_PATH)
            except (OSError, ValueError, KeyError) as exc:
                self.message = f"Could not load AI model: {exc}"
                return
        if mode == MODE_GESTURE:
            self.draw_center_message("Starting camera...", "Show your hand to the webcam")
            pygame.display.flip()
            self.gestures = GestureController(
                config.CAMERA_INDEX, config.GESTURE_MIN_CONFIDENCE, config.GESTURE_SMOOTHING_FRAMES,
                config.HAND_MODEL_PATH, config.SHOW_CAMERA_PREVIEW)
            try:
                self.gestures.start()
            except GestureError as exc:
                self.gestures = None
                self.message = str(exc)
                return
        self.mode = mode
        self.start_run()

    def to_menu(self):
        self.sounds.engine(False)
        if self.gestures:
            self.gestures.stop()
            self.gestures = None
        self.state = "menu"

    def start_run(self):
        self.env = HillClimbEnv()
        self.cam = pygame.Vector2(self.env.car.mid.x - W * 0.35, self.env.car.mid.y - H * 0.6)
        self.frame = 0
        self.popups = []
        self.over_timer = 0.0
        self.new_record = False
        self.state = "play"

    # ---------------- update ----------------
    def current_action(self):
        if self.mode == MODE_AI:
            if self.frame % ACTION_REPEAT == 0:
                key = state_key(self.env.observe())
                self.ai_q = tuple(self.agent.q_values(key))
                self.ai_action = self.agent.best_action(key)
            return self.ai_action
        if self.mode == MODE_GESTURE:
            return self.gestures.action
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return GAS
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return BRAKE
        return COAST

    def update_play(self):
        env = self.env
        action = self.current_action()
        env.step(action)
        self.frame += 1
        self.sounds.engine(action == GAS and env.fuel > 0)

        for ev in env.events:
            if ev in ("coin", "diamond", "fuel"):
                self.sounds.play(ev)
            elif ev == "flip":
                self.sounds.play("flip_bonus")
                self.popup("FLIP! +5 coins", YELLOW)
            elif ev == "air":
                self.sounds.play("jump")
                self.popup("AIR TIME!", WHITE)
            elif ev == "crash":
                self.sounds.play("crash")
        self.popups = [(t, c, life - 1) for t, c, life in self.popups if life > 1]

        mid = env.car.mid
        target = pygame.Vector2(mid.x - W * 0.35 + env.car.velocity.x * 0.25, mid.y - H * 0.6)
        self.cam += (target - self.cam) * 0.08

        if env.done:
            self.finish_run()

    def finish_run(self):
        env = self.env
        self.state = "over"
        self.sounds.engine(False)
        best = self.save["best"].get(self.mode, 0)
        if env.distance > best:
            self.save["best"][self.mode] = round(env.distance, 1)
            self.new_record = best > 0
        if self.mode != MODE_AI:
            self.save["total_coins"] += env.coins
            self.save["total_diamonds"] += env.diamonds
        write_save(self.save)

    def popup(self, text, color):
        self.popups.append((text, color, 90))

    # ---------------- drawing ----------------
    def to_screen(self, p):
        return (p[0] - self.cam.x, p[1] - self.cam.y)

    def draw(self):
        if self.state == "menu":
            self.draw_menu()
            return
        self.draw_world()
        self.draw_hud()
        if self.state == "paused":
            self.draw_overlay("PAUSED", "ESC / P = resume      M = menu")
        elif self.state == "over":
            env = self.env
            title = END_MESSAGES.get(env.end_reason, "GAME OVER")
            line = f"{env.distance:.0f} m   |   {env.coins} coins   |   {env.flips} flips"
            if self.new_record:
                line = "NEW RECORD!   " + line
            hint = "Restarting..." if self.mode == MODE_AI else "R = drive again      ESC = menu"
            self.draw_overlay(title, line, hint)

    def draw_background(self, cam_x, cam_y):
        self.screen.blit(self.sky, (0, 0))
        for i in range(6):
            cx = (i * 260 - cam_x * 0.1) % (W + 300) - 150
            cy = 60 + (i * 37) % 90
            for dx, dy, r in ((0, 0, 26), (24, -10, 30), (50, 0, 24), (24, 8, 24)):
                pygame.draw.circle(self.screen, (245, 250, 255), (cx + dx, cy + dy), r)
        shift = max(-80, min(80, -cam_y * 0.05))
        for factor, base, color in ((0.2, 0.52, HILL_FAR), (0.4, 0.66, HILL_NEAR)):
            pts = [(0, H)]
            for sx in range(0, W + 21, 20):
                wx = sx + cam_x * factor
                y = H * base + shift + 45 * math.sin(wx * 0.003 + factor * 9) + 22 * math.sin(wx * 0.009)
                pts.append((sx, y))
            pts.append((W, H))
            pygame.draw.polygon(self.screen, color, pts)

    def draw_world(self):
        env, cam = self.env, self.cam
        self.draw_background(cam.x, cam.y)

        top = []
        for sx in range(-10, W + 16, 6):
            top.append((sx, env.terrain.y(cam.x + sx) - cam.y))
        pygame.draw.polygon(self.screen, DIRT, top + [(W + 16, H), (-10, H)])
        pygame.draw.polygon(self.screen, DIRT_DARK,
                            [(x, y + 40) for x, y in top] + [(W + 16, H), (-10, H)])
        pygame.draw.polygon(self.screen, GRASS, top + [(x, y + 14) for x, y in reversed(top)])
        pygame.draw.lines(self.screen, GRASS_LIGHT, False, top, 4)

        # distance flags every 100 m, plus the record for this mode
        first = int((cam.x - env.START_X) / PX_PER_M // 100)
        for k in range(max(1, first), first + W // int(100 * PX_PER_M) + 3):
            self.draw_flag(env.START_X + k * 100 * PX_PER_M, f"{k * 100} m")
        best = self.save["best"].get(self.mode, 0)
        if best > 0:
            self.draw_flag(env.START_X + best * PX_PER_M, f"BEST {best:.0f} m", YELLOW)

        t = self.frame / 60
        for it in env.items:
            sx, sy = self.to_screen((it.x, it.y))
            if -40 < sx < W + 40:
                if it.kind == "coin":
                    img, sy = self.coin_img, sy + 3 * math.sin(t * 4 + it.x * 0.05)
                else:
                    img = self.diamond_img if it.kind == "diamond" else self.fuel_img
                self.screen.blit(img, img.get_rect(center=(sx, sy)))

        self.draw_car()

    def draw_flag(self, wx, label, color=WHITE):
        sx, sy = self.to_screen((wx, self.env.terrain.y(wx)))
        if -60 < sx < W + 60:
            rect = self.flag_img.get_rect(midbottom=(sx, sy + 4))
            self.screen.blit(self.flag_img, rect)
            self.text(label, self.font_small, color, midtop=(sx, rect.top - 18), shadow=True)

    def draw_car(self):
        car = self.env.car
        deg = math.degrees(car.angle)
        body = pygame.transform.rotozoom(self.car_img, deg, 1)
        center = self.to_screen(car.local_to_world(*self.car_offset))
        self.screen.blit(body, body.get_rect(center=center))
        for w in car.wheels:
            img = pygame.transform.rotozoom(self.wheel_img, -math.degrees(w.spin), 1)
            self.screen.blit(img, img.get_rect(center=self.to_screen(w.pos)))

    def draw_hud(self):
        env = self.env
        # fuel
        panel = pygame.Surface((250, 112), pygame.SRCALPHA)
        panel.fill(PANEL)
        self.screen.blit(panel, (10, 10))
        self.screen.blit(self.fuel_icon, (20, 20))
        frac = env.fuel / FUEL_MAX
        color = GREEN if frac > 0.5 else YELLOW if frac > 0.25 else RED
        pygame.draw.rect(self.screen, (60, 60, 60), (52, 24, 196, 18), border_radius=9)
        if frac > 0 and (frac > 0.2 or self.frame // 15 % 2):
            pygame.draw.rect(self.screen, color, (52, 24, round(196 * frac), 18), border_radius=9)
        self.screen.blit(self.coin_icon, (20, 56))
        self.text(f"{env.coins}", self.font, WHITE, topleft=(52, 56))
        self.screen.blit(self.diamond_img, (130, 54))
        self.text(f"{env.diamonds}", self.font, WHITE, topleft=(162, 56))
        kmh = env.car.velocity.length() / PX_PER_M * 3.6
        self.text(f"{kmh:3.0f} km/h", self.font_small, WHITE, topleft=(20, 92))
        self.text(f"flips {env.flips}", self.font_small, WHITE, topleft=(130, 92))

        # distance
        self.text(f"{env.distance:.0f} m", self.font_mid, WHITE, midtop=(W // 2, 10), shadow=True)
        best = self.save["best"].get(self.mode, 0)
        self.text(f"best {best:.0f} m  ·  {MODE_TITLES[self.mode]}", self.font_small, WHITE,
                  midtop=(W // 2, 48), shadow=True)

        if self.state == "play" and env.fuel <= 0:
            self.text("OUT OF FUEL", self.font_mid, RED, center=(W // 2, 110), shadow=True)
        for i, (txt, col, life) in enumerate(self.popups):
            self.text(txt, self.font_mid, col, center=(W // 2, 150 + i * 36 - (90 - life) * 0.4), shadow=True)

        if self.mode == MODE_GESTURE and self.gestures:
            self.draw_gesture_panel()
        elif self.mode == MODE_AI:
            self.draw_ai_panel()
        else:
            self.text("RIGHT = gas   LEFT = brake   P = pause", self.font_small, WHITE,
                      bottomright=(W - 12, H - 10), shadow=True)

    def draw_gesture_panel(self):
        g = self.gestures
        pw, ph = PREVIEW_SIZE
        x, y = W - pw - 12, H - ph - 42
        frame = g.get_preview()
        if frame:
            self.screen.blit(pygame.image.frombuffer(frame, PREVIEW_SIZE, "RGB"), (x, y))
        else:
            pygame.draw.rect(self.screen, BLACK, (x, y, pw, ph))
        color = {GAS: GREEN, BRAKE: RED, COAST: YELLOW}[g.action] if g.hand_visible else (150, 150, 150)
        pygame.draw.rect(self.screen, color, (x - 2, y - 2, pw + 4, ph + 4), 3)
        pygame.draw.rect(self.screen, color, (x - 2, y + ph + 2, pw + 4, 30))
        self.text(g.label, self.font, BLACK, center=(x + pw // 2, y + ph + 17))
        self.text(f"camera {g.fps:.0f} fps", self.font_small, WHITE, topright=(x + pw, y - 20), shadow=True)

    def draw_ai_panel(self):
        x, y, w = W - 232, H - 132, 220
        panel = pygame.Surface((w, 122), pygame.SRCALPHA)
        panel.fill(PANEL)
        self.screen.blit(panel, (x, y))
        self.text("AI (Q-learning) thinks:", self.font_small, WHITE, topleft=(x + 10, y + 6))
        lo, hi = min(self.ai_q), max(self.ai_q)
        span = max(hi - lo, 1e-6)
        for row, (act, name, col) in enumerate(((GAS, "GAS", GREEN), (COAST, "COAST", YELLOW), (BRAKE, "BRAKE", RED))):
            ry = y + 30 + row * 30
            chosen = act == self.ai_action
            self.text(name, self.font_small if not chosen else self.font, col if chosen else WHITE, topleft=(x + 10, ry))
            bar = 16 + 80 * (self.ai_q[act] - lo) / span
            pygame.draw.rect(self.screen, col if chosen else (110, 110, 110), (x + 78, ry + 3, round(bar), 14), border_radius=4)
            self.text(f"{self.ai_q[act]:.1f}", self.font_small, WHITE, topleft=(x + 82 + bar, ry))

    def draw_overlay(self, title, line, hint=""):
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        self.screen.blit(shade, (0, 0))
        self.text(title, self.font_big, WHITE, center=(W // 2, H // 2 - 50))
        self.text(line, self.font, WHITE, center=(W // 2, H // 2 + 10))
        if hint:
            self.text(hint, self.font, YELLOW, center=(W // 2, H // 2 + 50))

    def menu_rects(self):
        return [pygame.Rect(W // 2 - 200, 210 + i * 76, 400, 60) for i in range(len(MENU_ITEMS))]

    def draw_menu(self):
        t = pygame.time.get_ticks() / 1000
        cam_x = t * 120
        self.draw_background(cam_x, 0)
        self.text("HILL CLIMB", self.font_big, WHITE, center=(W // 2, 80), shadow=True)
        self.text("hand gestures  +  reinforcement learning", self.font, WHITE, center=(W // 2, 130), shadow=True)
        for i, ((mode, label), rect) in enumerate(zip(MENU_ITEMS, self.menu_rects())):
            selected = i == self.menu_index
            pygame.draw.rect(self.screen, BLUE if selected else (40, 70, 120), rect, border_radius=14)
            if selected:
                pygame.draw.rect(self.screen, WHITE, rect, 3, border_radius=14)
            self.text(f"{i + 1}.  {label}", self.font_mid if selected else self.font, WHITE,
                      midleft=(rect.x + 24, rect.centery))
            best = self.save["best"].get(mode)
            if best:
                self.text(f"best {best:.0f} m", self.font_small, WHITE, midright=(rect.right - 18, rect.centery))
        if self.message:
            self.text(self.message, self.font, (255, 230, 120), center=(W // 2, H - 92), shadow=True)
        self.text(f"Coins collected: {self.save['total_coins']}   Diamonds: {self.save['total_diamonds']}",
                  self.font_small, WHITE, center=(W // 2, H - 58), shadow=True)
        self.text("UP/DOWN + ENTER or 1-3 to choose   ·   ESC to quit", self.font_small, WHITE,
                  center=(W // 2, H - 32), shadow=True)

    def draw_center_message(self, title, line):
        self.screen.fill((30, 40, 60))
        self.text(title, self.font_mid, WHITE, center=(W // 2, H // 2 - 20))
        self.text(line, self.font, WHITE, center=(W // 2, H // 2 + 24))

    def text(self, s, font, color, shadow=False, **pos):
        surf = font.render(s, True, color)
        rect = surf.get_rect(**pos)
        if shadow:
            self.screen.blit(font.render(s, True, BLACK), rect.move(2, 2))
        self.screen.blit(surf, rect)


if __name__ == "__main__":
    Game().run()
