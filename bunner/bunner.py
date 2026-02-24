# If the window is too tall to fit on the screen, check your operating system display settings and reduce display
# scaling if it is enabled.
import pgzero, pgzrun, pygame, sys
from random import *
from enum import Enum
import sys

if sys.version_info < (3,5):
    print("This game requires at least version 3.5 of Python. Please download it from www.python.org")
    sys.exit()

pgzero_version = [int(s) if s.isnumeric() else s for s in pgzero.__version__.split('.')]
if pgzero_version < [1,2]:
    print("This game requires at least version 1.2 of Pygame Zero. You have version {0}. Please upgrade using the command 'pip3 install --upgrade pgzero'".format(pgzero.__version__))
    sys.exit()

WIDTH = 480
HEIGHT = 800
TITLE = "Infinite Bunner"

ROW_HEIGHT = 40

DEBUG_SHOW_ROW_BOUNDARIES = False


class MyActor(Actor):
    def __init__(self, image, pos, anchor=("center", "bottom")):
        super().__init__(image, pos, anchor)
        self.children = []

    def draw(self, offset_x, offset_y):
        self.x += offset_x
        self.y += offset_y
        super().draw()
        for child_obj in self.children:
            child_obj.draw(self.x, self.y)
        self.x -= offset_x
        self.y -= offset_y

    def update(self):
        for child_obj in self.children:
            child_obj.update()


class Eagle(MyActor):
    def __init__(self, pos):
        super().__init__("eagles", pos)
        self.children.append(MyActor("eagle", (0, -32)))

    def update(self):
        self.y += 12


class PlayerState(Enum):
    ALIVE = 0
    SPLAT = 1
    SPLASH = 2
    EAGLE = 3


DIRECTION_UP    = 0
DIRECTION_RIGHT = 1
DIRECTION_DOWN  = 2
DIRECTION_LEFT  = 3

direction_keys = [keys.UP, keys.RIGHT, keys.DOWN, keys.LEFT]

DX = [0,  4, 0, -4]
DY = [-4, 0, 4,  0]


class Bunner(MyActor):
    MOVE_DISTANCE = 10

    def __init__(self, pos):
        super().__init__("blank", pos)
        self.state       = PlayerState.ALIVE
        self.direction   = 2
        self.timer       = 0
        self.input_queue = []
        self.min_y       = self.y

    def handle_input(self, dir):
        if dir == 4:          # no-op action
            return
        for row in game.rows:
            if row.y == self.y + Bunner.MOVE_DISTANCE * DY[dir]:
                if row.allow_movement(self.x + Bunner.MOVE_DISTANCE * DX[dir]):
                    self.direction = dir
                    self.timer     = Bunner.MOVE_DISTANCE
                    game.play_sound("jump", 1)
                return

    def update(self, action=None):
        if action is None:
            for direction in range(4):
                if key_just_pressed(direction_keys[direction]):
                    self.input_queue.append(direction)
        else:
            if self.timer == 0 and len(self.input_queue) == 0:
                self.input_queue.append(action)

        if self.state == PlayerState.ALIVE:
            if self.timer == 0 and len(self.input_queue) > 0:
                self.handle_input(self.input_queue.pop(0))

            land = False
            if self.timer > 0:
                self.x    += DX[self.direction]
                self.y    += DY[self.direction]
                self.timer -= 1
                land = self.timer == 0

            current_row = None
            for row in game.rows:
                if row.y == self.y:
                    current_row = row
                    break

            if current_row:
                self.state, dead_obj_y_offset = current_row.check_collision(self.x)
                if self.state == PlayerState.ALIVE:
                    self.x += current_row.push()
                    if land:
                        current_row.play_sound()
                else:
                    if self.state == PlayerState.SPLAT:
                        current_row.children.insert(
                            0, MyActor("splat" + str(self.direction), (self.x, dead_obj_y_offset))
                        )
                    self.timer = 1
            else:
                if self.y > game.scroll_pos + HEIGHT + 80:
                    game.eagle = Eagle((self.x, game.scroll_pos))
                    self.state = PlayerState.EAGLE
                    self.timer = 1
                    game.play_sound("eagle")

            self.x = max(16, min(WIDTH - 16, self.x))
        else:
            self.timer -= 1

        self.min_y = min(self.min_y, self.y)

        self.image = "blank"
        if self.state == PlayerState.ALIVE:
            if self.timer > 0:
                self.image = "jump" + str(self.direction)
            else:
                self.image = "sit" + str(self.direction)
        elif self.state == PlayerState.SPLASH and self.timer > 84:
            self.image = "splash" + str(int((100 - self.timer) / 2))


class Mover(MyActor):
    def __init__(self, dx, image, pos):
        super().__init__(image, pos)
        self.dx = dx

    def update(self):
        self.x += self.dx


class Car(Mover):
    SOUND_ZOOM = 0
    SOUND_HONK = 1

    def __init__(self, dx, pos):
        image = "car" + str(randint(0, 3)) + ("0" if dx < 0 else "1")
        super().__init__(dx, image, pos)
        self.played = [False, False]
        self.sounds = [("zoom", 6), ("honk", 4)]

    def play_sound(self, num):
        if not self.played[num]:
            game.play_sound(*self.sounds[num])
            self.played[num] = True


class Log(Mover):
    def __init__(self, dx, pos):
        image = "log" + str(randint(0, 1))
        super().__init__(dx, image, pos)


class Train(Mover):
    def __init__(self, dx, pos):
        image = "train" + str(randint(0, 2)) + ("0" if dx < 0 else "1")
        super().__init__(dx, image, pos)


class Row(MyActor):
    def __init__(self, base_image, index, y):
        super().__init__(base_image + str(index), (0, y), ("left", "bottom"))
        self.index = index
        self.dx    = 0

    def next(self):
        return

    def collide(self, x, margin=0):
        for child_obj in self.children:
            if (x >= child_obj.x - (child_obj.width / 2) - margin and
                    x < child_obj.x + (child_obj.width / 2) + margin):
                return child_obj
        return None

    def push(self):
        return 0

    def check_collision(self, x):
        return PlayerState.ALIVE, 0

    def allow_movement(self, x):
        return x >= 16 and x <= WIDTH - 16


class ActiveRow(Row):
    def __init__(self, child_type, dxs, base_image, index, y):
        super().__init__(base_image, index, y)
        self.child_type = child_type
        self.timer      = 0
        self.dx         = choice(dxs)

        x = -WIDTH / 2 - 70
        while x < WIDTH / 2 + 70:
            x += randint(240, 480)
            pos = (WIDTH / 2 + (x if self.dx > 0 else -x), 0)
            self.children.append(self.child_type(self.dx, pos))

    def update(self):
        super().update()
        self.children = [c for c in self.children if c.x > -70 and c.x < WIDTH + 70]
        self.timer -= 1
        if self.timer < 0:
            pos = (WIDTH + 70 if self.dx < 0 else -70, 0)
            self.children.append(self.child_type(self.dx, pos))
            self.timer = (1 + random()) * (240 / abs(self.dx))


class Hedge(MyActor):
    def __init__(self, x, y, pos):
        super().__init__("bush" + str(x) + str(y), pos)


def generate_hedge_mask():
    mask = [random() < 0.01 for i in range(12)]
    mask[randint(0, 11)] = True
    mask = [sum(mask[max(0, i-1):min(12, i+2)]) > 0 for i in range(12)]
    return [mask[0]] + mask + 2 * [mask[-1]]


def classify_hedge_segment(mask, previous_mid_segment):
    if mask[1]:
        sprite_x = None
    else:
        sprite_x = 3 - 2 * mask[0] - mask[2]

    if sprite_x == 3:
        if previous_mid_segment == 4 and mask[3]:
            return 5, None
        else:
            if previous_mid_segment is None or previous_mid_segment == 4:
                sprite_x = 3
            elif previous_mid_segment == 3:
                sprite_x = 4
            return sprite_x, sprite_x
    else:
        return sprite_x, None


class Grass(Row):
    def __init__(self, predecessor, index, y):
        super().__init__("grass", index, y)
        self.hedge_row_index = None
        self.hedge_mask      = None

        if not isinstance(predecessor, Grass) or predecessor.hedge_row_index is None:
            if random() < 0.5 and index > 7 and index < 14:
                self.hedge_mask      = generate_hedge_mask()
                self.hedge_row_index = 0
        elif predecessor.hedge_row_index == 0:
            self.hedge_mask      = predecessor.hedge_mask
            self.hedge_row_index = 1

        if self.hedge_row_index is not None:
            previous_mid_segment = None
            for i in range(1, 13):
                sprite_x, previous_mid_segment = classify_hedge_segment(
                    self.hedge_mask[i - 1:i + 3], previous_mid_segment
                )
                if sprite_x is not None:
                    self.children.append(Hedge(sprite_x, self.hedge_row_index, (i * 40 - 20, 0)))

    def allow_movement(self, x):
        return super().allow_movement(x) and not self.collide(x, 8)

    def play_sound(self):
        game.play_sound("grass", 1)

    def next(self):
        if self.index <= 5:
            row_class, index = Grass, self.index + 8
        elif self.index == 6:
            row_class, index = Grass, 7
        elif self.index == 7:
            row_class, index = Grass, 15
        elif 8 <= self.index <= 14:
            row_class, index = Grass, self.index + 1
        else:
            row_class, index = choice((Road, Water)), 0
        return row_class(self, index, self.y - ROW_HEIGHT)


class Dirt(Row):
    def __init__(self, predecessor, index, y):
        super().__init__("dirt", index, y)

    def play_sound(self):
        game.play_sound("dirt", 1)

    def next(self):
        if self.index <= 5:
            row_class, index = Dirt, self.index + 8
        elif self.index == 6:
            row_class, index = Dirt, 7
        elif self.index == 7:
            row_class, index = Dirt, 15
        elif 8 <= self.index <= 14:
            row_class, index = Dirt, self.index + 1
        else:
            row_class, index = choice((Road, Water)), 0
        return row_class(self, index, self.y - ROW_HEIGHT)


class Water(ActiveRow):
    def __init__(self, predecessor, index, y):
        dxs = [-2, -1] * (predecessor.dx >= 0) + [1, 2] * (predecessor.dx <= 0)
        super().__init__(Log, dxs, "water", index, y)

    def update(self):
        super().update()
        for log in self.children:
            if game.bunner and self.y == game.bunner.y and log == self.collide(game.bunner.x, -4):
                log.y = 2
            else:
                log.y = 0

    def push(self):
        return self.dx

    def check_collision(self, x):
        if self.collide(x, -4):
            return PlayerState.ALIVE, 0
        else:
            game.play_sound("splash")
            return PlayerState.SPLASH, 0

    def play_sound(self):
        game.play_sound("log", 1)

    def next(self):
        if self.index == 7 or (self.index >= 1 and random() < 0.5):
            row_class, index = Dirt, randint(4, 6)
        else:
            row_class, index = Water, self.index + 1
        return row_class(self, index, self.y - ROW_HEIGHT)


class Road(ActiveRow):
    def __init__(self, predecessor, index, y):
        dxs = list(set(range(-5, 6)) - {0, predecessor.dx})
        super().__init__(Car, dxs, "road", index, y)

    def update(self):
        super().update()
        for y_offset, car_sound_num in [
            (-ROW_HEIGHT, Car.SOUND_ZOOM),
            (0,           Car.SOUND_HONK),
            (ROW_HEIGHT,  Car.SOUND_ZOOM),
        ]:
            if game.bunner and game.bunner.y == self.y + y_offset:
                for child_obj in self.children:
                    if isinstance(child_obj, Car):
                        dx = child_obj.x - game.bunner.x
                        if (abs(dx) < 100
                                and ((child_obj.dx < 0) != (dx < 0))
                                and (y_offset == 0 or abs(child_obj.dx) > 1)):
                            child_obj.play_sound(car_sound_num)

    def check_collision(self, x):
        if self.collide(x):
            game.play_sound("splat", 1)
            return PlayerState.SPLAT, 0
        else:
            return PlayerState.ALIVE, 0

    def play_sound(self):
        game.play_sound("road", 1)

    def next(self):
        if self.index == 0:
            row_class, index = Road, 1
        elif self.index < 5:
            r = random()
            if r < 0.8:
                row_class, index = Road, self.index + 1
            elif r < 0.88:
                row_class, index = Grass, randint(0, 6)
            elif r < 0.94:
                row_class, index = Rail, 0
            else:
                row_class, index = Pavement, 0
        else:
            r = random()
            if r < 0.6:
                row_class, index = Grass, randint(0, 6)
            elif r < 0.9:
                row_class, index = Rail, 0
            else:
                row_class, index = Pavement, 0
        return row_class(self, index, self.y - ROW_HEIGHT)


class Pavement(Row):
    def __init__(self, predecessor, index, y):
        super().__init__("side", index, y)

    def play_sound(self):
        game.play_sound("sidewalk", 1)

    def next(self):
        if self.index < 2:
            row_class, index = Pavement, self.index + 1
        else:
            row_class, index = Road, 0
        return row_class(self, index, self.y - ROW_HEIGHT)


class Rail(Row):
    def __init__(self, predecessor, index, y):
        super().__init__("rail", index, y)
        self.predecessor = predecessor

    def update(self):
        super().update()
        if self.index == 1:
            self.children = [c for c in self.children if c.x > -1000 and c.x < WIDTH + 1000]
            if (self.y < game.scroll_pos + HEIGHT
                    and len(self.children) == 0
                    and random() < 0.01):
                dx = choice([-20, 20])
                self.children.append(Train(dx, (WIDTH + 1000 if dx < 0 else -1000, -13)))
                game.play_sound("bell")
                game.play_sound("train", 2)

    def check_collision(self, x):
        if self.index == 2 and self.predecessor.collide(x):
            game.play_sound("splat", 1)
            return PlayerState.SPLAT, 8
        else:
            return PlayerState.ALIVE, 0

    def play_sound(self):
        game.play_sound("grass", 1)

    def next(self):
        if self.index < 3:
            row_class, index = Rail, self.index + 1
        else:
            item = choice(((Road, 0), (Water, 0)))
            row_class, index = item[0], item[1]
        return row_class(self, index, self.y - ROW_HEIGHT)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class Game:
    def __init__(self, bunner=None):
        self.bunner        = bunner
        self.looped_sounds = {}
        self.eagle         = None
        self.frame         = 0

        try:
            if bunner:
                music.set_volume(0.4)
            else:
                music.play("theme")
                music.set_volume(1)
        except:
            pass

        self.rows       = [Grass(None, 0, 0)]
        self.scroll_pos = -HEIGHT

    # ------------------------------------------------------------------
    # Core update / draw
    # ------------------------------------------------------------------

    def update(self, action=None):
        if self.bunner:
            self.scroll_pos -= max(1, min(5, float(self.scroll_pos + HEIGHT - self.bunner.y) / (HEIGHT // 4)))
        else:
            self.scroll_pos -= 1

        self.rows = [row for row in self.rows if row.y < int(self.scroll_pos) + HEIGHT + ROW_HEIGHT * 2]

        while self.rows[-1].y > int(self.scroll_pos) + ROW_HEIGHT:
            self.rows.append(self.rows[-1].next())

        for obj in self.rows + [self.bunner, self.eagle]:
            if obj:
                if isinstance(obj, Bunner):
                    obj.update(action)
                else:
                    obj.update()

        if self.bunner:
            for name, count, row_class in [("river", 2, Water), ("traffic", 3, Road)]:
                volume = sum([
                    16.0 / max(16.0, abs(r.y - self.bunner.y))
                    for r in self.rows if isinstance(r, row_class)
                ]) - 0.2
                self.loop_sound(name, count, min(0.4, volume))

        return self

    def draw(self):
        all_objs = list(self.rows)
        if self.bunner:
            all_objs.append(self.bunner)

        def sort_key(obj):
            return (obj.y + 39) // ROW_HEIGHT

        all_objs.sort(key=sort_key)
        all_objs.append(self.eagle)

        for obj in all_objs:
            if obj:
                obj.draw(0, -int(self.scroll_pos))

        if DEBUG_SHOW_ROW_BOUNDARIES:
            for obj in all_objs:
                if obj and isinstance(obj, Row):
                    pygame.draw.rect(
                        screen.surface, (255, 255, 255),
                        pygame.Rect(obj.x, obj.y - int(self.scroll_pos), screen.surface.get_width(), ROW_HEIGHT), 1
                    )
                    screen.draw.text(str(obj.index), (obj.x, obj.y - int(self.scroll_pos) - ROW_HEIGHT))

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self):
        return int(-320 - game.bunner.min_y) // 40

    # ------------------------------------------------------------------
    # Sound helpers
    # ------------------------------------------------------------------

    def play_sound(self, name, count=1):
        try:
            if self.bunner:
                sound = getattr(sounds, name + str(randint(0, count - 1)))
                sound.play()
        except:
            pass

    def loop_sound(self, name, count, volume):
        try:
            if volume > 0 and name not in self.looped_sounds:
                sound = getattr(sounds, name + str(randint(0, count - 1)))
                sound.play(-1)
                self.looped_sounds[name] = sound

            if name in self.looped_sounds:
                sound = self.looped_sounds[name]
                if volume > 0:
                    sound.set_volume(volume)
                else:
                    sound.stop()
                    del self.looped_sounds[name]
        except:
            pass

    def stop_looped_sounds(self):
        try:
            for sound in self.looped_sounds.values():
                sound.stop()
            self.looped_sounds.clear()
        except:
            pass

    # ------------------------------------------------------------------
    # RL helpers
    # ------------------------------------------------------------------

    def _is_action_invalid(self, action: int) -> bool:
        """Return True if the action would result in blocked / off-screen movement."""
        if action == 4:
            return False   # no-op is always valid
        target_y = self.bunner.y + Bunner.MOVE_DISTANCE * DY[action]
        target_x = self.bunner.x + Bunner.MOVE_DISTANCE * DX[action]
        for row in self.rows:
            if row.y == target_y:
                return not row.allow_movement(target_x)
        return False   # row not found yet — don't penalise

    def get_processed_frame(self) -> 'np.ndarray':
        """
        Capture the current display surface and return a (80, 80, 1) float32 array.
        draw() must be called before this method.
        """
        import cv2
        import numpy as np

        surface = pygame.surfarray.array3d(pygame.display.get_surface())  # (W, H, 3)
        gray    = cv2.cvtColor(surface, cv2.COLOR_RGB2GRAY)                # (W, H)
        resized = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA) # (80, 80)
        normed  = resized.astype(np.float32) / 255.0
        return np.expand_dims(normed, axis=-1)                             # (80, 80, 1)

    def game_step(self, action: int):
        """
        Execute one agent action and return (next_frame, reward, terminal).

        Reward design (clean, single-signal):
          +0.5  for each new row of forward progress
          -0.05 for attempting an invalid/blocked move
          -1.0  on death
        """
        import numpy as np

        prev_min_y = self.bunner.min_y
        invalid    = self._is_action_invalid(action)

        # Step game until bunner lands (timer reaches 0) or dies
        self.update(action)
        while self.bunner.timer > 0 and self.bunner.state == PlayerState.ALIVE:
            self.update()

        terminal = self.bunner.state != PlayerState.ALIVE

        # ---- Reward ----
        reward = 0.0

        # Forward progress: each row is ROW_HEIGHT=40 units; reward per row crossed
        rows_advanced = max(0, (prev_min_y - self.bunner.min_y)) / ROW_HEIGHT
        reward += 0.5 * rows_advanced

        # Penalty for wasted moves against walls/hedges
        if invalid:
            reward -= 0.05

        # Death penalty
        if terminal:
            reward -= 1.0

        # Render so get_processed_frame sees the latest state
        self.draw()
        frame = self.get_processed_frame()

        # Push frame into the shared state buffer
        dqn.state_buffer.push(frame)

        # Wait out death animation before returning control
        if terminal:
            while self.bunner.timer >= 0:
                self.update()

        print(f"reward={reward:+.3f}  rows={rows_advanced:.1f}  "
              f"invalid={invalid}  terminal={terminal}")

        return frame, reward, terminal


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

key_status = {}

def key_just_pressed(key):
    result       = False
    prev_status  = key_status.get(key, False)
    if not prev_status and keyboard[key]:
        result   = True
    key_status[key] = keyboard[key]
    return result


def display_number(n, colour, x, align):
    n = str(n)
    for i in range(len(n)):
        screen.blit("digit" + str(colour) + n[i], (x + (i - len(n) * align) * 25, 0))


# ---------------------------------------------------------------------------
# DQN setup
# ---------------------------------------------------------------------------

from dqn import DQN

dqn = DQN(
    num_actions=5,
    replay_size=50_000,
    learning_rate=6e-4,
    discount_factor=0.95,
    init_epsilon=0.2,
    final_epsilon=0.01,
    epsilon_decay=30_000,
    observation_limit=10_000,
    batch_size=32,
    target_update_frequency=1_000,
    save_frequency=1_000,
    seq_len=4,
)


# ---------------------------------------------------------------------------
# Game state machine
# ---------------------------------------------------------------------------

class State(Enum):
    MENU      = 1
    PLAY      = 2
    GAME_OVER = 3


def _reset_episode():
    """Create a fresh game episode and zero the state buffer."""
    import numpy as np
    dqn.state_buffer.reset()
    return Game(Bunner((240, -320)))


def update():
    global state, game, high_score

    if state == State.MENU:
        if key_just_pressed(keys.SPACE):
            state = State.PLAY
            game  = _reset_episode()
        else:
            game.update()

    elif state == State.PLAY:
        if game.bunner.state != PlayerState.ALIVE and game.bunner.timer < 0:
            high_score = max(high_score, game.score())
            try:
                with open("high.txt", "w") as f:
                    f.write(str(high_score))
            except:
                pass
            state = State.GAME_OVER
        else:
            # current state tensor is managed entirely inside train_step / game_step
            dqn.train_step(None, game.game_step)

    elif state == State.GAME_OVER:
        game.stop_looped_sounds()
        state = State.PLAY
        game  = _reset_episode()


def draw():
    game.draw()

    if state == State.MENU:
        screen.blit("title", (0, 0))
        screen.blit("start" + str([0, 1, 2, 1][game.scroll_pos // 6 % 4]),
                    ((WIDTH - 270) // 2, HEIGHT - 240))

    elif state == State.PLAY:
        display_number(game.score(),  0, 0,         0)
        display_number(high_score,    1, WIDTH - 10, 1)

    elif state == State.GAME_OVER:
        screen.blit("gameover", (0, 0))


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

try:
    pygame.mixer.set_num_channels(16)
except:
    pass

try:
    with open("high.txt", "r") as f:
        high_score = int(f.read())
except:
    high_score = 0

state = State.MENU
game  = Game()

pgzrun.go()

