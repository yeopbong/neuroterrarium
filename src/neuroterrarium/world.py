"""Fixed-step local sensing and synchronous two-dimensional bodies.

Distances, energy and food are simulation units. This is an engineering body,
not a biomechanical or metabolic model. Observation at time t selects the
action held over [t,t+0.02); computation time does not alter this convention.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass, field
import math

import numpy as np

ACTION_DT = 0.02
BODY_DT = 0.01
RAYS = 16
RANGE = 24.0
RADIUS = 0.55
OBSERVATION_SCHEMA = "local-rays-v1"
ACTION_SCHEMA = "drive-turn-escape-feed-v1"


def stream(seed: int, name: str) -> np.random.Generator:
    """Independent deterministic stream; Python hash is deliberately unused."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype="<u4"))


def segment_circle(a, b, center, radius: float) -> bool:
    a, b, center = np.asarray(a), np.asarray(b), np.asarray(center)
    ab = b - a
    u = float(np.clip(np.dot(center - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    return bool(np.linalg.norm(a + u * ab - center) <= radius)


@dataclass
class Body:
    x: float
    y: float
    heading: float
    speed: float = 0.0
    energy: float = 1.0
    food: float = 0.0
    collisions: int = 0
    action: list[float] = field(default_factory=lambda: [0.0] * 4)


@dataclass
class Food:
    x: float
    y: float
    amount: float = 1.0
    radius: float = 1.2


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float = 2.0


@dataclass
class Stimulus:
    x: float
    y: float
    radius: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    growth: float = 0.0
    physical: bool = False


class World:
    def __init__(self, seed: int = 0, count: int = 10, scenario: str = "open"):
        if not 1 <= count <= 10 or scenario not in {"open", "occluded", "looming"}:
            raise ValueError("invalid population or scenario")
        self.seed, self.scenario = seed, scenario
        self.width, self.height, self.step = 80.0, 56.0, 0
        rng = stream(seed, "initial-bodies")
        self.bodies = [Body(8 + i * 6.0, 25 + float(rng.uniform(-5, 5)),
                            float(rng.uniform(-math.pi, math.pi))) for i in range(count)]
        self.foods = [Food(20, 16), Food(57, 39), Food(39, 29)]
        self.obstacles = [] if scenario == "open" else [Obstacle(32, 25, 4), Obstacle(46, 36, 3)]
        self.stimuli = [] if scenario != "looming" else [Stimulus(50, 17, 1, -2, 0)]
        self.external_events: list[dict] = []
        self.previous_projection = np.zeros((count, RAYS), dtype=np.float64)
        self._observed_step = -1
        self._observations = None
        self.replenish = False

    @property
    def time(self) -> float:
        return self.step * ACTION_DT

    def visible(self, a, b) -> bool:
        return not any(segment_circle(a, b, [o.x, o.y], o.radius) for o in self.obstacles)

    def observe(self) -> np.ndarray:
        """Idempotent at a fixed world step; no target labels or global coordinates.

        Per ray: proximity, projected dark-object occupancy, signed temporal
        projection change. Four chemical samples, contact taste, energy,
        speed, and last applied action follow. All controllers get this schema.
        """
        rows, projections = [], []
        for i, body in enumerate(self.bodies):
            position = np.array([body.x, body.y])
            proximity, projection = np.zeros(RAYS), np.zeros(RAYS)
            for k in range(RAYS):
                angle = body.heading + 2 * math.pi * k / RAYS
                direction = np.array([math.cos(angle), math.sin(angle)])
                candidates = [(o.x, o.y, o.radius, True) for o in self.obstacles]
                candidates += [(s.x, s.y, s.radius, False) for s in self.stimuli
                               if self.visible(position, [s.x, s.y])]
                for x, y, radius, solid in candidates:
                    delta = np.array([x, y]) - position
                    distance = float(np.linalg.norm(delta))
                    if distance > RANGE + radius:
                        continue
                    bearing = math.atan2(delta[1], delta[0])
                    difference = math.atan2(math.sin(bearing-angle), math.cos(bearing-angle))
                    halfwidth = math.asin(min(radius/max(distance, 1e-9), 1.0))
                    half_bin = math.pi/RAYS
                    occupancy = max(0.0, min(half_bin,difference+halfwidth)
                                    - max(-half_bin,difference-halfwidth))/(2*half_bin)
                    projection[k] = max(projection[k], float(occupancy))
                    if solid and np.dot(delta, direction) > 0:
                        lateral2 = float(np.dot(delta, delta) - np.dot(delta, direction)**2)
                        if lateral2 <= radius**2:
                            near = np.dot(delta, direction) - math.sqrt(max(0, radius**2-lateral2))
                            proximity[k] = max(proximity[k], max(0, 1 - near/RANGE))
                for axis, coordinate, extent in [(0, body.x, self.width), (1, body.y, self.height)]:
                    d = direction[axis]
                    if abs(d) > 1e-12:
                        wall_distance = ((extent if d > 0 else 0) - coordinate)/d
                        proximity[k] = max(proximity[k], max(0, 1-wall_distance/RANGE))
            delta_projection = np.clip((projection-self.previous_projection[i])/ACTION_DT/20, -1, 1)
            if self.step == 0:
                delta_projection[:] = 0
            chemical = np.zeros(4)
            taste = 0.0
            for food in self.foods:
                if food.amount <= 0 or not self.visible(position, [food.x, food.y]):
                    continue
                distance = math.hypot(food.x-body.x, food.y-body.y)
                if distance <= food.radius+RADIUS:
                    taste = 1.0
                for k in range(4):
                    a = body.heading + k*math.pi/2
                    d = math.hypot(food.x-body.x-math.cos(a), food.y-body.y-math.sin(a))
                    chemical[k] += max(0, 1-d/RANGE) * min(food.amount, 1)
            row = np.r_[proximity, projection, delta_projection, np.clip(chemical, 0, 1),
                        taste, body.energy, body.speed/20, body.action]
            rows.append(row)
            projections.append(projection)
        self._observations = np.asarray(rows, dtype=np.float64)
        self._observed_step = self.step
        return self._observations.copy()

    def advance(self, actions: np.ndarray) -> dict:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (len(self.bodies), 4) or not np.isfinite(actions).all():
            raise ValueError("actions must be a finite population by four array")
        actions = np.clip(actions, [0,-1,0,0], [1,1,1,1])
        self.previous_projection = self.observe()[:,RAYS:2*RAYS].copy()
        before_food = sum(b.food for b in self.bodies)
        for _ in range(round(ACTION_DT/BODY_DT)):
            proposals = []
            for b, a in zip(self.bodies, actions, strict=True):
                heading = math.atan2(math.sin(b.heading+a[1]*5*BODY_DT), math.cos(b.heading+a[1]*5*BODY_DT))
                target_speed = (a[0]*7 + a[2]*13) * (0.25+0.75*b.energy)
                speed = b.speed + (target_speed-b.speed)*(1-math.exp(-BODY_DT/0.08))
                x,y = b.x+math.cos(heading)*speed*BODY_DT, b.y+math.sin(heading)*speed*BODY_DT
                blocked = not (RADIUS <= x <= self.width-RADIUS and RADIUS <= y <= self.height-RADIUS)
                blocked |= any(math.hypot(x-o.x,y-o.y) < RADIUS+o.radius for o in self.obstacles)
                proposals.append([x,y,heading,speed,blocked])
            # Resolve rollback conflicts to a fixed point before committing any
            # body. A stopped leader must remain an obstacle to its follower.
            for _resolution in range(len(proposals)+1):
                newly_blocked = set()
                positions = [(b.x,b.y) if p[4] else (p[0],p[1])
                             for b,p in zip(self.bodies,proposals,strict=True)]
                for i in range(len(proposals)):
                    for j in range(i+1,len(proposals)):
                        if math.hypot(positions[i][0]-positions[j][0],positions[i][1]-positions[j][1]) < 2*RADIUS:
                            if not proposals[i][4]: newly_blocked.add(i)
                            if not proposals[j][4]: newly_blocked.add(j)
                if not newly_blocked: break
                for i in newly_blocked: proposals[i][4] = True
            for b,a,p in zip(self.bodies,actions,proposals,strict=True):
                if p[4]:
                    b.collisions += 1
                    b.speed = 0
                else:
                    b.x,b.y,b.speed = p[0],p[1],p[3]
                b.heading = p[2]
                cost = BODY_DT*(0.001+0.003*a[0]+0.015*a[2]+0.001*abs(a[1]))
                b.energy = max(0,b.energy-cost)
                b.action = a.tolist()
            for food in self.foods:
                claims = np.array([BODY_DT*0.25*a[3] if math.hypot(b.x-food.x,b.y-food.y) <= RADIUS+food.radius else 0
                                   for b,a in zip(self.bodies,actions,strict=True)])
                demand = claims.sum()
                if demand:
                    claims *= min(1, food.amount/demand)
                    food.amount = max(0,food.amount-float(claims.sum()))
                    for b,amount in zip(self.bodies,claims,strict=True):
                        b.food += float(amount)
                        b.energy = min(1,b.energy+float(amount)*0.5)
            for s in self.stimuli:
                s.x += s.vx*BODY_DT; s.y += s.vy*BODY_DT
                s.radius = max(0.01,s.radius+s.growth*BODY_DT)
                if s.physical:
                    for b in self.bodies:
                        if math.hypot(b.x-s.x,b.y-s.y) <= s.radius+RADIUS:
                            b.energy = max(0,b.energy-0.1*BODY_DT)
        self.step += 1
        if self.replenish:
            for f in self.foods:
                if f.amount < 0.01: f.amount = 1.0
        return {"food_gained":sum(b.food for b in self.bodies)-before_food,
                "simulation_seconds":self.time}

    def snapshot(self) -> dict:
        return {"schema":"world-v1", "seed":self.seed, "scenario":self.scenario,
                "step":self.step, "bodies":[asdict(x) for x in self.bodies],
                "foods":[asdict(x) for x in self.foods], "obstacles":[asdict(x) for x in self.obstacles],
                "stimuli":[asdict(x) for x in self.stimuli],
                "external_events":copy.deepcopy(self.external_events),
                "previous_projection":self.previous_projection.tolist(),
                "replenish":self.replenish}

    @classmethod
    def restore(cls, state: dict) -> "World":
        fields = {"schema","seed","scenario","step","bodies","foods","obstacles","stimuli",
                  "external_events","previous_projection","replenish"}
        if not isinstance(state,dict) or set(state)!=fields or state.get("schema") != "world-v1":
            raise ValueError("world schema mismatch")
        if type(state['seed']) is not int or not 0<=state['seed']<2**53 or type(state['step']) is not int or not 0<=state['step']<=10**9:
            raise ValueError('invalid clock or seed')
        if type(state['replenish']) is not bool or state['external_events'] != []:
            raise ValueError('invalid world events or resource mode')
        for key,limit in [('bodies',10),('foods',256),('obstacles',256),('stimuli',32)]:
            if not isinstance(state[key],list) or len(state[key])>limit: raise ValueError('world capacity exceeded')
            for item in state[key]:
                if not isinstance(item,dict): raise ValueError('invalid body or object')
                for name,value in item.items():
                    if name=='action':
                        if not isinstance(value,list) or len(value)!=4: raise ValueError('invalid action')
                        if any(type(x) not in (int,float) or not math.isfinite(x) for x in value): raise ValueError('invalid action')
                        if any(x<lo or x>hi for x,lo,hi in zip(value,[0,-1,0,0],[1,1,1,1],strict=True)): raise ValueError('invalid action')
                    elif name=='physical':
                        if type(value) is not bool: raise ValueError('invalid stimulus flag')
                    elif (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating))
                          or not math.isfinite(value) or
                          (not (key=='stimuli' and name in {'x','y','radius'}) and
                           abs(value)>(10**9 if name=='collisions' else 10000))):
                        raise ValueError('invalid object value')
                if key=='bodies':
                    if not 0<=item.get('energy',-1)<=1 or not 0<=item.get('speed',-1)<=20.000001 or item.get('food',-1)<0:
                        raise ValueError('invalid body state')
                    if type(item.get('collisions')) is not int or item['collisions']<0: raise ValueError('invalid collision count')
                if key=='stimuli':
                    # A finite initial domain expands only by the stimulus's
                    # recorded velocity/growth over the elapsed simulation time.
                    # The allowance bounds accumulated binary64 addition error.
                    for name,rate_name,initial_limit in [('x','vx',10000),('y','vy',10000),('radius','growth',100)]:
                        if name not in item: continue
                        rate=abs(item.get(rate_name,0)) if name!='radius' else max(0,item.get(rate_name,0))
                        limit=initial_limit+rate*state['step']*ACTION_DT
                        allowance=(2*state['step']+8)*math.ulp(limit)
                        if abs(float(item[name]))>limit+allowance: raise ValueError('stimulus exceeds elapsed-time domain')
                if 'radius' in item and (item['radius']<=0 or (key!='stimuli' and item['radius']>100)):
                    raise ValueError('invalid object radius')
                if 'amount' in item and not 0<=item['amount']<=10000: raise ValueError('invalid food amount')
        projection=np.asarray(state['previous_projection'],dtype=float)
        if projection.shape!=(len(state['bodies']),RAYS) or not np.isfinite(projection).all() or np.any((projection<0)|(projection>1)):
            raise ValueError('invalid projection history')
        world = cls(state["seed"],len(state["bodies"]),state["scenario"])
        for key,kind in [("bodies",Body),("foods",Food),("obstacles",Obstacle),("stimuli",Stimulus)]:
            try: setattr(world,key,[kind(**item) for item in state[key]])
            except TypeError as exc: raise ValueError('invalid object fields') from exc
        world.step = state["step"]
        world.previous_projection = projection.copy()
        world.external_events = copy.deepcopy(state["external_events"])
        world.replenish = bool(state["replenish"])
        return world
