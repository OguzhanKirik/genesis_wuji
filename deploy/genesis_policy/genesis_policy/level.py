"""level.yaml: every constant the runtime needs, loaded with validation.

The file is the source of truth for numbers; this module only checks that
what adam-rl wrote is complete, well-typed and self-consistent and turns it
into a frozen `Level`. The schema (keys, units, sizes) is documented in
README.md "Handoff schema". `make_stub_policy.py` writes an example.

Anything missing, mistyped or out of range raises LevelError with the key
path in the message, before any motion.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import yaml

from . import so3

SCHEMA_VERSION = 1
N_ARM = 7
N_SYNERGY = 9
N_THUMB = 3                   # closure 0 drives these, closure 1 the other six
N_HAND = 20
N_ACTION = 8
TIP_NAMES = ('thumb', 'index', 'middle')
GROUP_NAMES = ('tips', 'palm', 'forearm', 'upperarm')
# Every term the policy may consume, with its fixed size. The layout in the
# file picks the subset and the order; a name outside this table is an error.
TERM_SIZES = {
    'bolt_pos_rel': 3, 'bolt_rot_rel': 6, 'wrist_lin_vel': 3, 'wrist_ang_vel': 3,
    'finger_q': N_SYNERGY, 'finger_qd': N_SYNERGY, 'last_action': N_ACTION,
    'tip_forces': 3, 'contacts': 3, 'ep_time': 1, 'arm_q_norm': N_ARM, 'tip_dist': 3,
    'safety': 16,
}


class LevelError(ValueError):
    pass


@dataclass(frozen=True)
class Layout:
    terms: Tuple[Tuple[str, int], ...]

    @property
    def size(self):
        return sum(n for _, n in self.terms)

    @property
    def names(self):
        return [n for n, _ in self.terms]

    def offset(self, name):
        at = 0
        for n, size in self.terms:
            if n == name:
                return at
            at += size
        raise KeyError(name)


LATCH_MODES = ('closure', 'torque')


@dataclass(frozen=True)
class Latch:
    """mode 'closure': both closures past closure_min and the synergy joints
    slower than finger_qd_max. mode 'torque' (adam-rl hw_latch, level 39+):
    the thumb's and the index's or middle's flexion-chain |effort| above
    tau_min_nm, each for `steps` steps, the fingertips inside grasp_region_m of
    the estimated rod, the pinch midpoint within pinch_centre_tol_m of its axis
    (0 = off), and the estimate moving with the wrist (rel_vel_max_m_s)."""
    mode: str
    steps: int
    closure_min: float = 0.0
    finger_qd_max: float = 0.0
    tau_min_nm: float = 0.0
    grasp_region_m: float = 0.0
    pinch_centre_tol_m: float = 0.0
    rel_vel_max_m_s: float = 0.0
    est_vel_window: int = 1       # torque mode: the estimate's velocity over this many steps (adam-rl level 40+)


@dataclass(frozen=True)
class Phase:
    settle_in: int
    grasp_max_steps: int
    grasp_retry_steps: int
    settle_steps: int
    drag_steps: int
    free_prelift: bool
    grip_check: bool
    grip_check_steps: int
    grip_test_total_m: float
    grip_check_ratio: float
    grip_check_min_tips_m: float
    grip_check_late: bool
    lift_steps: int
    lift_slow_steps: int
    lift_slow_step_m: float
    lift_height_m: float
    hold_steps: int
    retry_max: int
    success_lift_m: float
    up_threshold_m: float
    lift_dir: np.ndarray
    latch: Latch
    sense_est: bool = False       # the rod's rest height is the estimate's at step 1 (adam-rl sense_est)
    roll_cap: float = math.pi / 2  # rad, the most the wrist rolls with a leaning rod (adam-rl RGLevel.roll_cap)

    @property
    def pre_lift_steps(self):
        return self.drag_steps + self.grip_check_steps

    @property
    def lift_start(self):
        return self.settle_steps + self.pre_lift_steps

    @property
    def scripted_steps(self):
        return self.settle_steps + self.pre_lift_steps + self.lift_steps + self.hold_steps


@dataclass(frozen=True)
class Sphere:
    link: str
    offset: np.ndarray
    radius: float


@dataclass(frozen=True)
class Box:
    lo: np.ndarray
    hi: np.ndarray


@dataclass(frozen=True)
class Level:
    source: str
    rate_hz: float
    layout: Layout
    episode_max_steps: int
    tip_force_scale: float
    arm_joints: Tuple[str, ...]
    arm_q_lo: np.ndarray          # normalisation + null-space limits (training's)
    arm_q_hi: np.ndarray
    effective_translation: float
    effective_rotation: float
    wrist_smoothing: float
    action_smoothing: float
    yaw_rate_scale: float
    dls_damping: float
    nullspace_gain: float
    joint_margin: float
    max_joint_velocity: float
    singularity_threshold: float
    workspace_radius_xy: float
    workspace_z_min: float
    workspace_z_max: float
    nominal_wrist_offset_xy: np.ndarray
    wrist_link: str
    wrist_offset: np.ndarray      # 4x4, policy wrist frame in the link frame
    hand_joints: Tuple[str, ...]  # the 9 synergy joints, observation order: thumb (3) then fingers (6)
    hand_open: np.ndarray
    hand_closed: np.ndarray
    hand_held: Dict[str, float]   # the other 11
    tips: Dict[str, Tuple[str, np.ndarray]]
    phase: Phase
    safety_groups: Dict[str, Tuple[Sphere, ...]]
    safety_vec_clamp: float
    safety_dist_clamp: Tuple[float, float]
    extra_boxes: Tuple[Box, ...]
    training_boxes: Tuple[Box, ...]   # the bin walls the policy trained against (empty: none exported)
    rod_radius: float
    rod_half_len: float
    effort_obs: Tuple[float, float] = ()   # (N per N m, contact N m): tip_forces / contacts from the flexion-chain effort
    raw: dict = field(repr=False, compare=False, default_factory=dict)

    @property
    def dt(self):
        return 1.0 / self.rate_hz

    @property
    def all_hand_joints(self):
        return tuple(self.hand_joints) + tuple(self.hand_held)


# ---------------------------------------------------------------------------
# reading helpers: every failure names the key
# ---------------------------------------------------------------------------


def _get(d, path, typ=None):
    node = d
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node:
            raise LevelError(f'level.yaml: missing key {path!r}')
        node = node[key]
    if typ is not None and not isinstance(node, typ):
        raise LevelError(f'level.yaml: {path!r} must be {typ.__name__}, got {type(node).__name__}')
    return node


def _num(d, path, lo=None, hi=None):
    v = _get(d, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise LevelError(f'level.yaml: {path!r} must be a number, got {v!r}')
    v = float(v)
    if not np.isfinite(v):
        raise LevelError(f'level.yaml: {path!r} is not finite')
    if lo is not None and v < lo:
        raise LevelError(f'level.yaml: {path!r} = {v} is below {lo}')
    if hi is not None and v > hi:
        raise LevelError(f'level.yaml: {path!r} = {v} is above {hi}')
    return v


def _int(d, path, lo=0):
    v = _get(d, path)
    if isinstance(v, bool) or not isinstance(v, int):
        raise LevelError(f'level.yaml: {path!r} must be an integer, got {v!r}')
    if v < lo:
        raise LevelError(f'level.yaml: {path!r} = {v} is below {lo}')
    return v


def _bool(d, path):
    v = _get(d, path)
    if not isinstance(v, bool):
        raise LevelError(f'level.yaml: {path!r} must be true or false, got {v!r}')
    return v


def _vec(d, path, n):
    v = _get(d, path)
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise LevelError(f'level.yaml: {path!r} must be a list of {n} numbers')
    try:
        out = np.array([float(x) for x in v], dtype=float)
    except (TypeError, ValueError) as e:
        raise LevelError(f'level.yaml: {path!r} must be numeric: {e}') from e
    if not np.all(np.isfinite(out)):
        raise LevelError(f'level.yaml: {path!r} is not finite')
    return out


def _names(d, path, n):
    v = _get(d, path)
    if not isinstance(v, (list, tuple)) or len(v) != n or not all(isinstance(x, str) and x for x in v):
        raise LevelError(f'level.yaml: {path!r} must be {n} joint names')
    if len(set(v)) != n:
        raise LevelError(f'level.yaml: {path!r} has duplicates')
    return tuple(v)


def _table(d, path, names):
    """{joint: value} over exactly `names`."""
    v = _get(d, path, dict)
    missing = [n for n in names if n not in v]
    extra = [n for n in v if n not in names]
    if missing or extra:
        raise LevelError(f'level.yaml: {path!r} must list exactly {list(names)}; '
                         f'missing {missing}, extra {extra}')
    out = []
    for n in names:
        x = v[n]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
            raise LevelError(f'level.yaml: {path}.{n} must be a finite number')
        out.append(float(x))
    return np.array(out)


def _layout(d):
    raw = _get(d, 'observation.layout', list)
    terms = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or 'name' not in item or 'size' not in item:
            raise LevelError(f'level.yaml: observation.layout[{i}] must be {{name, size}}')
        name, size = item['name'], item['size']
        if name not in TERM_SIZES:
            raise LevelError(f'level.yaml: observation.layout[{i}] names unknown term {name!r}; '
                             f'known: {sorted(TERM_SIZES)}')
        if size != TERM_SIZES[name]:
            raise LevelError(f'level.yaml: observation.layout[{i}] {name} has size {size}, '
                             f'the term is {TERM_SIZES[name]} wide')
        terms.append((name, int(size)))
    if not terms:
        raise LevelError('level.yaml: observation.layout is empty')
    if len({n for n, _ in terms}) != len(terms):
        raise LevelError('level.yaml: observation.layout repeats a term')
    return Layout(tuple(terms))


def _spheres(d, group):
    raw = _get(d, f'safety.groups.{group}', list)
    out = []
    for i, item in enumerate(raw):
        where = f'safety.groups.{group}[{i}]'
        if not isinstance(item, (list, tuple)) or len(item) != 3 or not isinstance(item[0], str):
            raise LevelError(f'level.yaml: {where} must be [link, [x, y, z], radius]')
        link, off, r = item
        if not isinstance(off, (list, tuple)) or len(off) != 3:
            raise LevelError(f'level.yaml: {where} offset must be [x, y, z]')
        try:
            off = np.array([float(x) for x in off])
            r = float(r)
        except (TypeError, ValueError) as e:
            raise LevelError(f'level.yaml: {where}: {e}') from e
        if r <= 0.0:
            raise LevelError(f'level.yaml: {where} radius must be positive')
        out.append(Sphere(link, off, r))
    if not out:
        raise LevelError(f'level.yaml: safety.groups.{group} is empty')
    return tuple(out)


def _latch(d, p):
    mode = _get(d, p, dict).get('mode', 'closure')
    if mode not in LATCH_MODES:
        raise LevelError(f'level.yaml: {p}.mode must be one of {LATCH_MODES}, got {mode!r}')
    steps = _int(d, f'{p}.steps', 1)
    if mode == 'closure':
        return Latch(mode, steps, closure_min=_num(d, f'{p}.closure_min', 0.0, 1.0),
                     finger_qd_max=_num(d, f'{p}.finger_qd_max', 0.0))
    return Latch(mode, steps, tau_min_nm=_num(d, f'{p}.tau_min_nm', 0.0),
                 grasp_region_m=_num(d, f'{p}.grasp_region_m', 0.0),
                 pinch_centre_tol_m=_num(d, f'{p}.pinch_centre_tol_m', 0.0),
                 rel_vel_max_m_s=_num(d, f'{p}.rel_vel_max_m_s', 0.0),
                 est_vel_window=_int(d, f'{p}.est_vel_window', 1) if 'est_vel_window' in _get(d, p, dict) else 1)


def parse(d):
    """A Level from the parsed YAML dict."""
    if not isinstance(d, dict):
        raise LevelError('level.yaml: top level must be a mapping')
    if _get(d, 'schema') != SCHEMA_VERSION:
        raise LevelError(f'level.yaml: schema {d.get("schema")!r}, this runtime reads {SCHEMA_VERSION}')
    source = str(_get(d, 'source'))

    arm_joints = _names(d, 'arm.joints', N_ARM)
    lo = _vec(d, 'observation.arm_q_limits.lo', N_ARM)
    hi = _vec(d, 'observation.arm_q_limits.hi', N_ARM)
    if np.any(hi <= lo):
        raise LevelError('level.yaml: observation.arm_q_limits hi must exceed lo on every joint')

    thumb = _names(d, 'hand.thumb_joints', N_THUMB)
    finger = _names(d, 'hand.finger_joints', N_SYNERGY - N_THUMB)
    if set(thumb) & set(finger):
        raise LevelError('level.yaml: hand.thumb_joints and hand.finger_joints overlap')
    hand_joints = thumb + finger          # the observation order: thumb first
    held_raw = _get(d, 'hand.held', dict)
    held_names = tuple(held_raw)
    if len(held_names) != N_HAND - N_SYNERGY:
        raise LevelError(f'level.yaml: hand.held must list {N_HAND - N_SYNERGY} joints, has {len(held_names)}')
    if set(held_names) & set(hand_joints):
        raise LevelError('level.yaml: hand.held repeats a synergy joint')
    held = {n: float(v) for n, v in zip(held_names, _table(d, 'hand.held', held_names))}

    tips = {}
    for k in TIP_NAMES:
        link = _get(d, f'hand.tips.{k}.link')
        if not isinstance(link, str) or not link:
            raise LevelError(f'level.yaml: hand.tips.{k}.link must be a link name')
        tips[k] = (link, _vec(d, f'hand.tips.{k}.offset', 3))

    wrist_xyz = _vec(d, 'wrist.offset.xyz', 3)
    wrist_rpy = _vec(d, 'wrist.offset.rpy', 3)
    wrist_link = _get(d, 'wrist.link')
    if not isinstance(wrist_link, str) or not wrist_link:
        raise LevelError('level.yaml: wrist.link must be a link name')

    p = 'phase'
    phase = Phase(
        settle_in=_int(d, f'{p}.settle_in'),
        grasp_max_steps=_int(d, f'{p}.grasp_max_steps', 1),
        grasp_retry_steps=_int(d, f'{p}.grasp_retry_steps', 1),
        settle_steps=_int(d, f'{p}.settle_steps'),
        drag_steps=_int(d, f'{p}.drag_steps'),
        free_prelift=_bool(d, f'{p}.free_prelift'),
        grip_check=_bool(d, f'{p}.grip_check'),
        grip_check_steps=_int(d, f'{p}.grip_check_steps'),
        grip_test_total_m=_num(d, f'{p}.grip_test_total_m', 0.0),
        grip_check_ratio=_num(d, f'{p}.grip_check_ratio', 0.0, 1.0),
        grip_check_min_tips_m=_num(d, f'{p}.grip_check_min_tips_m', 0.0),
        grip_check_late=_bool(d, f'{p}.grip_check_late'),
        lift_steps=_int(d, f'{p}.lift_steps', 1),
        lift_slow_steps=_int(d, f'{p}.lift_slow_steps'),
        lift_slow_step_m=_num(d, f'{p}.lift_slow_step_m', 0.0),
        lift_height_m=_num(d, f'{p}.lift_height_m', 0.0),
        hold_steps=_int(d, f'{p}.hold_steps'),
        retry_max=_int(d, f'{p}.retry_max'),
        success_lift_m=_num(d, f'{p}.success_lift_m', 0.0),
        up_threshold_m=_num(d, f'{p}.up_threshold_m', 0.0),
        lift_dir=_vec(d, f'{p}.lift_dir', 3),
        latch=_latch(d, f'{p}.latch'),
        sense_est=_bool(d, f'{p}.sense_est') if 'sense_est' in _get(d, p, dict) else False,
        roll_cap=_num(d, f'{p}.roll_cap', 0.0, math.pi) if 'roll_cap' in _get(d, p, dict) else math.pi / 2,
    )
    if phase.lift_slow_steps >= phase.lift_steps:
        raise LevelError('level.yaml: phase.lift_slow_steps must be below phase.lift_steps')
    if phase.grip_check and phase.grip_check_steps < 1:
        raise LevelError('level.yaml: phase.grip_check needs grip_check_steps >= 1')
    n = np.linalg.norm(phase.lift_dir)
    if n < 1e-9:
        raise LevelError('level.yaml: phase.lift_dir must not be zero')
    phase = Phase(**{**phase.__dict__, 'lift_dir': phase.lift_dir / n})

    groups = {g: _spheres(d, g) for g in GROUP_NAMES}
    dist_clamp = _vec(d, 'safety.dist_clamp', 2)
    if dist_clamp[0] >= dist_clamp[1]:
        raise LevelError('level.yaml: safety.dist_clamp must be [lo, hi] with lo < hi')
    def boxes(key, required):
        raw = _get(d, f'safety.{key}', list) if required or key in _get(d, 'safety', dict) else []
        out = []
        for i, box in enumerate(raw):
            if not isinstance(box, dict):
                raise LevelError(f'level.yaml: safety.{key}[{i}] must be {{lo, hi}}')
            blo, bhi = _vec(box, 'lo', 3), _vec(box, 'hi', 3)
            if np.any(bhi <= blo):
                raise LevelError(f'level.yaml: safety.{key}[{i}] hi must exceed lo')
            out.append(Box(blo, bhi))
        return out
    extra = boxes('extra_boxes', True)
    training = boxes('training_boxes', False)

    layout = _layout(d)
    level = Level(
        source=source,
        rate_hz=_num(d, 'control.rate_hz', 1.0, 1000.0),
        layout=layout,
        episode_max_steps=_int(d, 'observation.episode_max_steps', 1),
        tip_force_scale=_num(d, 'observation.tip_force_scale', 1e-6),
        effort_obs=((_num(d, 'observation.effort_obs.force_per_nm', 0.0), _num(d, 'observation.effort_obs.contact_nm', 0.0))
                    if 'effort_obs' in _get(d, 'observation', dict) else ()),
        arm_joints=arm_joints,
        arm_q_lo=lo, arm_q_hi=hi,
        effective_translation=_num(d, 'action.effective_translation', 0.0),
        effective_rotation=_num(d, 'action.effective_rotation', 0.0),
        wrist_smoothing=_num(d, 'action.wrist_smoothing', 0.0, 1.0),
        action_smoothing=_num(d, 'action.action_smoothing', 0.0, 1.0),
        yaw_rate_scale=_num(d, 'action.yaw_rate_scale', 0.0),
        dls_damping=_num(d, 'ik.dls_damping', 0.0),
        nullspace_gain=_num(d, 'ik.nullspace_gain', 0.0),
        joint_margin=_num(d, 'ik.joint_margin', 0.0),
        max_joint_velocity=_num(d, 'ik.max_joint_velocity', 0.0),
        singularity_threshold=_num(d, 'ik.singularity_threshold', 0.0),
        workspace_radius_xy=_num(d, 'ik.workspace.radius_xy', 0.0),
        workspace_z_min=_num(d, 'ik.workspace.z_min'),
        workspace_z_max=_num(d, 'ik.workspace.z_max'),
        nominal_wrist_offset_xy=_vec(d, 'ik.workspace.nominal_wrist_offset_xy', 2),
        wrist_link=wrist_link,
        wrist_offset=so3.transform(wrist_xyz, so3.rpy(*wrist_rpy)),
        hand_joints=hand_joints,
        hand_open=_table(d, 'hand.open', hand_joints),
        hand_closed=_table(d, 'hand.closed', hand_joints),
        hand_held=held,
        tips=tips,
        phase=phase,
        safety_groups=groups,
        safety_vec_clamp=_num(d, 'safety.vec_clamp', 0.0),
        safety_dist_clamp=(float(dist_clamp[0]), float(dist_clamp[1])),
        extra_boxes=tuple(extra),
        training_boxes=tuple(training),
        rod_radius=_num(d, 'rod.radius', 0.0),
        rod_half_len=_num(d, 'rod.half_len', 0.0),
        raw=d,
    )
    if level.workspace_z_max <= level.workspace_z_min:
        raise LevelError('level.yaml: ik.workspace.z_max must exceed z_min')
    return level


def load(path):
    """Parse level.yaml at `path`."""
    with open(path, encoding='utf-8') as f:
        try:
            d = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise LevelError(f'{path}: {e}') from e
    return parse(d)
