"""Which rod of a bin to pick: perception's candidates in, the one to send to the policy out.

The rule (adam-rl docs/PICKABILITY_REPORT.md: the level-51 policy picks 66.9 % of rods whose centre is 35 mm or
more above the bin floor, 17.8 % of those under 25 mm, 47.5 % of random exposed rods):

1. drop rods with another rod resting on them (axes closer than a diameter plus `resting_gap_m`, the other's
   centre higher by half a radius), rods whose axis is between leaning and standing, and rods outside the
   window the policy trained on (level.yaml `workspace`);
2. of the rest, take the highest centre, a standing rod counted `stand_penalty_m` lower.

adam-rl's level 52 trains on exactly this choice (FillCfg.pick == "highest"). No ROS here; selector_node wraps it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import so3

LYING, LEANING, STANDING, OTHER = 0, 1, 2, -1
KIND_NAMES = {LYING: 'lying', LEANING: 'leaning', STANDING: 'standing', OTHER: 'steep'}


class SelectorError(ValueError):
    """level.yaml has no usable `workspace` section."""


@dataclass(frozen=True)
class Workspace:
    """level.yaml `workspace`: where the policy can pick, and the rule's constants."""
    rod_xy_centre: np.ndarray
    xy_core: float
    xy_edge: float
    hand_yaw: tuple
    hand_yaw_core: tuple
    pinch_angle: float
    yaw0: float
    lying_max_deg: float
    leaning_max_deg: float
    standing_min_deg: float
    stand_penalty: float
    resting_gap: float
    table_top: float
    rod_radius: float
    rod_half_len: float

    @classmethod
    def from_level(cls, level) -> 'Workspace':
        w = level.raw.get('workspace') if isinstance(level.raw, dict) else None
        if not isinstance(w, dict):
            raise SelectorError(f'{level.source}: level.yaml has no workspace section; re-export it with adam-rl '
                                'tools/export_policy.py (2026-10-16 or later)')
        try:
            return cls(rod_xy_centre=np.asarray(w['rod_xy_centre'], float), xy_core=float(w['xy_core_m']),
                       xy_edge=float(w['xy_edge_m']), hand_yaw=tuple(map(float, w['hand_yaw_rad'])),
                       hand_yaw_core=tuple(map(float, w['hand_yaw_core_rad'])), pinch_angle=float(w['pinch_angle_rad']),
                       yaw0=float(w['yaw0_rad']), lying_max_deg=float(w['lying_max_deg']),
                       leaning_max_deg=float(w['leaning_max_deg']), standing_min_deg=float(w['standing_min_deg']),
                       stand_penalty=float(w['stand_penalty_m']), resting_gap=float(w['resting_gap_m']),
                       table_top=float(w['table_top_m']), rod_radius=float(level.rod_radius),
                       rod_half_len=float(level.rod_half_len))
        except (KeyError, TypeError, ValueError) as e:
            raise SelectorError(f'{level.source}: level.yaml workspace: {e}') from e


@dataclass(frozen=True)
class Choice:
    index: int                 # into the candidates; -1: none can be picked
    kind: int
    height: float              # centre above the table (m)
    hand_yaw: float            # a hand yaw the policy trained with for this rod (rad)
    reasons: tuple             # per candidate: 'picked', or why it was passed over


def rod_axes(quats_xyzw: np.ndarray) -> np.ndarray:
    """(N, 3) unit axes (the rod's local z), the raised end up."""
    ax = np.stack([so3.from_quat_xyzw(q)[:, 2] for q in np.asarray(quats_xyzw, float).reshape(-1, 4)]) if len(quats_xyzw) else np.zeros((0, 3))
    return np.where(ax[:, 2:3] < 0.0, -ax, ax)


def kinds(axes: np.ndarray, ws: Workspace) -> np.ndarray:
    elev = np.degrees(np.arcsin(np.clip(axes[:, 2], 0.0, 1.0)))
    return np.where(elev < ws.lying_max_deg, LYING, np.where(elev <= ws.leaning_max_deg, LEANING,
                    np.where(elev >= ws.standing_min_deg, STANDING, OTHER)))


def _seg_seg(p0, p1, q0, q1) -> float:
    """Closest distance between segments p0-p1 and q0-q1 (Ericson 5.1.9), as the training env computes it."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    s = np.clip((b * f - c * e) / max(a * e - b * b, 1e-12), 0.0, 1.0)
    t = (b * s + f) / max(e, 1e-12)
    if t < 0.0:
        s = np.clip(-c / max(a, 1e-12), 0.0, 1.0)
    elif t > 1.0:
        s = np.clip((b - c) / max(a, 1e-12), 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    return float(np.linalg.norm((p0 + s * d1) - (q0 + t * d2)))


def covered(pos: np.ndarray, axes: np.ndarray, ws: Workspace) -> np.ndarray:
    """(N,) True where another rod rests on this one."""
    n = len(pos)
    p0, p1 = pos - ws.rod_half_len * axes, pos + ws.rod_half_len * axes
    out = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and pos[j, 2] > pos[i, 2] + 0.5 * ws.rod_radius \
                    and _seg_seg(p0[i], p1[i], p0[j], p1[j]) < 2 * ws.rod_radius + ws.resting_gap:
                out[i] = True
                break
    return out


def _wrap(x, period):
    return (x + period / 2) % period - period / 2


def hand_yaw_for(pos: np.ndarray, axis: np.ndarray, kind: int, ws: Workspace) -> Optional[float]:
    """A hand yaw inside the trained window that reaches this rod, preferring the core window (where the xy limit is
    the wider one), or None."""
    d = np.abs(pos[:2] - ws.rod_xy_centre)
    lo, hi = ws.hand_yaw
    clo, chi = ws.hand_yaw_core
    if kind == STANDING:                                     # a standing rod's pinch works at any hand yaw
        cands = [(clo, chi)]
    else:
        psi = math.atan2(axis[1], axis[0]) - ws.yaw0          # the rod's yaw in the training env's terms
        period = math.pi if kind == LYING else 2 * math.pi    # a leaning rod's raised end fixes its direction
        cands = []
        for k in range(-2, 3):
            c = _wrap(psi, period) + k * period
            cands.append((c - ws.pinch_angle, c + ws.pinch_angle))
    for wlo, whi, lim in ((clo, chi, ws.xy_core), (lo, hi, ws.xy_edge)):
        if np.all(d <= lim):
            for a, b in cands:
                a_, b_ = max(a, wlo), min(b, whi)
                if a_ <= b_:
                    return 0.5 * (a_ + b_)
    return None


def select(positions: Sequence, quats_xyzw: Sequence, ws: Workspace) -> Choice:
    """positions (N, 3) and xyzw quaternions (N, 4) of every rod perception sees, in the world frame."""
    pos = np.asarray(positions, float).reshape(-1, 3)
    if len(pos) == 0:
        return Choice(-1, OTHER, 0.0, 0.0, ())
    axes = rod_axes(quats_xyzw)
    kd = kinds(axes, ws)
    cov = covered(pos, axes, ws)
    reasons, best, best_score, best_yaw = [], -1, -math.inf, 0.0
    for i in range(len(pos)):
        if cov[i]:
            reasons.append('another rod rests on it')
            continue
        if kd[i] == OTHER:
            reasons.append('too steep to lie or stand')
            continue
        yaw = hand_yaw_for(pos[i], axes[i], int(kd[i]), ws)
        if yaw is None:
            reasons.append('outside the reachable window')
            continue
        score = pos[i, 2] - (ws.stand_penalty if kd[i] == STANDING else 0.0)
        reasons.append('candidate')
        if score > best_score:
            best, best_score, best_yaw = i, score, yaw
    reasons = tuple('picked' if i == best else ('lower than the pick' if r == 'candidate' else r)
                    for i, r in enumerate(reasons))
    if best < 0:
        return Choice(-1, OTHER, 0.0, 0.0, reasons)
    return Choice(best, int(kd[best]), float(pos[best, 2] - ws.table_top), best_yaw, reasons)
