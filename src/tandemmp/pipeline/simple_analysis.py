"""Simple, PSF-fit-free pair matching for CSV-exported iSCAT point-events."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import StatisticsError
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import tqdm
from scipy.spatial import cKDTree

from tandemmp.core.conversions import NM_PER_PIXEL, c_to_mass, mass_to_sigma
from tandemmp.core.functions import get_overlap_probability
from tandemmp.core.parameters import FullAnalysisParams
from tandemmp.core.types import BoolArray, FloatArray
from tandemmp.drift.corrector import get_vector
from tandemmp.events import RatiometricEvent
from tandemmp.pipeline.loader import Loader

# ---------------------------------------------------------------------------
# SimpleEvent
# ---------------------------------------------------------------------------

_next_simple_event_id: int = 0


def _allocate_simple_event_id() -> int:
    """Return the next sequential :class:`SimpleEvent` ID."""
    global _next_simple_event_id  # noqa: PLW0603
    id_ = _next_simple_event_id
    _next_simple_event_id += 1
    return id_


def _reset_simple_event_ids() -> None:
    """Reset the :class:`SimpleEvent`-ID counter (for tests only)."""
    global _next_simple_event_id  # noqa: PLW0603
    _next_simple_event_id = 0


@dataclass(frozen=True)
class SimpleEvent:
    """A single point-event loaded directly from a CSV export.

    Unlike :class:`~tandemmp.events.ratiometric.RatiometricEvent`, this
    carries no peak image or PSF fit — just the position, contrast, mass and
    frame index already present in the source data.
    """

    c: float
    m: float
    x: float
    y: float
    i: float
    id: int = field(default_factory=_allocate_simple_event_id)


def build_simple_events(events: Sequence[RatiometricEvent]) -> list[SimpleEvent]:
    """Convert first-pass :class:`~tandemmp.events.RatiometricEvent`\\ s into :class:`SimpleEvent`\\ s.

    Uses each event's ``particle_c``/``particle_m``/``particle_x``/``particle_y``
    (basic-fit values, available immediately after
    :class:`~tandemmp.pipeline.loader.Loader` finds/fits first-pass events,
    with no PSF-fit-based tracking pass required). Shared by
    :meth:`tandemmp.pipeline.pca.Experiment.simple_analysis` and
    :func:`tandemmp.pipeline.pair_matching.run_pair_matching`.
    """  # noqa: E501
    return [
        SimpleEvent(
            c=e.particle_c,
            m=e.particle_m,
            x=e.particle_x,
            y=e.particle_y,
            i=float(e.i),
        )
        for e in events
    ]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_events(
    events_filepath: str | Path,
    movie_filepath: str | Path | None = None,
    *,
    parallel: bool = True,
) -> list[SimpleEvent]:
    """Load point-events from a CSV export.

    *events_filepath* must have a header row followed by
    ``frame, x, y, contrast, mass`` columns (extra trailing columns are
    ignored). If the mass column is entirely ``NaN``, mass is derived from
    contrast via :func:`~tandemmp.core.conversions.c_to_mass`.

    If *movie_filepath* is given, the companion movie is loaded (without
    running the full pipeline) purely to estimate a drift vector via
    :func:`~tandemmp.drift.corrector.get_vector`, which is then subtracted
    from every event's position.
    """
    vector = None
    if movie_filepath is not None:
        movie = Loader(
            movie_filepath,
            just_movie=True,
            parameters=FullAnalysisParams(drift_correction=False, parallel=parallel),
        ).movie
        vector = get_vector(
            movie=movie,
            frame_gap=50,
            rolling_baseline=True,
            threshold=0,
            threshold_margin=1,
            parallel=parallel,
        )

    data = np.loadtxt(events_filepath, delimiter=",", skiprows=1)
    frames, xs, ys, cs, masses = data.T[:5].astype(float)
    if np.isnan(masses).all():
        masses = c_to_mass(cs)

    if vector is None:
        return [
            SimpleEvent(c=c, m=m, x=x, y=y, i=i)
            for i, x, y, c, m in zip(frames, xs, ys, cs, masses, strict=True)
        ]

    return [
        SimpleEvent(c=c, m=m, x=x - vector[int(i)][0], y=y - vector[int(i)][1], i=i)
        for i, x, y, c, m in zip(frames, xs, ys, cs, masses, strict=True)
    ]


# ---------------------------------------------------------------------------
# Pair matching
# ---------------------------------------------------------------------------


class Pair(NamedTuple):
    binding: SimpleEvent
    unbinding: SimpleEvent
    time: float
    distance: float
    overlap: float
    order: int

    def __repr__(self) -> str:
        return (
            f"Pair(binding: id={self.binding.id}, m={self.binding.m}, "
            f"x={np.round(self.binding.x, 2)}, y={np.round(self.binding.y, 2)}. "
            f"unbinding: id={self.unbinding.id}, m={self.unbinding.m}, "
            f"x={np.round(self.unbinding.x, 2)}, y={np.round(self.unbinding.y, 2)}, "
            f"time: {self.time}, distance: {self.distance}, overlap: {self.overlap})"
        )


class NewPair(NamedTuple):
    precursor: float
    product: float
    time: float
    distance: float
    overlap: float
    order: int


def _mass_to_sigma_px(mass: float) -> float:
    """``mass_to_sigma`` (nm) converted to pixels for pixel-space overlap scoring.

    ``match_pairs_sigma`` / ``match_pairs_tally`` score overlap against
    pixel-space coordinates, so the nanometre width from ``mass_to_sigma``
    must be divided by ``NM_PER_PIXEL`` first. Feeding the raw nanometre
    value in (as this module previously did) makes the widths ~``NM_PER_PIXEL``
    times too large, saturating every candidate's overlap to ~1 and
    collapsing the match to "nearest in mass" rather than "nearest in space".
    """
    return mass_to_sigma(mass) / NM_PER_PIXEL


def _match_pairs_sigma_loop(
    events: Sequence[SimpleEvent],
    *,
    sigma_multiplier: float = 3,
    max_pairs: int = 5,
    analyse_distance: float = 4,
    minimum_overlap: float = 0.01,
    match_binding: bool = False,
) -> list[Pair]:
    """Reference implementation of :func:`match_pairs_sigma`.

    Scans every binding event for each unbinding event, so cost grows
    quadratically with the number of events. :func:`match_pairs_sigma` is
    the batched equivalent and delegates here for ``match_binding=True``;
    tests assert the two give bit-identical output.
    """
    events_arr: npt.NDArray[np.object_] = np.array(events, dtype=object)
    contrast: FloatArray = np.array([e.c for e in events])
    x: FloatArray = np.array([e.x for e in events])
    y: FloatArray = np.array([e.y for e in events])
    i: FloatArray = np.array([e.i for e in events])

    unbinding: BoolArray = contrast > 0
    binding: BoolArray = contrast < 0

    if match_binding:
        unbinding = binding

    pairs_left = np.full(len(events), max_pairs)
    pairs: list[Pair] = []

    for unbinding_event in events_arr[unbinding]:
        spatial_distance = np.hypot(
            x[binding] - unbinding_event.x, y[binding] - unbinding_event.y
        )

        time_between = unbinding_event.i - i[binding]

        suitable = (spatial_distance < analyse_distance) & (time_between > 0)

        if match_binding:
            suitable &= np.array([unbinding_event != e for e in events_arr[unbinding]])

        if not suitable.any():
            continue

        probabilities = np.array(
            [
                get_overlap_probability(
                    x1=unbinding_event.x,
                    y1=unbinding_event.y,
                    s1=_mass_to_sigma_px(unbinding_event.m) * sigma_multiplier,
                    x2=be.x,
                    y2=be.y,
                    s2=_mass_to_sigma_px(be.m) * sigma_multiplier,
                )
                for be in events_arr[binding][suitable]
            ]
        )

        over_minimum = probabilities > minimum_overlap

        if not over_minimum.any():
            continue

        candidate = np.argmax(probabilities[over_minimum])
        index = np.arange(len(events))[binding][suitable][over_minimum][candidate]
        closest_event = events_arr[index]
        available_pairs = pairs_left[index]

        if available_pairs > 0:
            pairs.append(
                Pair(
                    binding=closest_event,
                    unbinding=unbinding_event,
                    time=time_between[suitable][over_minimum][candidate],
                    distance=spatial_distance[suitable][over_minimum][candidate],
                    overlap=probabilities[over_minimum][candidate],
                    order=max_pairs - available_pairs,
                )
            )
            pairs_left[index] -= 1

    return pairs


_SQRT2 = math.sqrt(2.0)


def _normal_cdf(mu: float, sigma: float, x: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * _SQRT2)))


def _overlap_from_distance(s1: float, distance: float, s2: float) -> float:
    """``NormalDist(0, s1).overlap(NormalDist(distance, s2))`` without the objects.

    Mirrors ``statistics.NormalDist.overlap`` / ``cdf`` operation for
    operation using the same ``math`` functions, so results are
    bit-identical to :func:`~tandemmp.core.functions.get_overlap_probability`
    while skipping the per-call object construction. (A numpy/scipy
    vectorised form is *not* bit-identical: ``log``/``erf`` differ in the
    last bit, enough to flip near-ties in the argmax.)
    """
    if s1 < 0.0 or s2 < 0.0:
        msg = "sigma must be non-negative"
        raise StatisticsError(msg)

    x_sigma, x_mu, y_sigma, y_mu = s1, 0.0, s2, distance
    if (y_sigma, y_mu) < (x_sigma, x_mu):  # sort to assure commutativity
        x_sigma, x_mu, y_sigma, y_mu = y_sigma, y_mu, x_sigma, x_mu

    x_var, y_var = x_sigma * x_sigma, y_sigma * y_sigma
    if not x_var or not y_var:
        msg = "overlap() not defined when sigma is zero"
        raise StatisticsError(msg)

    dv = y_var - x_var
    dm = math.fabs(y_mu - x_mu)
    if not dv:
        return 1.0 - math.erf(dm / (2.0 * x_sigma * _SQRT2))

    a = x_mu * y_var - y_mu * x_var
    b = x_sigma * y_sigma * math.sqrt(dm * dm + dv * math.log(y_var / x_var))
    x1 = (a + b) / dv
    x2 = (a - b) / dv
    return 1.0 - (
        math.fabs(_normal_cdf(y_mu, y_sigma, x1) - _normal_cdf(x_mu, x_sigma, x1))
        + math.fabs(_normal_cdf(y_mu, y_sigma, x2) - _normal_cdf(x_mu, x_sigma, x2))
    )


def match_pairs_sigma(
    events: Sequence[SimpleEvent],
    *,
    sigma_multiplier: float = 3,
    max_pairs: int = 5,
    analyse_distance: float = 4,
    minimum_overlap: float = 0.01,
    match_binding: bool = False,
) -> list[Pair]:
    """Match each "unbinding" event to its most-overlapping "binding" event.

    Events with negative contrast are "binding", positive contrast are
    "unbinding" (or vice versa if *match_binding* is set, pairing binding
    events to other binding events instead). A candidate binding event must
    be within *analyse_distance* pixels, precede the unbinding event in
    time, and have a Gaussian overlap coefficient (using
    ``mass_to_sigma(mass) * sigma_multiplier`` as each event's width) above
    *minimum_overlap*; the highest-overlap candidate is chosen. Each binding
    event may be reused up to *max_pairs* times.

    Candidate (unbinding, binding) pairs are found with a KD-tree radius
    query and scored in one batch, rather than scanning every binding event
    for each unbinding event; only the final selection, which depends on how
    many pairs each binding event has left, runs sequentially. Output is
    bit-identical to the straightforward per-event scan
    (:func:`_match_pairs_sigma_loop`), which is still used for
    ``match_binding=True``.
    """
    if match_binding:
        return _match_pairs_sigma_loop(
            events,
            sigma_multiplier=sigma_multiplier,
            max_pairs=max_pairs,
            analyse_distance=analyse_distance,
            minimum_overlap=minimum_overlap,
            match_binding=match_binding,
        )

    events_arr: npt.NDArray[np.object_] = np.array(events, dtype=object)
    contrast: FloatArray = np.array([e.c for e in events])
    x: FloatArray = np.array([e.x for e in events])
    y: FloatArray = np.array([e.y for e in events])
    i: FloatArray = np.array([e.i for e in events])

    # Non-finite coordinates can never satisfy `distance < analyse_distance`
    # (and would make the KD-tree raise), so they are left out of the query.
    finite = np.isfinite(x) & np.isfinite(y)
    unbinding_all = np.flatnonzero(contrast > 0)
    unbinding_idx = unbinding_all[finite[unbinding_all]]
    binding_idx = np.flatnonzero((contrast < 0) & finite)

    pairs_left = np.full(len(events), max_pairs)
    pairs: list[Pair] = []

    if unbinding_idx.size == 0 or binding_idx.size == 0:
        return pairs

    # Candidate pairs. The radius is padded so the KD-tree's own distance
    # arithmetic can't drop a boundary pair; the exact `< analyse_distance`
    # test below uses np.hypot, as the reference loop does.
    tree = cKDTree(np.column_stack([x[binding_idx], y[binding_idx]]))
    radius = analyse_distance + 1e-6 * (1.0 + abs(analyse_distance))
    neighbours = tree.query_ball_point(
        np.column_stack([x[unbinding_idx], y[unbinding_idx]]),
        r=radius,
        return_sorted=True,
    )
    counts = np.fromiter((len(n) for n in neighbours), dtype=np.intp)
    if counts.sum() == 0:
        return pairs

    unbinding_of = np.repeat(unbinding_idx, counts)
    binding_of = binding_idx[
        np.concatenate([np.asarray(n, dtype=np.intp) for n in neighbours if len(n)])
    ]

    spatial_distance = np.hypot(
        x[binding_of] - x[unbinding_of], y[binding_of] - y[unbinding_of]
    )
    time_between = i[unbinding_of] - i[binding_of]
    suitable = (spatial_distance < analyse_distance) & (time_between > 0)
    unbinding_of, binding_of = unbinding_of[suitable], binding_of[suitable]
    spatial_distance, time_between = spatial_distance[suitable], time_between[suitable]
    if unbinding_of.size == 0:
        return pairs

    # Overlap of every candidate pair, with each event's width computed once.
    sigma_px = {
        int(k): _mass_to_sigma_px(events[k].m) * sigma_multiplier
        for k in np.unique(np.concatenate([unbinding_of, binding_of]))
    }
    # get_overlap_probability measures unbinding -> binding; np.hypot ignores sign.
    overlap_distance = np.hypot(
        x[unbinding_of] - x[binding_of], y[unbinding_of] - y[binding_of]
    )
    probabilities = np.array(
        [
            _overlap_from_distance(sigma_px[int(u)], float(d), sigma_px[int(b)])
            for u, b, d in zip(unbinding_of, binding_of, overlap_distance, strict=True)
        ]
    )
    over_minimum = probabilities > minimum_overlap

    # Sequential selection, in the same order as the reference loop:
    # `unbinding_of` is grouped by ascending unbinding index, and binding
    # indices ascend within each group, so ties resolve to the first maximum.
    group_start = np.searchsorted(unbinding_of, unbinding_all, side="left")
    group_end = np.searchsorted(unbinding_of, unbinding_all, side="right")

    for start, end in zip(group_start, group_end, strict=True):
        over = over_minimum[start:end]
        if not over.any():
            continue

        candidate = np.argmax(probabilities[start:end][over])
        k = start + np.flatnonzero(over)[candidate]
        index = binding_of[k]
        available_pairs = pairs_left[index]

        if available_pairs > 0:
            pairs.append(
                Pair(
                    binding=events_arr[index],
                    unbinding=events_arr[unbinding_of[k]],
                    time=time_between[k],
                    distance=spatial_distance[k],
                    overlap=probabilities[k],
                    order=max_pairs - available_pairs,
                )
            )
            pairs_left[index] -= 1

    return pairs


def match_pairs_tally(
    events: Sequence[SimpleEvent],
    *,
    sigma_multiplier: float = 3,
    max_pairs: int = 5,
    analyse_distance: float = 4,
    minimum_overlap: float = 0.05,
) -> list[NewPair]:
    """Chain each "binding" event onto its up-to-*max_pairs* closest later events.

    Unlike :func:`match_pairs_sigma` (which reports raw binding/unbinding
    event pairs), this accumulates a running mass total across the ordered
    chain of matches, reporting ``precursor``/``product`` mass for each
    step.
    """
    events_arr: npt.NDArray[np.object_] = np.array(events, dtype=object)
    contrast: FloatArray = np.array([e.c for e in events])
    x: FloatArray = np.array([e.x for e in events])
    y: FloatArray = np.array([e.y for e in events])
    i: FloatArray = np.array([e.i for e in events])

    binding: BoolArray = contrast < 0

    pairs: list[NewPair] = []

    for binding_event in tqdm.tqdm(
        events_arr[binding], desc="Matching events (simple)"
    ):
        spatial_distance = np.hypot(x - binding_event.x, y - binding_event.y)
        time_between = i - binding_event.i
        not_self = np.array([binding_event != e for e in events_arr])

        suitable = (spatial_distance < analyse_distance) & (time_between > 0) & not_self

        if not suitable.any():
            continue

        probabilities = np.array(
            [
                get_overlap_probability(
                    x1=binding_event.x,
                    y1=binding_event.y,
                    s1=_mass_to_sigma_px(binding_event.m) * sigma_multiplier,
                    x2=e.x,
                    y2=e.y,
                    s2=_mass_to_sigma_px(e.m) * sigma_multiplier,
                )
                for e in events_arr[suitable]
            ]
        )

        above_maximum = probabilities > minimum_overlap
        time_sort = np.argsort(time_between[suitable][above_maximum])
        indexes = np.arange(len(events))[suitable][above_maximum][time_sort]

        running_tally = binding_event.m
        for order, index in enumerate(indexes[:max_pairs]):
            unbinding_event = events_arr[index]

            pairs.append(
                NewPair(
                    precursor=running_tally,
                    product=running_tally + unbinding_event.m,
                    time=time_between[index],
                    distance=spatial_distance[index],
                    overlap=probabilities[above_maximum][time_sort[order]],
                    order=order,
                )
            )

            running_tally += unbinding_event.m

    return pairs


def convert_to_product_precursor(pairs: Sequence[Pair]) -> list[Pair]:
    """Rewrite raw binding/unbinding :class:`Pair`\\ s as a running mass chain.

    Pairs sharing the same ``binding`` event are grouped and sorted by
    ``order``; each step's precursor mass is the previous step's product
    mass (starting from the original binding event's mass), mirroring the
    accumulation :func:`match_pairs_tally` performs directly.
    """
    binding_events = np.array([p.binding for p in pairs], dtype=object)
    parents = set(binding_events)
    pairs_combined = {
        parent: sorted((p for p in pairs if p.binding == parent), key=lambda p: p.order)
        for parent in parents
    }

    new_pairs: list[Pair] = []

    for binding, unbindings in pairs_combined.items():
        running_m = binding.m
        running_c = binding.c

        for old_pair in unbindings:
            unbinding_event = old_pair.unbinding

            new_b = SimpleEvent(
                c=running_c, m=running_m, x=binding.x, y=binding.y, i=binding.i
            )
            new_u = SimpleEvent(
                c=running_c + unbinding_event.c,
                m=running_m + unbinding_event.m,
                x=unbinding_event.x,
                y=unbinding_event.y,
                i=unbinding_event.i,
            )

            new_pairs.append(
                Pair(
                    binding=new_b,
                    unbinding=new_u,
                    time=old_pair.time,
                    distance=old_pair.distance,
                    overlap=old_pair.overlap,
                    order=old_pair.order,
                )
            )

            running_m += unbinding_event.m
            running_c += unbinding_event.c

    return new_pairs


# ---------------------------------------------------------------------------
# Radial distribution function
# ---------------------------------------------------------------------------


def area_of_ring_in_box(
    origin: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    box_size: tuple[float, float],
    resolution: int = 4000,
) -> float:
    """Numerically estimate the area of an annulus clipped to a box.

    *origin* and *box_size* share the box's coordinate system (box spans
    ``[0, box_size[0]] x [0, box_size[1]]``); the annulus is sampled on a
    *resolution* x *resolution* grid.
    """
    x = np.linspace(0, box_size[0], resolution)
    y = np.linspace(0, box_size[1], resolution)
    xx, yy = np.meshgrid(x, y)
    distances = np.sqrt((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2)
    in_ring = (distances >= inner_radius) & (distances <= outer_radius)
    return float(
        np.sum(in_ring) * (box_size[0] / resolution) * (box_size[1] / resolution)
    )


def calculate_rdf(
    x: FloatArray,
    y: FloatArray,
    i: FloatArray,
    bins: FloatArray,
    sel_first: BoolArray | None = None,
    sel_after: BoolArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute a radial distribution function g(r) over ``bins``.

    For every point selected by *sel_first*, counts later-or-same-frame
    points (optionally restricted by *sel_after*) falling in each radial
    bin, normalised by the expected count under a uniform density. Points
    are first shifted so the bounding box starts at the origin.
    """
    if sel_first is None:
        sel_first = np.full_like(x, fill_value=True, dtype=bool)
    if sel_after is None:
        sel_after = np.full_like(x, fill_value=True, dtype=bool)

    x, y, i = np.array(x), np.array(y), np.array(i)

    x = x - x.min()
    y = y - y.min()
    total_area = np.prod((y.max(), x.max()))

    bin_areas = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)

    rdf = np.zeros(len(bins) - 1)
    rho = len(x[sel_after]) / total_area

    indexes = np.arange(len(x))[sel_first]

    for index in tqdm.tqdm(indexes, desc="Calculating RDF..."):
        px, py = x[index], y[index]
        time = i[index]

        not_self = np.arange(len(x)) != index
        is_after = i >= time
        relevant = not_self & is_after & sel_after

        if not relevant.any():
            continue

        distances = np.hypot(px - x, py - y)[relevant]
        normalisation_factor = relevant.sum() * rho * bin_areas
        rdf += np.histogram(distances, bins=bins)[0] / normalisation_factor

    return rdf, (bins[1:] + bins[:-1]) / 2
