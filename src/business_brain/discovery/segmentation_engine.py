"""Segmentation engine -- k-means style clustering on numeric data.

Pure-function module.  No external dependencies (no numpy, scipy, sklearn).
Uses k-means++ initialisation and min-max normalisation for deterministic,
reproducible clustering of tabular rows.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single cluster produced by segmentation."""

    segment_id: int
    label: str  # auto-generated, e.g. "High revenue, Low cost"
    size: int
    center: dict[str, float]  # centroid values per feature (original scale)
    spread: dict[str, float]  # std-dev per feature (original scale)
    members: list[int] = field(default_factory=list)  # row indices


@dataclass
class SegmentationResult:
    """Full output of a segmentation run."""

    segments: list[Segment]
    n_segments: int
    total_rows: int
    features: list[str]
    quality_score: float  # 0-1, silhouette-like measure
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_numeric(rows: list[dict], features: list[str]) -> tuple[list[list[float]], list[str]]:
    """Extract numeric columns from *rows*.

    Non-numeric / missing values are replaced with ``None`` and then
    back-filled with the column mean.  Features that are entirely
    non-numeric are dropped.

    Returns (matrix, valid_features) where *matrix* is a list of
    float-lists and *valid_features* lists the features that survived.
    """
    n = len(rows)
    valid_features: list[str] = []
    columns: list[list[float | None]] = []

    for feat in features:
        col: list[float | None] = []
        for row in rows:
            val = row.get(feat)
            if val is None:
                col.append(None)
            elif isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                col.append(float(val))
            else:
                try:
                    col.append(float(val))
                except (ValueError, TypeError):
                    col.append(None)

        # Keep feature only if at least one numeric value exists
        numeric_count = sum(1 for v in col if v is not None)
        if numeric_count == 0:
            continue

        # Fill None with column mean
        col_mean = sum(v for v in col if v is not None) / numeric_count
        col = [v if v is not None else col_mean for v in col]
        columns.append(col)
        valid_features.append(feat)

    if not valid_features:
        return [], []

    # Transpose: columns -> row-major matrix
    matrix = [[columns[j][i] for j in range(len(valid_features))] for i in range(n)]
    return matrix, valid_features


def _min_max_normalize(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Normalise each feature to [0, 1].

    Returns (normalised_matrix, mins, maxs).
    Constant features (max == min) are set to 0.5 for every row.
    """
    if not matrix or not matrix[0]:
        return [], [], []

    n_features = len(matrix[0])
    mins = [min(row[j] for row in matrix) for j in range(n_features)]
    maxs = [max(row[j] for row in matrix) for j in range(n_features)]

    normalised: list[list[float]] = []
    for row in matrix:
        new_row: list[float] = []
        for j in range(n_features):
            span = maxs[j] - mins[j]
            if span == 0.0:
                new_row.append(0.5)
            else:
                new_row.append((row[j] - mins[j]) / span)
        normalised.append(new_row)

    return normalised, mins, maxs


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _mean_point(points: list[list[float]], n_features: int) -> list[float]:
    if not points:
        return [0.0] * n_features
    return [sum(p[j] for p in points) / len(points) for j in range(n_features)]


def _std_dev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# k-means++ initialisation
# ---------------------------------------------------------------------------

def _kmeans_plus_plus_init(
    matrix: list[list[float]],
    k: int,
    rng: random.Random,
) -> list[list[float]]:
    """Select *k* initial centres using k-means++ strategy."""
    n = len(matrix)
    first_idx = rng.randint(0, n - 1)
    centres: list[list[float]] = [list(matrix[first_idx])]

    for _ in range(1, k):
        # Distance of each point to nearest existing centre
        dists: list[float] = []
        for row in matrix:
            d = min(_euclidean(row, c) for c in centres)
            dists.append(d * d)  # squared distance as weight

        total = sum(dists)
        if total == 0.0:
            # All points identical -- just pick next index
            idx = rng.randint(0, n - 1)
        else:
            # Weighted random selection
            threshold = rng.random() * total
            cumulative = 0.0
            idx = 0
            for i, d in enumerate(dists):
                cumulative += d
                if cumulative >= threshold:
                    idx = i
                    break

        centres.append(list(matrix[idx]))

    return centres


# ---------------------------------------------------------------------------
# Core k-means
# ---------------------------------------------------------------------------

def _kmeans(
    matrix: list[list[float]],
    k: int,
    max_iterations: int,
    rng: random.Random,
) -> tuple[list[int], list[list[float]]]:
    """Run k-means clustering.

    Returns (assignments, centres).
    """
    n = len(matrix)
    n_features = len(matrix[0])
    centres = _kmeans_plus_plus_init(matrix, k, rng)
    assignments: list[int] = [0] * n

    for _ in range(max_iterations):
        # Assign step
        new_assignments: list[int] = []
        for row in matrix:
            best_c = 0
            best_d = float("inf")
            for ci, centre in enumerate(centres):
                d = _euclidean(row, centre)
                if d < best_d:
                    best_d = d
                    best_c = ci
            new_assignments.append(best_c)

        # Check convergence
        if new_assignments == assignments:
            assignments = new_assignments
            break
        assignments = new_assignments

        # Update step -- recompute centres
        for ci in range(k):
            members = [matrix[i] for i in range(n) if assignments[i] == ci]
            if members:
                centres[ci] = _mean_point(members, n_features)
        # If a cluster lost all members, re-seed it
        for ci in range(k):
            count = sum(1 for a in assignments if a == ci)
            if count == 0:
                centres[ci] = list(matrix[rng.randint(0, n - 1)])

    return assignments, centres


# ---------------------------------------------------------------------------
# Quality metric (silhouette-like)
# ---------------------------------------------------------------------------

def _quality_score(
    matrix: list[list[float]],
    assignments: list[int],
    centres: list[list[float]],
    k: int,
) -> float:
    """Compute a quality score in [0, 1].

    Uses ratio of mean inter-cluster distance to (mean intra-cluster distance
    + mean inter-cluster distance).  Higher means clusters are well separated.
    """
    n = len(matrix)
    if k <= 1 or n <= k:
        return 0.0

    # Mean intra-cluster distance (points to own centroid)
    intra_total = 0.0
    intra_count = 0
    for i in range(n):
        intra_total += _euclidean(matrix[i], centres[assignments[i]])
        intra_count += 1
    mean_intra = intra_total / max(intra_count, 1)

    # Mean inter-cluster distance (between centroids)
    inter_total = 0.0
    inter_count = 0
    for ci in range(k):
        for cj in range(ci + 1, k):
            inter_total += _euclidean(centres[ci], centres[cj])
            inter_count += 1
    mean_inter = inter_total / max(inter_count, 1)

    denom = mean_intra + mean_inter
    if denom == 0.0:
        return 0.0
    return mean_inter / denom


# ---------------------------------------------------------------------------
# De-normalisation helpers
# ---------------------------------------------------------------------------

def _denormalize_value(val: float, feat_min: float, feat_max: float) -> float:
    span = feat_max - feat_min
    if span == 0.0:
        return feat_min  # constant feature
    return val * span + feat_min


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_data(
    rows: list[dict],
    features: list[str],
    n_segments: int = 3,
    max_iterations: int = 50,
) -> SegmentationResult | None:
    """Cluster *rows* into segments based on numeric *features*.

    Returns ``None`` if insufficient data (fewer than ``n_segments * 2``
    rows or fewer than 1 usable numeric feature).

    Features are normalised to [0, 1] before clustering.  A seeded RNG
    (seed=42) is used for deterministic behaviour.
    """
    if not rows or not features:
        return None

    matrix, valid_features = _extract_numeric(rows, features)
    if not valid_features or not matrix:
        return None

    n = len(matrix)
    if n < n_segments * 2:
        return None

    # Clamp n_segments to sensible bounds
    k = max(1, min(n_segments, n))

    # Normalise
    norm_matrix, mins, maxs = _min_max_normalize(matrix)

    n_features = len(valid_features)

    # Run k-means
    rng = random.Random(42)
    assignments, centres_norm = _kmeans(norm_matrix, k, max_iterations, rng)

    # Build Segment objects
    segments: list[Segment] = []
    for ci in range(k):
        member_indices = [i for i in range(n) if assignments[i] == ci]
        if not member_indices:
            continue

        # De-normalise centre
        centre_orig: dict[str, float] = {}
        for j, feat in enumerate(valid_features):
            centre_orig[feat] = _denormalize_value(centres_norm[ci][j], mins[j], maxs[j])

        # Compute spread (std dev in original scale)
        spread_orig: dict[str, float] = {}
        for j, feat in enumerate(valid_features):
            col_vals = [matrix[i][j] for i in member_indices]
            col_mean = sum(col_vals) / len(col_vals)
            spread_orig[feat] = _std_dev(col_vals, col_mean)

        seg = Segment(
            segment_id=ci,
            label="",
            size=len(member_indices),
            center=centre_orig,
            spread=spread_orig,
            members=member_indices,
        )
        segments.append(seg)

    # Re-number segment_ids contiguously (in case empty clusters were dropped)
    for idx, seg in enumerate(segments):
        seg.segment_id = idx

    # Label segments
    segments = label_segments(segments, valid_features)

    # Quality
    q = _quality_score(norm_matrix, assignments, centres_norm, k)

    # Summary
    seg_desc = "; ".join(
        f"Segment {s.segment_id} ({s.label}): {s.size} rows"
        for s in segments
    )
    summary = (
        f"Segmented {n} rows into {len(segments)} segments "
        f"using {len(valid_features)} features. "
        f"Quality score: {q:.2f}. {seg_desc}."
    )

    return SegmentationResult(
        segments=segments,
        n_segments=len(segments),
        total_rows=n,
        features=valid_features,
        quality_score=q,
        summary=summary,
    )


def label_segments(segments: list[Segment], features: list[str]) -> list[Segment]:
    """Auto-label segments based on centroid values relative to overall mean.

    Features that are notably above or below the global average are
    highlighted.  The threshold is 0.3 standard deviations of centroid
    values for that feature across segments.
    """
    if not segments or not features:
        return segments

    # Compute global mean per feature (mean of centroids, weighted by size)
    total_size = sum(s.size for s in segments)
    if total_size == 0:
        return segments

    global_mean: dict[str, float] = {}
    for feat in features:
        global_mean[feat] = sum(
            s.center.get(feat, 0.0) * s.size for s in segments
        ) / total_size

    # Compute std-dev of centroid values across segments per feature
    centroid_std: dict[str, float] = {}
    for feat in features:
        vals = [s.center.get(feat, 0.0) for s in segments]
        if len(vals) < 2:
            centroid_std[feat] = 0.0
        else:
            m = sum(vals) / len(vals)
            centroid_std[feat] = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    for seg in segments:
        parts_high: list[str] = []
        parts_low: list[str] = []

        for feat in features:
            diff = seg.center.get(feat, 0.0) - global_mean[feat]
            threshold = centroid_std[feat] * 0.3 if centroid_std[feat] > 0 else 0.0

            if threshold == 0.0:
                continue  # constant feature -- skip

            if diff > threshold:
                parts_high.append(feat)
            elif diff < -threshold:
                parts_low.append(feat)

        parts: list[str] = []
        if parts_high:
            parts.append("High " + ", ".join(parts_high))
        if parts_low:
            parts.append("Low " + ", ".join(parts_low))

        if parts:
            seg.label = "; ".join(parts)
        else:
            seg.label = f"Average (segment {seg.segment_id})"

    return segments


def find_segment_drivers(segments: list[Segment], features: list[str]) -> list[dict]:
    """Find which features differentiate segments the most.

    Returns a list of dicts sorted by importance descending::

        [{"feature": "revenue", "importance": 0.85,
          "range_across_segments": 450.0}, ...]

    Importance is the range of centroid values normalised to [0, 1]
    relative to the maximum range observed across all features.
    """
    if not segments or not features or len(segments) < 2:
        return []

    results: list[dict] = []
    max_range = 0.0

    for feat in features:
        vals = [s.center.get(feat, 0.0) for s in segments]
        feat_range = max(vals) - min(vals)
        if feat_range > max_range:
            max_range = feat_range
        results.append({"feature": feat, "range_across_segments": feat_range})

    # Normalise importance
    for r in results:
        if max_range > 0:
            r["importance"] = r["range_across_segments"] / max_range
        else:
            r["importance"] = 0.0

    results.sort(key=lambda x: x["importance"], reverse=True)
    return results


def compare_segments(seg_a: Segment, seg_b: Segment, features: list[str]) -> dict:
    """Compare two segments.

    Returns a dict with:

    - ``dominant_a``: features where A's centroid > B's by a notable margin.
    - ``dominant_b``: features where B's centroid > A's by a notable margin.
    - ``similar``: features where both are close.
    - ``distance``: euclidean distance between centroids (original scale).
    """
    dominant_a: list[str] = []
    dominant_b: list[str] = []
    similar: list[str] = []

    vec_a: list[float] = []
    vec_b: list[float] = []

    for feat in features:
        va = seg_a.center.get(feat, 0.0)
        vb = seg_b.center.get(feat, 0.0)
        vec_a.append(va)
        vec_b.append(vb)

        # Determine threshold: 10% of the larger absolute value, with a
        # tiny floor so zero-centred values still classify sensibly.
        magnitude = max(abs(va), abs(vb), 1e-9)
        threshold = magnitude * 0.10

        diff = va - vb
        if diff > threshold:
            dominant_a.append(feat)
        elif diff < -threshold:
            dominant_b.append(feat)
        else:
            similar.append(feat)

    distance = _euclidean(vec_a, vec_b)

    return {
        "dominant_a": dominant_a,
        "dominant_b": dominant_b,
        "similar": similar,
        "distance": distance,
    }
