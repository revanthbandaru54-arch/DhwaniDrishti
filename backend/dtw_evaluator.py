"""
ISL-STEM Coach Evaluation Engine (DTW + Vector Cosine Distance)
===============================================================
Implements Dynamic Time Warping (DTW) with Sakoe-Chiba band constraint and
Vector Cosine Distance to compare a student's live multi-frame MediaPipe
Holistic trajectories against standard ISL STEM benchmark vectors.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.spatial.distance import cdist
except ImportError:
    def cdist(xa: np.ndarray, xb: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        return np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)


# Key anatomical joints monitored for live Green/Red overlay painting and angular coaching
TRACKED_JOINTS = [
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_index_mcp",
    "right_index_tip",
    "right_thumb_tip",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_index_mcp",
    "left_index_tip",
    "eyebrows",
]


def _generate_benchmark_trajectory(
    num_frames: int,
    right_arc: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    left_arc: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    palm_normal_deg: float,
    eyebrow_elevation: float,
    wave_freq: float = 1.0,
) -> np.ndarray:
    """
    Generates a canonical 30-frame benchmark feature trajectory of shape (T, 36)
    representing 12 key 3D anatomical landmarks normalized to shoulder-center coordinates.
    """
    t = np.linspace(0.0, 1.0, num_frames)
    frames = np.zeros((num_frames, len(TRACKED_JOINTS), 3), dtype=np.float64)

    r_start, r_end = np.array(right_arc[0]), np.array(right_arc[1])
    l_start, l_end = np.array(left_arc[0]), np.array(left_arc[1])

    for i, alpha in enumerate(t):
        osc = 0.04 * math.sin(2.0 * math.pi * wave_freq * alpha)
        r_wrist = (1.0 - alpha) * r_start + alpha * r_end + np.array([0.0, osc, 0.0])
        l_wrist = (1.0 - alpha) * l_start + alpha * l_end + np.array([0.0, -osc * 0.5, 0.0])

        # Shoulders (fixed anchor in normalized torso frame)
        frames[i, 0] = [-0.18, 0.30, 0.0]  # right_shoulder
        frames[i, 6] = [0.18, 0.30, 0.0]   # left_shoulder

        # Elbows (interpolated between shoulder and wrist)
        frames[i, 1] = 0.5 * (frames[i, 0] + r_wrist) + np.array([-0.06, 0.12, 0.04])
        frames[i, 7] = 0.5 * (frames[i, 6] + l_wrist) + np.array([0.06, 0.12, 0.04])

        # Wrists
        frames[i, 2] = r_wrist
        frames[i, 8] = l_wrist

        # Hand orientation offsets determined by canonical palm angle (isotropic XY plane)
        rad = math.radians(palm_normal_deg)
        hand_dir = np.array([0.08 * math.cos(rad), -0.08 * math.sin(rad), -0.04])
        frames[i, 3] = r_wrist + hand_dir * 0.6                       # right_index_mcp
        frames[i, 4] = r_wrist + hand_dir * 1.2                       # right_index_tip
        frames[i, 5] = r_wrist + np.array([0.05, -0.05, -0.03])       # right_thumb_tip

        frames[i, 9] = l_wrist + np.array([-0.05, -0.04, -0.02])      # left_index_mcp
        frames[i, 10] = l_wrist + np.array([-0.09, -0.07, -0.04])     # left_index_tip

        # Eyebrows (Non-Manual Marker elevation in y)
        frames[i, 11] = [0.0, 0.12 - eyebrow_elevation * 0.03, 0.0]

    return frames.reshape(num_frames, -1)


# Canonical ISL STEM Benchmark Dictionary
STEM_BENCHMARKS: Dict[str, Dict] = {
    "Algorithm": {
        "id": "Algorithm",
        "title": "Algorithm (అల్గారిథమ్ / एल्गोरिदम)",
        "category": "Computer Science & Data Structures",
        "difficulty": "Intermediate",
        "isl_description": (
            "Hold non-dominant left palm flat as a reference register; step the right index "
            "and middle fingers forward in sequential logic tiers while maintaining a 25° outward right palm angle."
        ),
        "target_palm_angle_deg": 25.0,
        "target_eyebrow_raise": 0.35,
        "trajectory": _generate_benchmark_trajectory(
            num_frames=30,
            right_arc=((-0.12, 0.52, -0.10), (0.05, 0.38, -0.22)),
            left_arc=((0.10, 0.55, -0.12), (0.10, 0.52, -0.12)),
            palm_normal_deg=25.0,
            eyebrow_elevation=0.35,
            wave_freq=2.0,
        ),
    },
    "Derivative": {
        "id": "Derivative",
        "title": "Derivative — dy/dx (అవకలనం / अवकलन)",
        "category": "Calculus & Mathematical Analysis",
        "difficulty": "Advanced",
        "isl_description": (
            "Form a curved function arc with the left hand; glide the right index finger "
            "along the tangent slope at a 40° inclination while raising eyebrows to emphasize instantaneous rate of change."
        ),
        "target_palm_angle_deg": 40.0,
        "target_eyebrow_raise": 0.55,
        "trajectory": _generate_benchmark_trajectory(
            num_frames=30,
            right_arc=((-0.18, 0.62, -0.08), (0.08, 0.32, -0.18)),
            left_arc=((0.06, 0.48, -0.14), (0.08, 0.48, -0.14)),
            palm_normal_deg=40.0,
            eyebrow_elevation=0.55,
            wave_freq=1.0,
        ),
    },
    "Photosynthesis": {
        "id": "Photosynthesis",
        "title": "Photosynthesis (కిరణజన్య సంయోగక్రియ / प्रकाश संश्लेषण)",
        "category": "Biology & Biochemistry",
        "difficulty": "Beginner",
        "isl_description": (
            "Hold the left hand horizontally as a plant leaf; shower downward light rays "
            "from an elevated right hand towards the leaf surface, then bloom both hands upward."
        ),
        "target_palm_angle_deg": 15.0,
        "target_eyebrow_raise": 0.45,
        "trajectory": _generate_benchmark_trajectory(
            num_frames=30,
            right_arc=((-0.20, 0.22, -0.16), (-0.02, 0.50, -0.12)),
            left_arc=((0.12, 0.54, -0.10), (0.06, 0.42, -0.15)),
            palm_normal_deg=15.0,
            eyebrow_elevation=0.45,
            wave_freq=1.5,
        ),
    },
    "Momentum": {
        "id": "Momentum",
        "title": "Momentum — p = mv (ద్రవ్యవేగం / संवेग)",
        "category": "Classical Mechanics (Physics)",
        "difficulty": "Intermediate",
        "isl_description": (
            "Clench the left fist to signify mass (m), then drive the right open palm "
            "horizontally forward with a brisk velocity vector (v) aligned at 0° pitch."
        ),
        "target_palm_angle_deg": 0.0,
        "target_eyebrow_raise": 0.25,
        "trajectory": _generate_benchmark_trajectory(
            num_frames=30,
            right_arc=((-0.22, 0.48, -0.06), (0.14, 0.48, -0.24)),
            left_arc=((0.10, 0.50, -0.12), (0.10, 0.50, -0.12)),
            palm_normal_deg=0.0,
            eyebrow_elevation=0.25,
            wave_freq=1.0,
        ),
    },
    "Recursion": {
        "id": "Recursion",
        "title": "Recursion (పునరావృతం / पुनरावृत्ति)",
        "category": "Computer Science & Algorithms",
        "difficulty": "Advanced",
        "isl_description": (
            "Rotate both index fingers in nested inward spirals converging toward a base-case "
            "stop gesture with a sharp eyebrow raise on termination."
        ),
        "target_palm_angle_deg": 30.0,
        "target_eyebrow_raise": 0.60,
        "trajectory": _generate_benchmark_trajectory(
            num_frames=30,
            right_arc=((-0.14, 0.42, -0.14), (-0.04, 0.46, -0.20)),
            left_arc=((0.14, 0.44, -0.14), (0.04, 0.46, -0.20)),
            palm_normal_deg=30.0,
            eyebrow_elevation=0.60,
            wave_freq=3.0,
        ),
    },
}


def compute_dtw_with_path(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    sakoe_chiba_band: Optional[int] = 8,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Computes Dynamic Time Warping (DTW) distance and optimal warping alignment path
    between two multivariate time series of shape (T1, D) and (T2, D).

    Uses Euclidean local cost matrix with an optional Sakoe-Chiba band constraint
    for O(T * band) efficiency.
    """
    n, m = seq_a.shape[0], seq_b.shape[0]
    local_cost = cdist(seq_a, seq_b, metric="euclidean")

    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw_matrix[0, 0] = 0.0

    band = sakoe_chiba_band if sakoe_chiba_band is not None else max(n, m)
    band = max(band, abs(n - m) + 2)

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m + 1, i + band + 1)
        for j in range(j_start, j_end):
            cost = local_cost[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    # Backtrack optimal warping path
    i, j = n, m
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [
            (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
            (dtw_matrix[i - 1, j], i - 1, j),
            (dtw_matrix[i, j - 1], i, j - 1),
        ]
        _, i, j = min(steps, key=lambda x: x[0])

    path.reverse()
    normalized_dtw = float(dtw_matrix[n, m] / max(len(path), 1))
    return normalized_dtw, path


def compute_warped_cosine_similarity(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    path: List[Tuple[int, int]],
) -> Tuple[float, float]:
    """
    Computes mean Vector Cosine Similarity and Vector Cosine Distance (1 - cos_sim)
    across the DTW-aligned frame pairs.
    """
    similarities: List[float] = []
    for idx_a, idx_b in path:
        vec_a = seq_a[idx_a]
        vec_b = seq_b[idx_b]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            similarities.append(0.0)
        else:
            cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            similarities.append(max(-1.0, min(1.0, cos_sim)))

    mean_sim = float(np.mean(similarities)) if similarities else 0.0
    cosine_distance = 1.0 - mean_sim
    return mean_sim, cosine_distance


def _compute_palm_angle_deg(joints_3d: np.ndarray) -> float:
    """
    Computes the student's right palm inclination angle in degrees from
    wrist (idx 2) -> index_tip (idx 4) in the frontal XY plane.
    """
    wrist = joints_3d[-1, 2]
    index_tip = joints_3d[-1, 4]
    delta = index_tip - wrist
    angle_rad = math.atan2(-delta[1], delta[0] + 1e-6)
    return math.degrees(angle_rad)


def evaluate_stem_sign(
    term_id: str,
    student_sequence: List[List[float]],
    measured_palm_angle_deg: Optional[float] = None,
    measured_eyebrow_raise: Optional[float] = None,
) -> Dict:
    """
    Evaluates a student's live trajectory against the benchmark for `term_id` using
    DTW and Vector Cosine Distance, returning joint-by-joint Green/Red status and
    actionable coaching hints.
    """
    benchmark = STEM_BENCHMARKS.get(term_id, STEM_BENCHMARKS["Algorithm"])
    ref_traj: np.ndarray = benchmark["trajectory"]  # shape (30, 36)

    arr = np.array(student_sequence, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Align feature dimension to 36 (12 tracked joints * 3 coords)
    target_dim = ref_traj.shape[1]
    if arr.shape[1] < target_dim:
        pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=np.float64)
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > target_dim:
        arr = arr[:, :target_dim]

    dtw_dist, path = compute_dtw_with_path(arr, ref_traj)
    cos_sim, cos_dist = compute_warped_cosine_similarity(arr, ref_traj, path)

    # Per-joint spatial error calculation across aligned path
    student_joints = arr.reshape(arr.shape[0], len(TRACKED_JOINTS), 3)
    ref_joints = ref_traj.reshape(ref_traj.shape[0], len(TRACKED_JOINTS), 3)

    joint_errors: Dict[str, float] = {}
    joint_status: Dict[str, str] = {}
    joint_threshold = 0.14  # Normalized torso space threshold for Green vs Red

    for j_idx, j_name in enumerate(TRACKED_JOINTS):
        errs = [
            float(np.linalg.norm(student_joints[ia, j_idx] - ref_joints[ib, j_idx]))
            for ia, ib in path
        ]
        mean_err = float(np.mean(errs)) if errs else 0.0
        joint_errors[j_name] = round(mean_err, 4)
        joint_status[j_name] = "green" if mean_err <= joint_threshold else "red"

    # Compute angular palm deviation & non-manual eyebrow deviation
    if measured_palm_angle_deg is None:
        measured_palm_angle_deg = _compute_palm_angle_deg(student_joints)

    target_angle = float(benchmark["target_palm_angle_deg"])
    angle_diff = target_angle - float(measured_palm_angle_deg)

    if measured_eyebrow_raise is None:
        # Derive from tracked eyebrow landmark y-offset
        measured_eyebrow_raise = float(max(0.0, min(1.0, (0.12 - student_joints[-1, 11, 1]) / 0.03)))

    target_eyebrow = float(benchmark["target_eyebrow_raise"])
    eyebrow_diff = target_eyebrow - float(measured_eyebrow_raise)

    # Generate actionable, biomechanical coaching hints
    actionable_hints: List[str] = []
    if abs(angle_diff) > 8.0:
        direction = "outward" if angle_diff > 0 else "inward"
        deg_int = int(round(abs(angle_diff)))
        actionable_hints.append(f"Angle your right palm {deg_int}° {direction} (Target: {int(target_angle)}°)")
        joint_status["right_wrist"] = "red"
        joint_status["right_index_mcp"] = "red"
        joint_status["right_index_tip"] = "red"

    if eyebrow_diff > 0.15:
        actionable_hints.append("Raise your eyebrows to mark the ISL STEM topic non-manual boundary")
        joint_status["eyebrows"] = "red"

    if joint_status.get("right_elbow") == "red":
        actionable_hints.append("Tuck your right elbow 15° closer to your torso for a cleaner signing plane")

    if joint_status.get("left_wrist") == "red":
        actionable_hints.append("Stabilize your non-dominant left base hand at chest height")

    # Overall composite score combining DTW trajectory alignment, Cosine Similarity, and angular accuracy
    dtw_score = max(0.0, 100.0 * math.exp(-1.65 * dtw_dist))
    cos_score = max(0.0, min(100.0, ((cos_sim + 1.0) / 2.0) * 100.0))
    angle_penalty = min(25.0, abs(angle_diff) * 0.45)
    overall_score = round(max(0.0, min(100.0, 0.55 * dtw_score + 0.45 * cos_score - angle_penalty)), 1)

    if not actionable_hints:
        actionable_hints.append(
            f"Excellent ISL execution for '{benchmark['id']}'! Trajectory and palm orientation are locked in."
        )

    return {
        "term_id": benchmark["id"],
        "title": benchmark["title"],
        "category": benchmark["category"],
        "dtw_distance": round(dtw_dist, 4),
        "cosine_similarity": round(cos_sim, 4),
        "cosine_distance": round(cos_dist, 4),
        "overall_score": overall_score,
        "passed": overall_score >= 75.0,
        "measured_palm_angle_deg": round(float(measured_palm_angle_deg), 1),
        "target_palm_angle_deg": target_angle,
        "joint_status": joint_status,
        "joint_errors": joint_errors,
        "actionable_hints": actionable_hints,
        "alignment_path_length": len(path),
    }


def list_stem_benchmarks() -> List[Dict]:
    """Returns metadata and benchmark reference trajectories for the frontend STEM Coach."""
    results = []
    for key, item in STEM_BENCHMARKS.items():
        results.append(
            {
                "id": item["id"],
                "title": item["title"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "isl_description": item["isl_description"],
                "target_palm_angle_deg": item["target_palm_angle_deg"],
                "target_eyebrow_raise": item["target_eyebrow_raise"],
                "benchmark_trajectory": item["trajectory"].tolist(),
            }
        )
    return results
