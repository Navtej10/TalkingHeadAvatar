"""
Audio-to-Motion Module

Converts per-frame audio features (mel energy + Wav2Vec2 embeddings + viseme classes)
into a sequence of LivePortrait keypoint deltas (shape: [num_frames, 21, 3]).

Design goals:
  - No driving video dependency.
  - Jaw open/close driven by per-frame audio energy (mouth opens when energy is high).
  - Viseme class modulates jaw shape (different vowels/consonants get different widths).
  - Subtle eye-blink and brow motion driven by prosody peaks.
  - Smooth, temporally coherent head sway (low-frequency sinusoidal + smoothed noise).
  - All outputs are relative *deltas* on top of the source keypoints — no absolute pose
    from a foreign face is ever applied.

LivePortrait 21-point keypoint layout (approximate):
  0-4:   Jaw / chin lower face
  5-6:   Left/right outer cheek corners
  7-8:   Left/right mid cheek
  9-10:  Nose tip / base
  11-12: Left/right inner eye corner
  13-14: Left/right outer eye corner
  15-16: Left/right brow inner
  17-18: Left/right brow outer
  19-20: Forehead crown
"""
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Viseme-to-jaw-shape table  (jaw_open_scale, jaw_width_scale)
# Classes match stage3_audio_encoding:
#   0: silence  1: bilabial(P/B/M)  2: labiodental(F/V)  3: open vowel(A/E/I)
#   4: rounded vowel(O/U/W)
# ──────────────────────────────────────────────────────────────────────────────
VISEME_JAW_SCALE = {
    0: (0.0, 0.0),   # silence — closed
    1: (0.1, 0.4),   # bilabial — slight open, wide
    2: (0.2, 0.3),   # labiodental — moderate open
    3: (0.7, 0.8),   # open vowel — wide open
    4: (0.5, 0.3),   # rounded vowel — moderate open, narrow
}

# Which keypoint indices (in [0..20]) control jaw / mouth open / mouth wide
_JAW_OPEN_KPS   = [0, 1, 2, 3, 4]    # lower-face / chin: push down to open jaw
_JAW_WIDTH_KPS  = [5, 6]             # cheek corners: push out to widen
_EYE_BLINK_KPS  = [11, 12, 13, 14]  # eye corners: push toward center to blink
_BROW_KPS       = [15, 16, 17, 18]  # brows: push up to raise brows
_HEAD_PITCH_KPS = [19, 20]           # forehead: push forward/back for head nod
_HEAD_YAW_KPS   = [9, 10]           # nose: push sideways for head turn

# Amplitude limits (these are keypoint-space units, roughly -0.1 .. 0.1 is subtle)
_MAX_JAW_OPEN  = 0.06   # max delta-Y for jaw-open keypoints
_MAX_JAW_WIDTH = 0.025  # max delta-X for jaw-width keypoints
_MAX_EYE_BLINK = 0.04   # max delta for eye corners during blink
_MAX_BROW      = 0.015  # max brow lift/furrow delta
_MAX_PITCH     = 0.02   # head nod amplitude
_MAX_YAW       = 0.025  # head turn amplitude


def _smooth(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """1-D causal gaussian smooth (applied per-channel)."""
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
        squeezed = True
    else:
        squeezed = False

    k = max(1, kernel_size | 1)  # ensure odd
    sigma = k / 4.0
    half = k // 2
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[1]):
        out[:, c] = np.convolve(arr[:, c].astype(np.float32), kernel, mode="same")

    return out[:, 0] if squeezed else out


def _poisson_blinks(num_frames: float, fps: int, mean_interval_s: float = 4.0,
                    duration_frames: int = 5, rng: np.random.Generator = None) -> np.ndarray:
    """
    Return a float array [0..1] of shape (num_frames,) where blinks are 1 during
    a blink event and 0 otherwise.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mask = np.zeros(int(num_frames), dtype=np.float32)
    t = 0
    mean_gap = int(mean_interval_s * fps)
    while t < int(num_frames):
        gap = int(rng.exponential(mean_gap))
        t += max(1, gap)
        for d in range(duration_frames):
            idx = t + d
            if idx < int(num_frames):
                # triangular blink shape
                progress = d / duration_frames
                mask[idx] = np.sin(np.pi * progress)
    return mask


def _head_motion(num_frames: int, fps: int, rng: np.random.Generator) -> tuple:
    """
    Generate smooth low-frequency head pitch and yaw signals.
    Uses a sum of two low-frequency sinusoids + smoothed noise.
    Returns (pitch, yaw) both shape (num_frames,) in [-1, 1].
    """
    t = np.linspace(0, num_frames / fps, num_frames)

    # Low-frequency "breathing" and "thinking" sinusoids
    freq_slow = rng.uniform(0.05, 0.12)   # very slow head nod
    freq_fast = rng.uniform(0.2, 0.4)     # slight rhythmic sway
    phase_p = rng.uniform(0, 2 * np.pi)
    phase_y = rng.uniform(0, 2 * np.pi)

    pitch = (0.4 * np.sin(2 * np.pi * freq_slow * t + phase_p)
             + 0.2 * np.sin(2 * np.pi * freq_fast * t + phase_p + 1.0))

    yaw = (0.3 * np.sin(2 * np.pi * freq_slow * t + phase_y)
           + 0.2 * np.sin(2 * np.pi * freq_fast * t + phase_y + 0.5))

    # Add a small smooth random drift
    noise_p = rng.normal(0, 0.3, num_frames).astype(np.float32)
    noise_y = rng.normal(0, 0.3, num_frames).astype(np.float32)
    noise_p = _smooth(noise_p, int(fps * 0.8))
    noise_y = _smooth(noise_y, int(fps * 0.8))

    pitch = np.clip(pitch + 0.3 * noise_p, -1, 1).astype(np.float32)
    yaw   = np.clip(yaw   + 0.3 * noise_y, -1, 1).astype(np.float32)
    return pitch, yaw


def audio_to_motion_deltas(
    audio_waveform_path: str,
    speech_features: dict,
    num_frames: int,
    fps: int,
    seed: int = 0,
) -> list:
    """
    Main entry point.  Converts audio into a list of (21, 3) keypoint delta arrays.

    Args:
        audio_waveform_path: Path to the .wav file (used for mel energy computation).
        speech_features: dict with keys:
            'continuous'  np.ndarray [num_frames, D] — Wav2Vec2 hidden states per frame
            'discrete'    np.ndarray [num_frames]    — viseme class 0..4 per frame
        num_frames: Total number of frames to generate.
        fps: Frames per second for this profile.
        seed: Random seed for deterministic head motion.

    Returns:
        List of num_frames np.ndarray of shape (21, 3), dtype float32.
        Each array is a *delta* to be added to the source face keypoints.
    """
    import librosa

    rng = np.random.default_rng(seed)

    # ── 1. Mel energy per frame ────────────────────────────────────────────────
    try:
        y, sr = librosa.load(audio_waveform_path, sr=16000, mono=True)
        hop_length = int(sr / fps)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,
                                             hop_length=hop_length, n_fft=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)      # shape (40, T)
        energy_per_frame = mel_db.mean(axis=0)              # (T,)

        # Normalise to [0, 1]
        e_min, e_max = energy_per_frame.min(), energy_per_frame.max()
        if e_max > e_min:
            energy_norm = (energy_per_frame - e_min) / (e_max - e_min)
        else:
            energy_norm = np.zeros_like(energy_per_frame)

        # Interpolate to num_frames
        from scipy.interpolate import interp1d
        T = len(energy_norm)
        if T != num_frames:
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, num_frames)
            energy_norm = interp1d(x_old, energy_norm, kind='linear',
                                   fill_value='extrapolate')(x_new)
    except Exception as e:
        print(f"[audio_to_motion] mel energy fallback (error: {e})")
        energy_norm = np.zeros(num_frames, dtype=np.float32)

    energy_norm = np.clip(energy_norm.astype(np.float32), 0.0, 1.0)

    # Smooth energy slightly to reduce jitter
    energy_smooth = _smooth(energy_norm, max(3, fps // 5))

    # ── 2. Viseme sequence ──────────────────────────────────────────────────────
    visemes = speech_features.get("discrete")
    if visemes is None or len(visemes) == 0:
        visemes = np.zeros(num_frames, dtype=np.int32)
    elif len(visemes) != num_frames:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(visemes))
        x_new = np.linspace(0, 1, num_frames)
        visemes = interp1d(x_old, visemes, kind='nearest',
                           fill_value='extrapolate')(x_new).astype(np.int32)

    # ── 3. Blink mask ──────────────────────────────────────────────────────────
    blink = _poisson_blinks(num_frames, fps, rng=rng)
    blink = _smooth(blink, 3)

    # ── 4. Head motion ──────────────────────────────────────────────────────────
    pitch, yaw = _head_motion(num_frames, fps, rng)

    # ── 5. Brow from energy peaks ───────────────────────────────────────────────
    # Brows rise slightly during high-energy / stressed syllables
    brow_signal = _smooth(energy_smooth * 0.5, max(5, fps // 3)).astype(np.float32)

    # ── 6. Build per-frame delta arrays ─────────────────────────────────────────
    deltas = []
    for i in range(num_frames):
        delta = np.zeros((21, 3), dtype=np.float32)

        v_class = int(np.clip(visemes[i], 0, 4))
        jaw_open_s, jaw_width_s = VISEME_JAW_SCALE[v_class]

        # Combine: viseme-shape * audio-energy drives final jaw openness
        e = float(energy_smooth[i])
        jaw_open_amt  = jaw_open_s  * e * _MAX_JAW_OPEN
        jaw_width_amt = jaw_width_s * e * _MAX_JAW_WIDTH

        # Jaw-open: push lower-face keypoints down (+Y)
        for kp in _JAW_OPEN_KPS:
            delta[kp, 1] += jaw_open_amt

        # Jaw-width: push cheek keypoints outward (±X)
        delta[5, 0] -= jaw_width_amt  # left cheek → left
        delta[6, 0] += jaw_width_amt  # right cheek → right

        # Eye blink: push eye-corner keypoints inward/down
        blink_amt = float(blink[i]) * _MAX_EYE_BLINK
        for kp in _EYE_BLINK_KPS:
            delta[kp, 1] += blink_amt   # close down

        # Brow raise
        brow_amt = float(brow_signal[i]) * _MAX_BROW
        for kp in _BROW_KPS:
            delta[kp, 1] -= brow_amt   # raise = move up (-Y)

        # Head pitch (nod): forehead kps move in Z, chin kps in opposite Z
        pitch_amt = float(pitch[i]) * _MAX_PITCH
        for kp in _HEAD_PITCH_KPS:
            delta[kp, 2] += pitch_amt
        for kp in _JAW_OPEN_KPS[:2]:
            delta[kp, 2] -= pitch_amt * 0.3

        # Head yaw (turn): nose kps move in X
        yaw_amt = float(yaw[i]) * _MAX_YAW
        for kp in _HEAD_YAW_KPS:
            delta[kp, 0] += yaw_amt

        deltas.append(delta)

    print(f"[audio_to_motion] Generated {len(deltas)} keypoint delta frames "
          f"(fps={fps}, energy_range=[{energy_norm.min():.2f}, {energy_norm.max():.2f}])",
          flush=True)

    return deltas
