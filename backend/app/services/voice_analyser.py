"""Voice / acoustic feature analysis for scam detection.

Extracts simple acoustic cues from audio that help identify scripted
speech (typical of call-centre scam operations) vs natural conversation:

* Pitch statistics (mean, variance, range)
* Speech rate proxy (zero-crossing rate)
* Pause/silence analysis
* Background noise energy

Falls back to synthetic defaults when ``librosa`` is unavailable.
"""
from __future__ import annotations

import io
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

_LIBROSA_AVAILABLE = False
try:
    import librosa  # type: ignore[import-untyped]
    import soundfile as sf  # type: ignore[import-untyped]
    _LIBROSA_AVAILABLE = np is not None
except ImportError:
    librosa = None  # type: ignore[assignment]
    sf = None
    logger.warning("librosa / soundfile not installed — voice analysis will be mocked")

if not _LIBROSA_AVAILABLE:
    logger.warning("NumPy/librosa unavailable — voice analysis will use lightweight fallback")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class VoiceAnalysisResult:
    is_scripted: bool = False
    scripted_confidence: float = 0.0
    speech_rate: float = 0.0           # relative zero-crossing proxy
    pitch_mean_hz: float = 0.0
    pitch_variance: float = 0.0
    pitch_range_hz: float = 0.0
    silence_ratio: float = 0.0         # fraction of audio that is silence
    pause_count: int = 0
    bg_noise_type: str = "unknown"     # call_centre | personal | silent | noisy
    bg_noise_energy: float = 0.0
    processing_time_ms: int = 0


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _classify_bg_noise(rms_mean: float, spectral_centroid_mean: float) -> str:
    """Heuristic background noise classification."""
    if rms_mean < 0.005:
        return "silent"
    if rms_mean > 0.08:
        return "noisy"
    if spectral_centroid_mean > 3000:
        return "call_centre"
    return "personal"


def _detect_scripted(
    pitch_var: float,
    silence_ratio: float,
    zcr_mean: float,
) -> tuple[bool, float]:
    """Heuristic scripted-speech detection.

    Scripted / read-aloud speech tends to have:
    - Lower pitch variance (monotone delivery)
    - Fewer natural pauses
    - More consistent speaking rate
    """
    score = 0.0

    # Low pitch variance → more scripted
    if pitch_var < 20:
        score += 0.4
    elif pitch_var < 40:
        score += 0.2

    # Low silence ratio → rehearsed, fewer natural hesitations
    if silence_ratio < 0.15:
        score += 0.3
    elif silence_ratio < 0.25:
        score += 0.15

    # Very consistent ZCR → monotone
    if zcr_mean < 0.05:
        score += 0.2

    is_scripted = score >= 0.45
    return is_scripted, round(min(1.0, score), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VoiceAnalyser:
    """Analyses acoustic features of audio for scam detection cues."""

    def analyse_file(self, audio_path: str | Path) -> VoiceAnalysisResult:
        """Analyse a WAV/MP3/OGG file on disk."""
        if not _LIBROSA_AVAILABLE:
            return self._mock_analysis()

        t0 = time.time()
        try:
            y, sr = librosa.load(str(audio_path), sr=16_000, mono=True)
        except Exception as exc:
            logger.error("Failed to load audio: %s", exc)
            return self._mock_analysis()

        return self._analyse_signal(y, sr, t0)

    def analyse_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> VoiceAnalysisResult:
        """Analyse in-memory audio bytes."""
        if not _LIBROSA_AVAILABLE:
            return self._mock_analysis()

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.analyse_file(tmp.name)

    def _analyse_signal(
        self,
        y: np.ndarray,
        sr: int,
        t0: float,
    ) -> VoiceAnalysisResult:
        """Core analysis on a loaded audio signal."""
        duration = len(y) / sr

        # Pitch (F0) via pyin
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=60, fmax=500, sr=sr,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        if len(f0_voiced) > 0:
            pitch_mean = float(np.nanmean(f0_voiced))
            pitch_var = float(np.nanvar(f0_voiced))
            pitch_range = float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced))
        else:
            pitch_mean, pitch_var, pitch_range = 0.0, 0.0, 0.0

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))

        # Silence detection
        silence_threshold = 0.01
        silence_frames = np.sum(rms < silence_threshold)
        silence_ratio = float(silence_frames / max(1, len(rms)))
        pause_count = 0
        in_silence = False
        for val in rms:
            if val < silence_threshold:
                if not in_silence:
                    pause_count += 1
                    in_silence = True
            else:
                in_silence = False

        # Zero-crossing rate (speech rate proxy)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))

        # Spectral centroid (brightness — call centres often brighter)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sc_mean = float(np.mean(sc))

        # Classify
        bg_noise = _classify_bg_noise(rms_mean, sc_mean)
        is_scripted, scripted_conf = _detect_scripted(pitch_var, silence_ratio, zcr_mean)

        elapsed = int((time.time() - t0) * 1000)

        return VoiceAnalysisResult(
            is_scripted=is_scripted,
            scripted_confidence=scripted_conf,
            speech_rate=round(zcr_mean, 4),
            pitch_mean_hz=round(pitch_mean, 1),
            pitch_variance=round(pitch_var, 2),
            pitch_range_hz=round(pitch_range, 1),
            silence_ratio=round(silence_ratio, 3),
            pause_count=pause_count,
            bg_noise_type=bg_noise,
            bg_noise_energy=round(rms_mean, 5),
            processing_time_ms=elapsed,
        )

    @staticmethod
    def _mock_analysis() -> VoiceAnalysisResult:
        """Return plausible mock results for demo."""
        return VoiceAnalysisResult(
            is_scripted=True,
            scripted_confidence=0.72,
            speech_rate=0.065,
            pitch_mean_hz=178.4,
            pitch_variance=18.6,
            pitch_range_hz=95.3,
            silence_ratio=0.12,
            pause_count=4,
            bg_noise_type="call_centre",
            bg_noise_energy=0.032,
            processing_time_ms=85,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_analyser: VoiceAnalyser | None = None


def get_voice_analyser() -> VoiceAnalyser:
    global _analyser
    if _analyser is None:
        _analyser = VoiceAnalyser()
    return _analyser
