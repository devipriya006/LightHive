
from __future__ import annotations

from collections import deque

import librosa
import numpy as np


class LiveBeatTracker:
    """Estimate beat events and tempo from successive mono audio chunks."""

    def __init__(
        self,
        *,
        sample_rate: int = 22050,
        hop_length: int = 256,
        history_seconds: float = 8.0,
        min_bpm: float = 40.0,
        max_bpm: float = 200.0,
        onset_threshold: float = 1.5,
        refractory_seconds: float = 0.32,
        analysis_seconds: float = 6.0,
    ) -> None:
        if sample_rate <= 0 or hop_length <= 0:
            raise ValueError("sample_rate and hop_length must be positive.")
        if not 0 < min_bpm < max_bpm:
            raise ValueError("Require 0 < min_bpm < max_bpm.")
        if history_seconds <= 0 or analysis_seconds <= 0:
            raise ValueError("History and analysis durations must be positive.")

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.history_seconds = history_seconds
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.onset_threshold = onset_threshold
        self.refractory_seconds = refractory_seconds

        self.buffer = np.zeros(0, dtype=np.float32)
        self.max_buffer_samples = int(sample_rate * analysis_seconds)
        self.audio_samples_seen = 0

        self.last_beat_time: float | None = None
        self.history: deque[tuple[float, float]] = deque()

        # Separate tempo estimate from the event detector.
        self.last_bpm: float | None = None
        self.last_interval: float | None = None

    def update(self, audio_chunk: np.ndarray) -> dict:
        """Process one audio chunk and return a beat/tempo result."""
        y = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)

        if y.size == 0:
            return self._result(False, None, 0.0)

        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        self.audio_samples_seen += y.size
        self.buffer = np.concatenate((self.buffer, y))

        if self.buffer.size > self.max_buffer_samples:
            self.buffer = self.buffer[-self.max_buffer_samples :]

        newest_time = self.audio_samples_seen / self.sample_rate

        # Need enough audio to estimate a meaningful onset pattern.
        if self.buffer.size < int(1.0 * self.sample_rate):
            return self._result(False, None, 0.0)

        if np.max(np.abs(self.buffer)) < 1e-6:
            return self._result(False, None, 0.0)

        onset_env = librosa.onset.onset_strength(
            y=self.buffer,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        if onset_env.size < 3:
            return self._result(False, None, 0.0)

        frame_times = librosa.frames_to_time(
            np.arange(onset_env.size),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        buffer_start = (
            self.audio_samples_seen - self.buffer.size
        ) / self.sample_rate
        absolute_times = frame_times + buffer_start

        # Estimate tempo from the onset envelope over the rolling buffer.
        # This is more robust than treating every onset as a beat.
        try:
            tempo = librosa.feature.tempo(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                aggregate=np.median,
            )
            raw_bpm = float(np.asarray(tempo).reshape(-1)[0])

            if np.isfinite(raw_bpm) and self.min_bpm <= raw_bpm <= self.max_bpm:
                # Smooth successive estimates to reduce sudden fluctuations.
                if self.last_bpm is None:
                    self.last_bpm = raw_bpm
                else:
                    self.last_bpm = 0.75 * self.last_bpm + 0.25 * raw_bpm

                self.last_interval = 60.0 / self.last_bpm

        except (ValueError, FloatingPointError):
            pass

        # Only inspect the newest section for beat events.
        eligible = np.flatnonzero(
            (absolute_times >= newest_time - 0.35)
            & (absolute_times <= newest_time)
        )

        detected_time = None
        strength = 0.0

        if eligible.size:
            baseline = float(np.median(onset_env))
            spread = float(np.median(np.abs(onset_env - baseline)))
            threshold = baseline + self.onset_threshold * max(spread, 1e-6)

            candidates = []
            for i in eligible:
                if i <= 0 or i >= onset_env.size - 1:
                    continue

                if (
                    onset_env[i] >= onset_env[i - 1]
                    and onset_env[i] > onset_env[i + 1]
                    and onset_env[i] > threshold
                ):
                    candidates.append(int(i))

            # Prefer the latest candidate not already emitted.
            for peak in reversed(candidates):
                candidate_time = float(absolute_times[peak])
                candidate_strength = float(onset_env[peak])

                if self.last_beat_time is not None:
                    elapsed = candidate_time - self.last_beat_time

                    if elapsed < self.refractory_seconds:
                        continue

                detected_time = candidate_time
                strength = candidate_strength
                break

        if detected_time is not None:
            self.last_beat_time = detected_time
            self.history.append((detected_time, strength))

        while self.history and newest_time - self.history[0][0] > self.history_seconds:
            self.history.popleft()

        return self._result(
            detected_time is not None,
            detected_time,
            strength,
            bpm=self.last_bpm,
            interval=self.last_interval,
        )

    def _result(
        self,
        detected: bool,
        timestamp: float | None,
        strength: float,
        *,
        bpm: float | None = None,
        interval: float | None = None,
    ) -> dict:
        return {
            "beat_detected": detected,
            "timestamp": round(timestamp, 3) if timestamp is not None else None,
            "onset_strength": round(float(strength), 3),
            "current_bpm": round(bpm, 2) if bpm is not None else None,
            "beat_interval_seconds": (
                round(interval, 3) if interval is not None else None
            ),
        }
