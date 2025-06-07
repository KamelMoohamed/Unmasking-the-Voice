from typing import List, Optional, Tuple

import numpy as np
import scipy.signal


class AirEnvironmentSimulator:
    def __init__(self, 
                 reverb_delays: Optional[List[Tuple[float, float]]] = None,
                 noise_level: float = 0.005,
                 lowpass_freq: float = 4000,
                 filter_order: int = 4):
        """
        Initialize the air environment simulator.
        
        Args:
            reverb_delays: List of (delay, amplitude) pairs for reverb simulation
            noise_level: Standard deviation of noise to add
            lowpass_freq: Cutoff frequency for lowpass filter
            filter_order: Order of the lowpass filter
        """
        self.reverb_delays = reverb_delays or [
            (0.03, 0.6),  # 30ms delay, 0.6 amplitude
            (0.06, 0.3),  # 60ms delay, 0.3 amplitude
            (0.1, 0.1)    # 100ms delay, 0.1 amplitude
        ]
        self.noise_level = noise_level
        self.lowpass_freq = lowpass_freq
        self.filter_order = filter_order

    def simulate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simulate audio in an air environment by adding reverb, noise, and filtering.
        
        Args:
            audio: Input audio data
            sr: Sample rate of the audio
            
        Returns:
            np.ndarray: Processed audio with environmental effects
        """
        # Add synthetic room reverb
        room_ir = np.zeros(sr)
        room_ir[0] = 1.0  # Direct sound
        
        # Add delayed reflections
        for delay, amplitude in self.reverb_delays:
            delay_samples = int(delay * sr)
            room_ir[delay_samples] = amplitude
            
        reverb = np.convolve(audio, room_ir)[:len(audio)]

        # Add background noise
        noise = np.random.normal(0, self.noise_level, len(audio))
        noisy = reverb + noise

        # Apply lowpass filter for mic frequency drop-off
        sos = scipy.signal.butter(
            self.filter_order, 
            self.lowpass_freq, 
            btype='low', 
            fs=sr, 
            output='sos'
        )
        degraded = scipy.signal.sosfilt(sos, noisy)

        return degraded

