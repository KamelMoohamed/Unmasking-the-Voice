import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import scipy.signal
import soundfile as sf


@dataclass
class EnvironmentConfig:
    """Configuration for environment simulation"""
    name: str
    sample_rate: int = 16000
    down_sample_rate: Optional[int] = None
    bandpass_freqs: Optional[Tuple[float, float]] = None
    filter_order: int = 10

class EnvironmentSimulator:
    def __init__(self, output_dir: str = "simulated_environments"):
        """
        Initialize the environment simulator.
        
        Args:
            output_dir: Directory to save simulated audio files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Predefined environment configurations
        self.environments = {
            "phone": EnvironmentConfig(
                name="phone",
                sample_rate=16000,
                down_sample_rate=8000,
                bandpass_freqs=(300, 3400),
                filter_order=10
            ),
            "voip": EnvironmentConfig(
                name="voip",
                sample_rate=16000,
                down_sample_rate=8000,
                bandpass_freqs=(300, 3400),
                filter_order=10
            ),
            # Add more environments as needed
        }

    def _load_audio(self, path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            path: Path to audio file
            sr: Target sample rate
            
        Returns:
            Tuple[np.ndarray, int]: (audio data, sample rate)
        """
        audio, _ = librosa.load(path, sr=sr)
        return audio, sr

    def _save_audio(self, path: str, audio: np.ndarray, sr: int) -> None:
        """
        Save audio file.
        
        Args:
            path: Path to save audio file
            audio: Audio data
            sr: Sample rate
        """
        sf.write(path, audio, sr)

    def _apply_bandpass_filter(self, 
                             audio: np.ndarray, 
                             sr: int, 
                             freqs: Tuple[float, float], 
                             order: int) -> np.ndarray:
        """
        Apply bandpass filter to audio.
        
        Args:
            audio: Audio data
            sr: Sample rate
            freqs: (low_freq, high_freq) for bandpass filter
            order: Filter order
            
        Returns:
            np.ndarray: Filtered audio
        """
        sos = scipy.signal.butter(order, freqs, btype='band', fs=sr, output='sos')
        return scipy.signal.sosfilt(sos, audio)

    def simulate_environment(self, 
                           audio: Union[str, np.ndarray],
                           environment: Union[str, EnvironmentConfig],
                           output_filename: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        Simulate audio in a specific environment.
        
        Args:
            audio: Audio data or path to audio file
            environment: Environment name or configuration
            output_filename: Optional output filename
            
        Returns:
            Tuple[np.ndarray, str]: (processed audio, output path)
        """
        # Get environment configuration
        if isinstance(environment, str):
            if environment not in self.environments:
                raise ValueError(f"Unknown environment: {environment}")
            config = self.environments[environment]
        else:
            config = environment

        # Load audio if path is provided
        if isinstance(audio, str):
            audio_data, sr = self._load_audio(audio, config.sample_rate)
        else:
            audio_data = audio
            sr = config.sample_rate

        # Apply downsampling if configured
        if config.down_sample_rate:
            audio_down = librosa.resample(
                audio_data, 
                orig_sr=sr, 
                target_sr=config.down_sample_rate
            )
            audio_up = librosa.resample(
                audio_down, 
                orig_sr=config.down_sample_rate, 
                target_sr=sr
            )
        else:
            audio_up = audio_data

        # Apply bandpass filter if configured
        if config.bandpass_freqs:
            processed_audio = self._apply_bandpass_filter(
                audio_up,
                sr,
                config.bandpass_freqs,
                config.filter_order
            )
        else:
            processed_audio = audio_up

        # Save to file if output filename is provided
        if output_filename:
            output_path = os.path.join(self.output_dir, output_filename)
            self._save_audio(output_path, processed_audio, sr)
        else:
            output_path = None

        return processed_audio, output_path

    def add_environment(self, config: EnvironmentConfig) -> None:
        """
        Add a new environment configuration.
        
        Args:
            config: Environment configuration
        """
        self.environments[config.name] = config
