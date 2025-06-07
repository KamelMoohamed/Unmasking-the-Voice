import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import librosa
import soundfile as sf

from cloning_models.fishspeech import FishSpeechCloner
from cloning_models.openvoice import OpenVoiceCloner
from dataloaders.dataloader import DataLoader
from other_environments.over_the_air_simulation import AirEnvironmentSimulator
from other_environments.over_the_line_simulation import EnvironmentSimulator
from tasks.csi import ClosedSetIdentification
from tasks.osi import OpenSetIdentification
from tasks.speaker_verification import SpeakerVerification


@dataclass
class AttackConfig:
    """Configuration for the attack framework"""
    dataset: str
    cloner: Literal["fishspeech", "openvoice"]
    auth_model: Literal["azure", "deep_speaker", "xvector"]
    task: Literal["verification", "csi", "osi"]
    environment: Optional[Literal["air", "line", None]] = None
    threshold: float = 0.5
    num_enrollment_files: int = 3
    num_test_files: int = 2
    num_attack_files: int = 1

class AttackFramework:
    def __init__(self, config: AttackConfig, **kwargs):
        """
        Initialize the attack framework.
        
        Args:
            config: Attack configuration
            **kwargs: Additional arguments for specific components:
                - For fishspeech: api_key
                - For openvoice: converter_checkpoint_dir
                - For azure: subscription_key, region
                - For deep_speaker: model_path
                - For xvector: model_path
        """
        self.config = config
        
        # Initialize data loader
        self.data_loader = DataLoader(dataset=config.dataset)
        
        # Initialize voice cloner
        if config.cloner == "fishspeech":
            self.cloner = FishSpeechCloner(
                api_key=kwargs.get("api_key"),
                output_dir="cloned_audio"
            )
        else:  # openvoice
            self.cloner = OpenVoiceCloner(
                converter_checkpoint_dir=kwargs.get("converter_checkpoint_dir"),
                device=kwargs.get("device")
            )
        
        # Initialize authentication system
        if config.task == "verification":
            self.auth_system = SpeakerVerification(
                backend=config.auth_model,
                **kwargs
            )
        elif config.task == "csi":
            self.auth_system = ClosedSetIdentification(
                backend=config.auth_model,
                threshold=config.threshold,
                **kwargs
            )
        else:  # osi
            self.auth_system = OpenSetIdentification(
                backend=config.auth_model,
                threshold=config.threshold,
                **kwargs
            )
        
        # Initialize environment simulator if needed
        self.env_simulator = None
        if config.environment == "air":
            self.env_simulator = AirEnvironmentSimulator()
        elif config.environment == "line":
            self.env_simulator = EnvironmentSimulator()

    def _prepare_audio(self, audio_path: str) -> str:
        """
        Apply environment simulation if configured.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            str: Path to the processed audio file
        """
        if self.env_simulator is None:
            return audio_path
            
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Apply environment simulation
        if isinstance(self.env_simulator, AirEnvironmentSimulator):
            processed_audio = self.env_simulator.simulate(audio, sr)
        else:  # EnvironmentSimulator
            processed_audio, _ = self.env_simulator.simulate_environment(
                audio=audio,
                environment="phone"  # Default to phone environment
            )
        
        # Save processed audio
        output_path = f"processed_{os.path.basename(audio_path)}"
        sf.write(output_path, processed_audio, sr)
        
        return output_path

    def run_attack(self, target_speaker_id: str, attack_text: str) -> Dict:
        """
        Run the attack process.
        
        Args:
            target_speaker_id: ID of the target speaker
            attack_text: Text to generate for the attack
            
        Returns:
            Dict: Results of the attack including:
                - enrollment_success: Whether enrollment was successful
                - real_test_results: Results of testing with real audio
                - attack_results: Results of the attack
        """
        # Get audio files for the target speaker
        files = self.data_loader.get_files()
        if target_speaker_id not in files:
            raise ValueError(f"Speaker {target_speaker_id} not found in dataset")
            
        speaker_files = files[target_speaker_id]
        if len(speaker_files) < self.config.num_enrollment_files + self.config.num_test_files:
            raise ValueError(f"Not enough files for speaker {target_speaker_id}")
        
        # Split files into enrollment and test sets
        enrollment_files = speaker_files[:self.config.num_enrollment_files]
        test_files = speaker_files[self.config.num_enrollment_files:
                                 self.config.num_enrollment_files + self.config.num_test_files]
        
        # Enroll the speaker
        enrollment_success = self.auth_system.enroll_speaker(
            speaker_id=target_speaker_id,
            wav_files=enrollment_files
        )
        
        results = {
            "enrollment_success": enrollment_success,
            "real_test_results": [],
            "attack_results": []
        }
        
        # Test with real audio
        for test_file in test_files:
            processed_file = self._prepare_audio(test_file)
            if self.config.task == "verification":
                is_verified, score = self.auth_system.verify(processed_file)
                results["real_test_results"].append({
                    "file": test_file,
                    "verified": is_verified,
                    "score": score
                })
            else:  # csi or osi
                speaker_id, score = self.auth_system.identify(processed_file)
                results["real_test_results"].append({
                    "file": test_file,
                    "identified_speaker": speaker_id,
                    "score": score
                })
        
        # Generate and test adversarial audio
        for _ in range(self.config.num_attack_files):
            # Create voice model
            if isinstance(self.cloner, FishSpeechCloner):
                model_id = self.cloner.create_model(
                    voice_paths=enrollment_files,
                    model_name=f"attack_{target_speaker_id}"
                )
                # Generate attack audio
                attack_file = self.cloner.generate_audio(
                    text=attack_text,
                    model_id=model_id,
                    output_filename=f"attack_{target_speaker_id}.wav"
                )
            else:  # OpenVoiceCloner
                # Use first enrollment file as reference
                attack_file = self.cloner.generate_audio(
                    text=attack_text,
                    reference_audio_path=enrollment_files[0],
                    output_path=f"attack_{target_speaker_id}.wav"
                )
            
            # Process attack audio
            processed_attack = self._prepare_audio(attack_file)
            
            # Test attack
            if self.config.task == "verification":
                is_verified, score = self.auth_system.verify(processed_attack)
                results["attack_results"].append({
                    "file": attack_file,
                    "verified": is_verified,
                    "score": score
                })
            else:  # csi or osi
                speaker_id, score = self.auth_system.identify(processed_attack)
                results["attack_results"].append({
                    "file": attack_file,
                    "identified_speaker": speaker_id,
                    "score": score
                })
        
        return results

def main():
    # Example usage
    config = AttackConfig(
        dataset="VoxCeleb1",
        cloner="openvoice",
        auth_model="deep_speaker",
        task="verification",
        environment="air",
        threshold=0.5,
        num_enrollment_files=3,
        num_test_files=2,
        num_attack_files=1
    )
    
    framework = AttackFramework(
        config,
        model_path="path/to/deep_speaker/model.h5",
        converter_checkpoint_dir="path/to/openvoice/checkpoints"
    )
    
    results = framework.run_attack(
        target_speaker_id="id10001",
        attack_text="Hello, this is a test message."
    )
    
    print("Attack Results:")
    print(f"Enrollment successful: {results['enrollment_success']}")
    print("\nReal Test Results:")
    for result in results["real_test_results"]:
        print(f"File: {result['file']}")
        print(f"Verified: {result['verified']}")
        print(f"Score: {result['score']}")
    print("\nAttack Results:")
    for result in results["attack_results"]:
        print(f"File: {result['file']}")
        print(f"Verified: {result['verified']}")
        print(f"Score: {result['score']}")

if __name__ == "__main__":
    main()
