from typing import List, Literal, Tuple, Union

from authentication_models import (AzureSpeakerVerification,
                                   DeepSpeakerVerification,
                                   XVectorVerification)


class SpeakerVerification:
    def __init__(self, 
                 backend: Literal["deep_speaker", "azure", "xvector"],
                 **kwargs):
        """
        Initialize the speaker verification system with the specified backend.
        
        Args:
            backend: The verification backend to use ("deep_speaker", "azure", or "xvector")
            **kwargs: Additional arguments for the specific backend:
                - For deep_speaker: model_path, threshold
                - For azure: subscription_key, region
                - For xvector: model_path, threshold
        """
        self.backend = backend
        
        if backend == "deep_speaker":
            self.verifier = DeepSpeakerVerification(
                model_path=kwargs.get("model_path"),
                threshold=kwargs.get("threshold", 0.5)
            )
        elif backend == "azure":
            self.verifier = AzureSpeakerVerification(
                subscription_key=kwargs.get("subscription_key"),
                region=kwargs.get("region", "eastus")
            )
        elif backend == "xvector":
            self.verifier = XVectorVerification(
                model_path=kwargs.get("model_path"),
                threshold=kwargs.get("threshold", 0.5)
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def enroll(self, wav_files: Union[str, List[str]]) -> bool:
        """
        Enroll a speaker using one or multiple voice samples.
        
        Args:
            wav_files: Either a single WAV file path or a list of WAV file paths
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        return self.verifier.enroll(wav_files)

    def verify(self, wav_file_path: str) -> Tuple[bool, float]:
        """
        Verify a speaker using their voice sample.
        
        Args:
            wav_file_path: Path to the WAV file containing the voice sample to verify
            
        Returns:
            Tuple[bool, float]: (verification result, similarity score)
        """
        return self.verifier.verify(wav_file_path) 