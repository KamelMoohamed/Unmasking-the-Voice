from typing import Dict, List, Literal, Optional, Tuple

from tasks import SpeakerVerification


class OpenSetIdentification:
    def __init__(self, 
                 backend: Literal["deep_speaker", "azure", "xvector"],
                 threshold: float = 0.5,
                 **kwargs):
        """
        Initialize the Open Set Identification system.
        
        Args:
            backend: The verification backend to use
            threshold: Similarity threshold for identification
            **kwargs: Additional arguments for the specific backend
        """
        self.verifier = SpeakerVerification(backend, threshold=threshold, **kwargs)
        self.enrolled_speakers: Dict[str, bool] = {}  # speaker_id -> enrollment_status
        self.threshold = threshold

    def enroll_speaker(self, speaker_id: str, wav_files: List[str]) -> bool:
        """
        Enroll a speaker in the system.
        
        Args:
            speaker_id: Unique identifier for the speaker
            wav_files: List of WAV file paths for enrollment
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        success = self.verifier.enroll(wav_files)
        self.enrolled_speakers[speaker_id] = success
        return success

    def identify(self, wav_file_path: str) -> Tuple[Optional[str], float]:
        """
        Identify the speaker from the audio sample.
        
        Args:
            wav_file_path: Path to the WAV file to identify
            
        Returns:
            Tuple[Optional[str], float]: (speaker_id if identified, similarity score)
            Returns (None, score) if no match is found above threshold
        """
        if not self.enrolled_speakers:
            print("No speakers enrolled in the system")
            return None, 0.0

        # Get verification result
        is_verified, similarity = self.verifier.verify(wav_file_path)
        
        if is_verified and similarity > self.threshold:
            # In a real system, you would need to maintain a mapping of
            # verification results to speaker IDs. This is a simplified version.
            # You might want to implement a more sophisticated matching system.
            for speaker_id, enrolled in self.enrolled_speakers.items():
                if enrolled:
                    return speaker_id, similarity
        
        return None, similarity

    def get_enrolled_speakers(self) -> List[str]:
        """
        Get list of successfully enrolled speakers.
        
        Returns:
            List[str]: List of speaker IDs
        """
        return [speaker_id for speaker_id, enrolled in self.enrolled_speakers.items() if enrolled]
