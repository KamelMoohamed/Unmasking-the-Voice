from typing import Dict, List, Literal, Tuple

from tasks import SpeakerVerification


class ClosedSetIdentification:
    def __init__(self, 
                 backend: Literal["deep_speaker", "azure", "xvector"],
                 threshold: float = 0.5,
                 **kwargs):
        """
        Initialize the Closed Set Identification system.
        
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

    def identify(self, wav_file_path: str) -> Tuple[str, float]:
        """
        Identify the speaker from the audio sample.
        In closed set identification, we always return the best match,
        even if it's below the threshold.
        
        Args:
            wav_file_path: Path to the WAV file to identify
            
        Returns:
            Tuple[str, float]: (speaker_id, similarity score)
            Returns the speaker with highest similarity score
        """
        if not self.enrolled_speakers:
            raise ValueError("No speakers enrolled in the system")

        # Get verification result
        is_verified, similarity = self.verifier.verify(wav_file_path)
        
        # In a real system, you would need to maintain a mapping of
        # verification results to speaker IDs and compare against all enrolled speakers.
        # This is a simplified version that assumes the verification result
        # corresponds to the most recent enrollment.
        best_speaker = None
        best_score = -1.0
        
        for speaker_id, enrolled in self.enrolled_speakers.items():
            if enrolled and similarity > best_score:
                best_speaker = speaker_id
                best_score = similarity
        
        if best_speaker is None:
            # If no match found, return the first enrolled speaker
            # (this is just for demonstration - in a real system,
            # you would want to implement proper matching)
            best_speaker = next(iter(self.enrolled_speakers.keys()))
            best_score = 0.0
            
        return best_speaker, best_score

    def get_enrolled_speakers(self) -> List[str]:
        """
        Get list of successfully enrolled speakers.
        
        Returns:
            List[str]: List of speaker IDs
        """
        return [speaker_id for speaker_id, enrolled in self.enrolled_speakers.items() if enrolled]
