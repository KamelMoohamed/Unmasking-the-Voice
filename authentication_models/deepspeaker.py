from typing import List, Tuple, Union

import numpy as np
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import NUM_FRAMES, SAMPLE_RATE
from deep_speaker.conv_models import DeepSpeakerModel

from utils import cosine_similarity


class DeepSpeakerVerification:
    def __init__(self, model_path: str, threshold: float = 0.5):
        """
        Initialize the DeepSpeaker verification system.
        
        Args:
            model_path: Path to the pretrained model weights (.h5 file)
            threshold: Similarity threshold for verification (default: 0.5)
        """
        self.model = DeepSpeakerModel()
        self.model.m.load_weights(model_path, by_name=True)
        self.threshold = threshold
        self.enrollment_embedding = None

    def _get_embedding(self, file_path: str) -> np.ndarray:
        """
        Get the embedding for a single audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            np.ndarray: The embedding vector for the audio file
        """
        try:
            mfcc = sample_from_mfcc(read_mfcc(file_path, SAMPLE_RATE), NUM_FRAMES)
            embedding = self.model.m.predict(np.expand_dims(mfcc, axis=0), verbose=0)
            return embedding
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None

    def enroll(self, wav_files: Union[str, List[str]]) -> bool:
        """
        Enroll a speaker using one or multiple voice samples.
        
        Args:
            wav_files: Either a single WAV file path or a list of WAV file paths
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        # Convert single file to list for uniform processing
        if isinstance(wav_files, str):
            wav_files = [wav_files]

        if not wav_files:
            print("No WAV files provided for enrollment")
            return False

        # Get embeddings for all files
        embeddings = []
        successful_enrollments = 0
        
        for file in wav_files:
            embedding = self._get_embedding(file)
            if embedding is not None:
                embeddings.append(embedding)
                successful_enrollments += 1

        if successful_enrollments > 0:
            # Calculate mean embedding
            self.enrollment_embedding = np.mean(embeddings, axis=0)
            print(f"Successfully enrolled {successful_enrollments} out of {len(wav_files)} files")
            return True
        else:
            print("Failed to enroll any files")
            return False

    def verify(self, wav_file_path: str) -> Tuple[bool, float]:
        """
        Verify a speaker using their voice sample.
        
        Args:
            wav_file_path: Path to the WAV file containing the voice sample to verify
            
        Returns:
            Tuple[bool, float]: (verification result, similarity score)
        """
        if self.enrollment_embedding is None:
            print("No enrollment data available. Please enroll first.")
            return False, 0.0

        # Get embedding for the test file
        test_embedding = self._get_embedding(wav_file_path)
        if test_embedding is None:
            return False, 0.0

        # Calculate similarity
        similarity = cosine_similarity(test_embedding, self.enrollment_embedding)
        is_verified = similarity > self.threshold

        print(f"Verification result: {'Verified' if is_verified else 'Not verified'} (similarity: {similarity:.3f})")
        return is_verified, float(similarity)