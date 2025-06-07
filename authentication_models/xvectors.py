from typing import List, Optional, Tuple, Union

import numpy as np
import torchaudio
from speechbrain.pretrained import SpeakerRecognition

from utils import cosine_similarity


class XVectorVerification:
    def __init__(self, model_path: str = "speechbrain/spkrec-xvect-voxceleb", threshold: float = 0.5):
        """
        Initialize the xVector verification system.
        
        Args:
            model_path: Path to the pretrained model or model identifier
            threshold: Similarity threshold for verification (default: 0.5)
        """
        self.model = SpeakerRecognition.from_hparams(
            source=model_path,
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        )
        self.threshold = threshold
        self.enrollment_embedding = None

    def _get_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a single audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Optional[np.ndarray]: The embedding vector for the audio file, or None if processing fails
        """
        try:
            signal, fs = torchaudio.load(file_path)
            embedding = self.model.encode_batch(signal).squeeze(0).detach().cpu().numpy()
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