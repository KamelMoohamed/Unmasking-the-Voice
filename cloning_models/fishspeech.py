import os
from typing import List

from fish_audio_sdk import Session, TTSRequest


class FishSpeechCloner:
    def __init__(self, api_key: str, output_dir: str = "/path/to/cloned_with_fish"):
        """
        Initialize the FishSpeech voice cloner.
        
        Args:
            api_key: FishSpeech API key
            output_dir: Directory to save generated audio files
        """
        self.session = Session(api_key)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_model(self, voice_paths: List[str], model_name: str) -> str:
        """
        Create a voice model from reference audio files.
        
        Args:
            voice_paths: List of paths to reference audio files
            model_name: Name for the model
            
        Returns:
            str: Model ID for future use
        """
        voices = []
        for path in voice_paths:
            with open(path, "rb") as f:
                voices.append(f.read())
        
        model = self.session.create_model(
            title=model_name,
            voices=voices,
        )
        return model.id

    def generate_audio(self, 
                      text: str,
                      model_id: str,
                      output_filename: str) -> str:
        """
        Generate audio using the voice cloner.
        
        Args:
            text: Text to convert to speech
            model_id: ID of the voice model to use
            output_filename: Name of the output file (without path)
            
        Returns:
            str: Path to the generated audio file
        """
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, "wb") as f:
            for chunk in self.session.tts(TTSRequest(
                reference_id=model_id,
                text=text
            )):
                f.write(chunk)
                
        return output_path

