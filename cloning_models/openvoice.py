import os
import shutil
from typing import Optional

import torch
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter


class OpenVoiceCloner:
    def __init__(self, 
                 converter_checkpoint_dir: str = 'checkpoints_v2/converter',
                 device: Optional[str] = None,
                 speed: float = 1.0):
        """
        Initialize the OpenVoice voice cloner.
        
        Args:
            converter_checkpoint_dir: Path to the converter checkpoint directory
            device: Device to use for inference (default: cuda if available, else cpu)
            speed: Speech speed (default: 1.0)
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.speed = speed
        self.converter_checkpoint_dir = converter_checkpoint_dir
        
        # Initialize the tone color converter
        self.tone_color_converter = ToneColorConverter(
            f'{converter_checkpoint_dir}/config.json',
            device=self.device
        )
        self.tone_color_converter.load_ckpt(f'{converter_checkpoint_dir}/checkpoint.pth')
        
        # Initialize TTS model
        self.tts_model = TTS(language="EN_NEWEST", device=self.device)
        self.speaker_ids = self.tts_model.hps.data.spk2id
        
        # Create temporary directories if they don't exist
        self.temp_dir = "/kaggle/working/DUMMY_TONE"
        self.src_dir = "/kaggle/working/DUMMY_SRC"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.src_dir, exist_ok=True)

    def create_model(self, reference_audio_path: str, model_name: str) -> str:
        """
        Create a voice model from a reference audio file.
        
        Args:
            reference_audio_path: Path to the reference audio file
            model_name: Name for the model
            
        Returns:
            str: Path to the saved speaker embedding
        """
        # Create a copy of the reference file
        file_name = os.path.basename(reference_audio_path)
        copied_file_path = os.path.join(self.temp_dir, f"copy_{file_name}")
        shutil.copyfile(reference_audio_path, copied_file_path)
        
        # Extract speaker embedding
        target_se, _ = se_extractor.get_se(
            copied_file_path,
            self.tone_color_converter,
            vad=True
        )
        
        # Save the speaker embedding
        output_path = os.path.join(self.temp_dir, f"{model_name}_se.pth")
        torch.save(target_se, output_path)
        
        # Clean up
        os.remove(copied_file_path)
        
        return output_path

    def generate_audio(self, 
                      text: str,
                      reference_audio_path: str,
                      output_path: str,
                      speaker_name: Optional[str] = None) -> str:
        """
        Generate audio using the voice cloner.
        
        Args:
            text: Text to convert to speech
            reference_audio_path: Path to the reference audio file
            output_path: Path to save the generated audio
            speaker_name: Name of the speaker to use (default: use first available speaker)
            
        Returns:
            str: Path to the generated audio file
        """
        # Get speaker embedding
        target_se, _ = se_extractor.get_se(
            reference_audio_path,
            self.tone_color_converter,
            vad=True
        )
        
        # Select speaker
        if speaker_name is None:
            speaker_key = next(iter(self.speaker_ids.keys()))
        else:
            speaker_key = speaker_name
            
        speaker_id = self.speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        # Generate source audio
        src_path = os.path.join(self.src_dir, 'tmp.wav')
        self.tts_model.tts_to_file(text, speaker_id, src_path, speed=self.speed)
        
        # Load source speaker embedding
        source_se = torch.load(
            f'/kaggle/working/OpenVoice/checkpoints_v2/base_speakers/ses/{speaker_key}.pth',
            map_location=self.device
        )
        
        # Convert voice
        encode_message = "@MyShell"
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message=encode_message
        )
        
        # Clean up
        os.remove(src_path)
        
        return output_path