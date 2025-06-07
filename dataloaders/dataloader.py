import glob
from pathlib import Path
from typing import Dict, List


class DataLoader:
    VALID_DATASETS = ["VoxCeleb1", "VoxCeleb2", "LibriSpeech"]
    
    def __init__(
        self,
        dataset: str,
        gender_metadata: Path = None,
        mode: str = "intergender",
    ):
        """
        Initializes the DataLoader.
        :param dataset: One of "VoxCeleb1", "VoxCeleb2", or "LibriSpeech".
        :param gender_metadata: Path to gender metadata CSV/TXT file (required for intergender mode).
        :param mode: "intragender" (same-gender attackers) or "intergender" (different-gender attackers).
        """
        if dataset not in self.VALID_DATASETS:
            raise ValueError(f"Dataset must be one of {self.VALID_DATASETS}, got {dataset}")

        self.dataset = dataset
        dataset_folder = (
            "LibriSpeech/LibriSpeech"
            if dataset == "LibriSpeech"
            else "VoxCeleb1/wav" if dataset == "VoxCeleb1" else "VoxCeleb2/aac"
        )

        self.dataset_folder = dataset_folder
        self.base_path = (
            Path(__file__).resolve().parent.parent.parent / "data" / dataset_folder
        )
        self.mode = mode
        self.gender_metadata = gender_metadata
        self._files = self._collect_files()

    def get_files(self) -> Dict[str, List[str]]:
        """
        Get the dictionary of user IDs and their corresponding audio file paths.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping user IDs to their audio file paths
        """
        return self._files

    def _collect_files(self) -> Dict[str, List[str]]:
        """
        Collects files for each user in the dataset.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping user IDs to their audio file paths
        """
        files = {}
        if not self.base_path.exists():
            return files

        ext = (
            "m4a"
            if self.dataset_folder == "aac"
            else "wav" if self.dataset_folder == "wav" else "flac"
        )

        if self.dataset == "LibriSpeech":
            for subset in self.base_path.iterdir():
                if subset.is_dir():
                    for user_id in subset.iterdir():
                        if user_id.is_dir():
                            uid = user_id.name
                            if uid not in files:
                                files[uid] = []
                            files[uid].extend(
                                glob.glob(
                                    str(user_id / "**" / f"*.{ext}"), recursive=True
                                )
                            )
        else:
            for user_id in self.base_path.iterdir():
                if user_id.is_dir():
                    uid = user_id.name
                    files[uid] = glob.glob(
                        str(user_id / "**" / f"*.{ext}"), recursive=True
                    )

        return files

