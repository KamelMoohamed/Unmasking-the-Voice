import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def download_file(url: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    file_name = url.split("/")[-1]
    output_path = os.path.join(output_dir, file_name)

    if not os.path.exists(output_path):
        print(f"Downloading {file_name}...")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", output_path, url], check=True
        )
    else:
        print(f"{file_name} already exists, skipping download.")

    return output_path


def extract_archive(archive_path: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)

    if archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"Unsupported archive format: {archive_path}")

    print(f"Extracted {archive_path} to {extract_to}")


class DatasetDownloader:
    LIBRISPEECH_URLS = [
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "http://www.openslr.org/resources/12/test-clean.tar.gz",
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    ]

    VOXCELEB1_URLS = [
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip",
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip",
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_meta.csv",
    ]

    VOXCELEB2_URLS = [
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_test_aac.zip",
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_meta.csv",
    ]

    def __init__(self):
        self.base_dir = BASE_OUTPUT_DIR

    def download_and_extract_librispeech(self):
        librispeech_dir = self.base_dir / "LibriSpeech"
        for url in self.LIBRISPEECH_URLS:
            archive_path = download_file(url, librispeech_dir)
            extract_archive(archive_path, librispeech_dir)

    def download_and_extract_voxceleb1(self):
        voxceleb1_dir = self.base_dir / "wav"
        for url in self.VOXCELEB1_URLS:
            archive_path = download_file(url, voxceleb1_dir)
            extract_archive(archive_path, voxceleb1_dir)

    def download_and_extract_voxceleb2(self):
        voxceleb2_dir = self.base_dir / "aac"
        for url in self.VOXCELEB2_URLS:
            archive_path = download_file(url, voxceleb2_dir)
            extract_archive(archive_path, voxceleb2_dir)

    def download_dataset(self, dataset_name: str):
        dataset_name = dataset_name.lower()
        if dataset_name == "librispeech":
            self.download_and_extract_librispeech()
        elif dataset_name == "voxceleb1":
            self.download_and_extract_voxceleb1()
        elif dataset_name == "voxceleb2":
            self.download_and_extract_voxceleb2()
        else:
            print(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    downloader = DatasetDownloader()
    downloader.download_dataset(dataset_name)