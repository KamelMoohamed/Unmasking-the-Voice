from typing import List, Union

import requests


class AzureSpeakerVerification:
    def __init__(self, subscription_key: str, region: str = 'eastus'):
        self.subscription_key = subscription_key
        self.region = region
        self.base_url = f'https://{region}.api.cognitive.microsoft.com/speaker/verification/v2.0'
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Content-Type': 'application/json'
        }
        self.profile_id = None

    def _create_profile(self) -> str:
        """Create a new verification profile."""
        data = {
            "locale": "en-US"
        }

        response = requests.post(
            f"{self.base_url}/text-independent/profiles",
            headers=self.headers,
            json=data
        )

        if response.status_code == 201:
            profile_id = response.json()['profileId']
            print("Profile created:", profile_id)
            return profile_id
        else:
            print("Failed to create profile:", response.text)
            return None

    def _get_profile_status(self, profile_id: str) -> str:
        """Get the enrollment status of a profile."""
        status_url = f"{self.base_url}/text-independent/profiles/{profile_id}"
        status_response = requests.get(status_url, headers=self.headers)
        if status_response.status_code == 200:
            profile = status_response.json()
            return profile["enrollmentStatus"]
        else:
            print("Failed to get profile status:", status_response.text)
            return None

    def _enroll_single_file(self, wav_file_path: str) -> bool:
        """
        Enroll a single WAV file for the current profile.
        
        Args:
            wav_file_path: Path to the WAV file to enroll
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        enroll_url = f"{self.base_url}/text-independent/profiles/{self.profile_id}/enrollments"

        try:
            with open(wav_file_path, 'rb') as audio:
                audio_data = audio.read()

            enroll_headers = {
                'Ocp-Apim-Subscription-Key': self.subscription_key,
                'Content-Type': 'audio/wav'
            }

            response = requests.post(enroll_url, headers=enroll_headers, data=audio_data)
            if response.status_code == 200:
                print(f"Successfully enrolled file: {wav_file_path}")
                return True
            else:
                print(f"Failed to enroll file {wav_file_path}:", response.text)
                return False
        except Exception as e:
            print(f"Error processing file {wav_file_path}:", str(e))
            return False

    def enroll(self, wav_files: Union[str, List[str]]) -> str:
        """
        Enroll a speaker using multiple voice samples.
        
        Args:
            wav_files: Either a single WAV file path or a list of WAV file paths
            
        Returns:
            str: Enrollment status if successful, None otherwise
        """
        # Convert single file to list for uniform processing
        if isinstance(wav_files, str):
            wav_files = [wav_files]

        if not wav_files:
            print("No WAV files provided for enrollment")
            return None

        # Create profile if it doesn't exist
        if not self.profile_id:
            self.profile_id = self._create_profile()
            if not self.profile_id:
                return None

        # Enroll each file
        successful_enrollments = 0
        for wav_file in wav_files:
            if self._enroll_single_file(wav_file):
                successful_enrollments += 1

        if successful_enrollments > 0:
            print(f"Successfully enrolled {successful_enrollments} out of {len(wav_files)} files")
            return self._get_profile_status(self.profile_id)
        else:
            print("Failed to enroll any files")
            return None

    def verify(self, wav_file_path: str) -> bool:
        """
        Verify a speaker using their voice sample.
        
        Args:
            wav_file_path: Path to the WAV file containing the voice sample to verify
            
        Returns:
            bool: True if verification is successful, False otherwise
        """
        if not self.profile_id:
            print("No profile ID available. Please enroll first.")
            return False

        verify_url = f"{self.base_url}/text-independent/profiles/{self.profile_id}/verify"

        with open(wav_file_path, 'rb') as audio:
            audio_data = audio.read()

        verify_headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Content-Type': 'audio/wav'
        }

        response = requests.post(verify_url, headers=verify_headers, data=audio_data)
        if response.status_code == 200:
            result = response.json()
            print(result)
            return result['recognitionResult']
        else:
            print("Verification failed:", response.text)
            return False