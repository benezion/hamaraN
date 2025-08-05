import os
import re
import tempfile
import numpy as np
import soundfile as sf
import sounddevice as sd
import time
from google.cloud import texttospeech
from gtts import gTTS

class TTSProcessor:
    def __init__(self, engine='gtts', debug=False):
        self.engine = engine
        self.last_audio = None
        self.last_audio_rate = None
        self.last_audio_path = None
        self.debug = debug

    def _debug(self, msg):
        if getattr(self, 'debug', False):
            print(f"[B_DEBUG] {msg}")

    def text_to_speech(self, text, gender='male', voice_id=None, output_path=None):
        if self.engine == 'google':
            self._debug(f"Using Google Cloud TTS")
            client = texttospeech.TextToSpeechClient()

            # Prepare SSML text
            if not text.strip().startswith('<speak>'):
                ssml_text = f'<speak>{text}</speak>'
            else:
                ssml_text = text

            # Configure TTS request
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
            voice_name = voice_id if voice_id else ('he-IL-Wavenet-B' if gender == 'male' else 'he-IL-Wavenet-A')
            voice = texttospeech.VoiceSelectionParams(
                language_code="he-IL",
                name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.90
            )

            # Get audio content
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Save to file
            if output_path is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    f.write(response.audio_content)
                    audio_path = f.name
            else:
                with open(output_path, 'wb') as f:
                    f.write(response.audio_content)
                audio_path = output_path

            self.last_audio_path = audio_path
            self.play_audio(audio_path)
            return audio_path

        else:
            # gTTS (local, free)
            tts = gTTS(text, lang='iw')
            if output_path is None:
                # Use a more efficient temp file handling
                temp_dir = os.path.join(tempfile.gettempdir(), 'tts_cache')
                os.makedirs(temp_dir, exist_ok=True)
                audio_path = os.path.join(temp_dir, f'tts_{hash(text)}.mp3')

                # Only generate if not cached
                if not os.path.exists(audio_path):
                    tts.save(audio_path)
            else:
                tts.save(output_path)
                audio_path = output_path

            self.last_audio_path = audio_path
            self.play_audio(audio_path)
            return audio_path

    def play_audio_async(self, audio_path=None):
        """Play audio asynchronously without blocking"""
        if audio_path is None:
            audio_path = self.last_audio_path
        if audio_path is None:
            raise Exception('No audio file to play.')

        if os.name == 'nt':  # Windows
            import subprocess
            # Use start command which is non-blocking
            subprocess.Popen(['start', '', os.path.abspath(audio_path)], shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os.name == 'posix':  # Linux/Mac
            pass
            # import subprocess
            # subprocess.Popen(['xdg-open', audio_path],
                        #    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print(f"Please play the audio file manually: {audio_path}")

    def play_audio(self, audio_path=None):
        """Legacy synchronous audio playback (kept for compatibility)"""
        return self.play_audio_async(audio_path)

    def play_audio_array(self, audio_array=None, samplerate=None):
        """Play audio array data (kept for compatibility)"""
        if audio_array is None:
            audio_array = self.last_audio
        if samplerate is None:
            samplerate = self.last_audio_rate
        if audio_array is None or samplerate is None:
            raise Exception('No audio data to play.')
        sd.play(audio_array, samplerate)
        sd.wait()
