import os
import re
import tempfile
import numpy as np
import soundfile as sf
import sounddevice as sd
import time
import hashlib
from google.cloud import texttospeech
from gtts import gTTS
from pydub import AudioSegment

class TTSProcessor:
    def __init__(self, engine='gtts', debug=False):
        self.engine = engine
        self.last_audio = None
        self.last_audio_rate = None
        self.last_audio_path = None
        self.debug = debug
        self._google_client = None  # Cache Google Cloud client
        self._client_initialized = False  # Track if client is ready

    def _debug(self, msg):
        if getattr(self, 'debug', False):
            print(f"[B_DEBUG] {msg}")

    def _get_cache_path(self, text, engine):
        """Generate cache file path based on text hash"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        temp_dir = os.path.join(tempfile.gettempdir(), 'hamaraN_tts_cache')
        os.makedirs(temp_dir, exist_ok=True)
        return os.path.join(temp_dir, f'{engine}_{text_hash}.mp3')

    def text_to_speech(self, text, gender='male', voice_id=None, output_path=None, play_audio=True):
        start_time = time.time()
        
        # Check cache first for faster response
        cache_path = self._get_cache_path(text, self.engine)
        if os.path.exists(cache_path):
            if output_path is None:
                # No specific output path requested - return cache path
                self._debug(f"🚀 Using cached audio file (saved ~2000ms): {cache_path}")
                self.last_audio_path = cache_path
                if play_audio:
                    play_start = time.time()
                    self.play_audio(cache_path)
                    self._debug(f"Audio play took: {(time.time() - play_start)*1000:.1f}ms")
                self._debug(f"Total cached time: {(time.time() - start_time)*1000:.1f}ms")
                return cache_path
            else:
                # Specific output path requested - copy cache to that location
                import shutil
                shutil.copy2(cache_path, output_path)
                self._debug(f"🚀 Copied cached audio to specified path: {cache_path} → {output_path}")
                self.last_audio_path = output_path
                if play_audio:
                    play_start = time.time()
                    self.play_audio(output_path)
                    self._debug(f"Audio play took: {(time.time() - play_start)*1000:.1f}ms")
                self._debug(f"Total cached copy time: {(time.time() - start_time)*1000:.1f}ms")
                return output_path
        if self.engine == 'google':
            self._debug(f"Using Google Cloud TTS")
            
            # Use cached client or create new one
            if not self._client_initialized:
                client_start = time.time()
                try:
                    self._google_client = texttospeech.TextToSpeechClient()
                    self._client_initialized = True
                    self._debug(f"✅ Google Cloud client initialized in {(time.time() - client_start)*1000:.1f}ms")
                except Exception as e:
                    self._debug(f"❌ Failed to initialize Google Cloud client: {e}")
                    # Fallback to gTTS
                    self.engine = 'gtts'
                    return self.text_to_speech(text, gender, voice_id, output_path)
            else:
                self._debug(f"🚀 Using cached Google Cloud client (saved ~1000ms)")
            
            client = self._google_client

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
            api_start = time.time()
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            self._debug(f"API call took: {(time.time() - api_start)*1000:.1f}ms")

            # Save to file
            if output_path is None:
                # Save to cache
                with open(cache_path, 'wb') as f:
                    f.write(response.audio_content)
                audio_path = cache_path
                self._debug(f"💾 Saved to cache: {cache_path}")
            else:
                with open(output_path, 'wb') as f:
                    f.write(response.audio_content)
                audio_path = output_path

            self.last_audio_path = audio_path
            self._debug(f"Total Google TTS time: {(time.time() - start_time)*1000:.1f}ms")
            if play_audio:
                play_start = time.time()
                self.play_audio(audio_path)
                self._debug(f"Audio play took: {(time.time() - play_start)*1000:.1f}ms")
            return audio_path

        else:
            # gTTS (local, free)
            self._debug(f"Using gTTS")
            gtts_start = time.time()
            tts = gTTS(text, lang='iw')
            self._debug(f"gTTS init took: {(time.time() - gtts_start)*1000:.1f}ms")
            if output_path is None:
                # Use the same cache system as Google TTS
                save_start = time.time()
                tts.save(cache_path)
                self._debug(f"gTTS save took: {(time.time() - save_start)*1000:.1f}ms")
                self._debug(f"💾 Saved to cache: {cache_path}")
                audio_path = cache_path
            else:
                save_start = time.time()
                tts.save(output_path)
                self._debug(f"gTTS save took: {(time.time() - save_start)*1000:.1f}ms")
                audio_path = output_path

            self.last_audio_path = audio_path
            self._debug(f"Total gTTS time: {(time.time() - start_time)*1000:.1f}ms")
            if play_audio:
                play_start = time.time()
                self.play_audio(audio_path)
                self._debug(f"Audio play took: {(time.time() - play_start)*1000:.1f}ms")
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

    def split_text_to_sentences(self, text):
        """Split text into sentences, preserving Hebrew sentence structure"""
        # Hebrew sentence delimiters: period, question mark, exclamation mark
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def text_to_speech_by_sentences(self, text, gender='male', voice_id=None, output_dir=None, base_name="sentence", play_individual=False):
        """
        Generate TTS for each sentence individually, then concatenate into final MP3.
        
        Args:
            text (str): Full text to process
            gender (str): TTS gender ('male'/'female')
            voice_id (str): Specific voice ID to use
            output_dir (str): Directory to save individual sentence files (optional)
            base_name (str): Base name for individual files (default: "sentence")
            play_individual (bool): Whether to play each sentence as it's generated (default: False)
            
        Returns:
            dict: {
                'individual_files': [list of sentence MP3 paths],
                'combined_file': path to concatenated MP3,
                'sentence_texts': [list of sentence texts],
                'total_time_ms': processing time in milliseconds
            }
        """
        start_time = time.time()
        
        # Split text into sentences
        sentences = self.split_text_to_sentences(text)
        if not sentences:
            return None
            
        if self.debug:
            print(f"[SENTENCE_TTS] Split text into {len(sentences)} sentences")
            
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), 'hamaraN_sentence_tts')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate individual sentence MP3s
        individual_files = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            sentence_file = os.path.join(output_dir, f"{base_name}_{i+1:03d}.mp3")
            
            if self.debug:
                print(f"[SENTENCE_TTS] Processing sentence {i+1}/{len(sentences)}: '{sentence[:50]}...'")
            
            # Generate TTS for this sentence
            sentence_path = self.text_to_speech(
                sentence,
                gender=gender,
                voice_id=voice_id,
                output_path=sentence_file,
                play_audio=play_individual
            )
            
            if sentence_path:
                individual_files.append(sentence_path)
                if self.debug:
                    print(f"[SENTENCE_TTS] ✅ Generated: {sentence_file}")
            else:
                if self.debug:
                    print(f"[SENTENCE_TTS] ❌ Failed to generate: {sentence_file}")
        
        # Concatenate all individual files
        if individual_files:
            combined_file = self.concatenate_mp3_files(individual_files, output_dir, f"{base_name}_combined.mp3")
            
            processing_time = (time.time() - start_time) * 1000
            
            if self.debug:
                print(f"[SENTENCE_TTS] ✅ Combined {len(individual_files)} files into: {combined_file}")
                print(f"[SENTENCE_TTS] Total processing time: {processing_time:.1f}ms")
            
            return {
                'individual_files': individual_files,
                'combined_file': combined_file,
                'sentence_texts': sentences,
                'total_time_ms': processing_time
            }
        else:
            if self.debug:
                print(f"[SENTENCE_TTS] ❌ No sentences were successfully processed")
            return None

    def concatenate_mp3_files(self, mp3_files, output_dir, output_filename="combined.mp3"):
        """
        Concatenate multiple MP3 files into a single MP3 file using pydub.
        
        Args:
            mp3_files (list): List of MP3 file paths to concatenate
            output_dir (str): Directory to save the combined file
            output_filename (str): Name of the output file
            
        Returns:
            str: Path to the concatenated MP3 file
        """
        if not mp3_files:
            return None
            
        try:
            # Load first audio file
            combined = AudioSegment.from_mp3(mp3_files[0])
            
            # Add each subsequent file
            for mp3_file in mp3_files[1:]:
                if os.path.exists(mp3_file):
                    audio = AudioSegment.from_mp3(mp3_file)
                    combined += audio
                    if self.debug:
                        print(f"[CONCATENATE] Added: {mp3_file}")
                else:
                    if self.debug:
                        print(f"[CONCATENATE] ⚠️ File not found: {mp3_file}")
            
            # Export combined file
            output_path = os.path.join(output_dir, output_filename)
            combined.export(output_path, format="mp3")
            
            if self.debug:
                print(f"[CONCATENATE] ✅ Combined MP3 saved: {output_path}")
                print(f"[CONCATENATE] Total duration: {len(combined)/1000:.1f} seconds")
            
            return output_path
            
        except Exception as e:
            if self.debug:
                print(f"[CONCATENATE] ❌ Error concatenating files: {e}")
            return None
