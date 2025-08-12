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
    def __init__(self, engine='gtts', debug=False, speed_rate=1.0, slow=False, lang_check=False, tld='co.il'):
        self.engine = engine
        self.last_audio = None
        self.last_audio_rate = None
        self.last_audio_path = None
        self.debug = debug
        self._google_client = None  # Cache Google Cloud client
        self._client_initialized = False  # Track if client is ready
        self.speed_rate = speed_rate  # 1.0 = normal, 1.1 = 10% faster, 0.9 = 10% slower
        
        # Additional gTTS parameters for better quality
        self.slow = slow              # True = slower, clearer speech
        self.lang_check = lang_check  # False = faster initialization (skip language validation)
        self.tld = tld               # Google domain: 'co.il' for Israeli accent (default), 'com' standard, 'co.uk' British

    def _debug(self, msg):
        if getattr(self, 'debug', False):
            print(f"[B_DEBUG] {msg}")

    def _get_cache_path(self, text, engine):
        """Generate cache file path based on text hash"""
        # Clean text for consistent hashing (remove SSML tags, normalize whitespace)
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        text_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()[:12]
        temp_dir = os.path.join(tempfile.gettempdir(), 'hamaraN_tts_cache')
        os.makedirs(temp_dir, exist_ok=True)
        cache_file = os.path.join(temp_dir, f'{engine}_{text_hash}.mp3')
        return cache_file

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
            voice_name = voice_id if voice_id else 'he-IL-Wavenet-C'
            voice = texttospeech.VoiceSelectionParams(
                language_code="he-IL",
                name=voice_name
            )
            # Apply speed rate - for Google TTS: 1.0 = normal, >1.0 = faster
            google_speaking_rate = 0.90 * self.speed_rate
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=google_speaking_rate
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
            self._debug(f"Using gTTS (slow={self.slow}, tld={self.tld}, lang_check={self.lang_check})")
            gtts_start = time.time()
            tts = gTTS(text=text, lang='iw', slow=self.slow, lang_check=self.lang_check, tld=self.tld)
            self._debug(f"gTTS init took: {(time.time() - gtts_start)*1000:.1f}ms")
            
            # Create temporary file for gTTS if speed adjustment is needed
            temp_path = None
            if self.speed_rate != 1.0:
                temp_path = cache_path + '_temp.mp3'
                save_start = time.time()
                tts.save(temp_path)
                self._debug(f"gTTS save took: {(time.time() - save_start)*1000:.1f}ms")
                
                # Apply speed adjustment using pydub
                speed_start = time.time()
                audio = AudioSegment.from_mp3(temp_path)
                # Speed up audio: speed_rate > 1.0 = faster, < 1.0 = slower
                fast_audio = audio.speedup(playback_speed=self.speed_rate)
                
                # Save speed-adjusted audio
                final_path = output_path if output_path else cache_path
                fast_audio.export(final_path, format="mp3")
                self._debug(f"Speed adjustment took: {(time.time() - speed_start)*1000:.1f}ms (rate: {self.speed_rate}x)")
                
                # Clean up temp file
                os.remove(temp_path)
                audio_path = final_path
                
            else:
                # No speed adjustment needed
                if output_path is None:
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

        # Try direct system player first (more reliable on Windows)
        if os.name == 'nt':  # Windows
            try:
                print(f"🔊 Playing audio with system player...")
                import subprocess
                import time
                
                # Wait a moment for file to be fully written
                time.sleep(0.1)
                
                # Use the Windows start command to open with default MP3 player
                result = subprocess.run(['cmd', '/c', 'start', '', os.path.abspath(audio_path)], 
                                      capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    print(f"✅ Audio player launched successfully!")
                    return  # Exit early if system player works
                else:
                    print(f"❌ System player failed: {result.stderr}")
            except Exception as e:
                print(f"[AUDIO_ERROR] System player failed: {e}")
        
        # Fallback: Direct audio playback for faster response  
        try:
            import threading
            # Load and play audio in a separate thread for non-blocking behavior
            def play_in_thread():
                try:
                    print(f"🔊 Starting direct audio playback...")
                    audio = AudioSegment.from_mp3(audio_path)
                    # Convert to numpy array for sounddevice
                    audio_array = np.array(audio.get_array_of_samples())
                    if audio.channels == 2:
                        audio_array = audio_array.reshape((-1, 2))
                    
                    # Play audio directly
                    sd.play(audio_array, samplerate=audio.frame_rate)
                    sd.wait()  # Wait for playback to complete
                    print(f"✅ Direct audio playback completed!")
                except Exception as e:
                    print(f"[AUDIO_ERROR] Direct playback failed: {e}")
            
            # Start playback in background thread
            audio_thread = threading.Thread(target=play_in_thread, daemon=True)
            audio_thread.start()
            
        except Exception as e:
            print(f"[AUDIO_ERROR] Threading failed: {e}")
            # Fallback to original system player method
            if os.name == 'nt':  # Windows
                import subprocess
                subprocess.Popen(['start', '', os.path.abspath(audio_path)], shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif os.name == 'posix':  # Linux/Mac
                import subprocess
                subprocess.Popen(['xdg-open', audio_path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

    def text_to_speech_line_by_line(self, text, gender='male', voice_id=None, output_dir=None, base_name="line", play_sequential=True):
        """
        Generate separate TTS MP3 for each line and optionally play them sequentially.
        
        Args:
            text (str): Full text to process
            gender (str): TTS gender ('male'/'female')
            voice_id (str): Specific voice ID to use
            output_dir (str): Directory to save individual line files (optional)
            base_name (str): Base name for individual files (default: "line")
            play_sequential (bool): Whether to play each line sequentially (default: True)
            
        Returns:
            dict: {
                'individual_files': [list of line MP3 paths],
                'line_texts': [list of line texts],
                'total_time_ms': processing time in milliseconds
            }
        """
        start_time = time.time()
        
        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None
            
        if self.debug:
            print(f"[LINE_TTS] Processing {len(lines)} lines individually...")
            
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), 'hamaraN_line_tts')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate individual line MP3s
        individual_files = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            line_file = os.path.join(output_dir, f"{base_name}_{i+1:03d}.mp3")
            
            if self.debug:
                print(f"[LINE_TTS] Processing line {i+1}/{len(lines)}: '{line[:50]}{'...' if len(line) > 50 else ''}'")
            
            # Generate TTS for this line
            line_path = self.text_to_speech(
                line,
                gender=gender,
                voice_id=voice_id,
                output_path=line_file,
                play_audio=False  # Don't play individual files yet
            )
            
            if line_path:
                individual_files.append(line_path)
                if self.debug:
                    print(f"[LINE_TTS] ✅ Generated: {line_file}")
                    
                # Play this line immediately and wait for it to finish
                if play_sequential:
                    if self.debug:
                        print(f"[LINE_TTS] 🔊 Playing line {i+1}: '{line[:30]}{'...' if len(line) > 30 else ''}'")
                    self.play_audio_sync(line_path)  # Wait for this line to finish before next
                    
            else:
                if self.debug:
                    print(f"[LINE_TTS] ❌ Failed to generate: {line_file}")
        
        processing_time = (time.time() - start_time) * 1000
        
        if self.debug:
            print(f"[LINE_TTS] ✅ Generated {len(individual_files)} line MP3 files")
            print(f"[LINE_TTS] Total processing time: {processing_time:.1f}ms")
        
        return {
            'individual_files': individual_files,
            'line_texts': lines,
            'total_time_ms': processing_time
        }

    def play_audio_sync(self, audio_path=None):
        """Play audio synchronously (wait for playback to complete)"""
        if audio_path is None:
            audio_path = self.last_audio_path
        if audio_path is None:
            raise Exception('No audio file to play.')

        try:
            # Use pydub + sounddevice for synchronous playback
            audio = AudioSegment.from_mp3(audio_path)
            # Convert to numpy array for sounddevice
            audio_array = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                audio_array = audio_array.reshape((-1, 2))
            
            # Play audio and WAIT for completion
            sd.play(audio_array, samplerate=audio.frame_rate)
            sd.wait()  # This blocks until playback is complete
            
            if self.debug:
                print(f"[SYNC_AUDIO] ✅ Finished playing: {os.path.basename(audio_path)}")
                
        except Exception as e:
            if self.debug:
                print(f"[SYNC_AUDIO] ❌ Error playing {audio_path}: {e}")
            # Fallback to system player with wait
            if os.name == 'nt':  # Windows
                import subprocess
                subprocess.run(['cmd', '/c', 'start', '/wait', '', os.path.abspath(audio_path)], 
                             capture_output=True, shell=True)
