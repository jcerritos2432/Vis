import cv2
import numpy as np
from collections import deque
import time
import pygame
import pygame.midi
import threading
import queue

class MotionControlledSynth:
    def __init__(self):
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Movement tracking parameters
        self.prev_frame = None
        self.movement_history = deque(maxlen=30)
        self.object_history = {}
        self.next_object_id = 0
        
        # Motion detection parameters
        self.min_contour_area = 100
        self.max_contour_area = 50000
        self.movement_threshold = 10
        
        # Display settings
        self.debug_mode = False
        self.show_trajectories = True
        self.fullscreen_mode = False
        
        # Screen size options
        self.screen_sizes = {
            '1': (320, 240),
            '2': (640, 480),
            '3': (800, 600),
            '4': (1024, 768),
            '5': (1280, 720)
        }
        self.current_size_index = '2'
        self.update_camera_resolution()
        
        # Audio synthesis parameters
        self.audio_enabled = True
        self.synth_params = {
            'base_freq': 110.0,  # Base frequency (A2)
            'freq_range': 440.0,  # Frequency range (up to A4)
            'lpf_cutoff': 2000.0,  # Low-pass filter cutoff
            'lpf_resonance': 0.1,  # Filter resonance
            'attack': 0.1,  # Attack time
            'decay': 0.3,   # Decay time
            'sustain': 0.7, # Sustain level
            'release': 0.5  # Release time
        }
        
        # Dance music parameters
        self.dance_params = {
            'bpm': 128,  # Beats per minute
            'beat_interval': 0.46875,  # 60/128 seconds
            'kick_freq': 60,  # Kick drum frequency
            'snare_freq': 200,  # Snare frequency
            'hihat_freq': 800,  # Hi-hat frequency
            'bass_freq': 55,  # Bass frequency
            'lead_freq': 220,  # Lead frequency
            'chord_progression': [0, 5, 3, 4],  # I-V-vi-IV progression
            'current_chord': 0,
            'beat_count': 0,
            'last_beat_time': 0,
            'measure_count': 0,
            'current_key': 0,  # C major
            'current_mode': 'major'
        }
        
        # Musical scales and chord progressions for better harmony
        self.major_scale = [0, 2, 4, 5, 7, 9, 11, 12]  # C major scale
        self.minor_scale = [0, 2, 3, 5, 7, 8, 10, 12]  # C minor scale
        self.current_scale = self.major_scale
        
        # Dance music chord progressions
        self.chord_progressions = {
            'house': [[0, 5, 3, 4], [0, 3, 4, 0], [0, 4, 5, 3]],  # I-V-vi-IV, I-vi-IV-I, I-IV-V-vi
            'techno': [[0, 5, 0, 5], [0, 3, 0, 3], [0, 4, 0, 4]],  # I-V-I-V, I-vi-I-vi, I-IV-I-IV
            'trance': [[0, 5, 3, 4], [0, 3, 4, 5], [0, 4, 5, 3]]   # I-V-vi-IV, I-vi-IV-V, I-IV-V-vi
        }
        
        # Dance music patterns
        self.dance_patterns = {
            'kick_pattern': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Four-on-the-floor
            'snare_pattern': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # On 2 and 4
            'hihat_pattern': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # Every beat
        }
        
        # Motion to audio mapping
        self.motion_mapping = {
            'avg_movement': 'lead_freq',     # Average movement -> lead frequency
            'object_count': 'bass_freq',     # Number of objects -> bass frequency
            'max_movement': 'energy',        # Maximum movement -> overall energy
            'movement_variance': 'rhythm'    # Movement variance -> rhythm complexity
        }
        
        # Audio thread and queue
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.running = True
        
        # Volume controls for different parameters
        self.volume_controls = {
            'kick': 0.25,      # Kick drum volume
            'snare': 0.15,     # Snare drum volume
            'hihat': 0.08,     # Hi-hat volume
            'bass': 0.2,       # Bass line volume
            'lead': 0.12,      # Lead melody volume
            'harmony': 0.08,   # Harmony volume
            'pad': 0.05        # Pad volume
        }
        
        # Volume control mode (which parameter to adjust)
        self.volume_control_mode = 'kick'  # Default to kick
        self.volume_modes = ['kick', 'snare', 'hihat', 'bass', 'lead', 'harmony', 'pad']
        
        # Initialize audio
        self.init_audio()
        
        print("Motion Controlled Synth initialized")
        print(f"Current resolution: {self.screen_sizes[self.current_size_index][0]}x{self.screen_sizes[self.current_size_index][1]}")
        print("\nControls:")
        print("  'd' - Toggle debug mode")
        print("  't' - Toggle trajectory display")
        print("  '+'/'-' - Adjust motion sensitivity")
        print("  '1'-'5' - Change screen size")
        print("  'f' - Toggle fullscreen")
        print("  'a' - Toggle audio")
        print("  Arrow keys - Adjust volume levels")
        print("  'v' - Cycle through volume parameters")
        print("  'q' - Quit")
        print("  'r' - Reset tracking")
    
    def init_audio(self):
        """Initialize pygame and MIDI for audio synthesis"""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.init()
            print("Audio system initialized successfully")
        except Exception as e:
            print(f"Failed to initialize audio: {e}")
            self.audio_enabled = False
    
    def calculate_movement_score(self, contour, prev_positions):
        """Calculate movement score (0-1) for a contour based on its movement history"""
        if not prev_positions:
            return 0.0
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        current_pos = (cx, cy)
        
        total_movement = 0
        for prev_pos in prev_positions:
            distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
            total_movement += distance
        
        max_possible_movement = len(prev_positions) * 100
        movement_score = min(1.0, total_movement / max_possible_movement)
        
        return movement_score
    
    def detect_moving_objects(self, frame):
        """Detect moving objects in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return []
        
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        moving_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    moving_objects.append({
                        'contour': contour,
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        self.prev_frame = gray
        return moving_objects
    
    def track_objects(self, moving_objects):
        """Track individual objects and assign IDs"""
        current_objects = {}
        
        for obj in moving_objects:
            obj_center = obj['center']
            best_match_id = None
            min_distance = float('inf')
            
            for obj_id, tracked_obj in self.object_history.items():
                if tracked_obj['positions']:
                    last_pos = tracked_obj['positions'][-1]
                    distance = np.sqrt((obj_center[0] - last_pos[0])**2 + (obj_center[1] - last_pos[1])**2)
                    
                    if distance < 50 and distance < min_distance:
                        min_distance = distance
                        best_match_id = obj_id
            
            if best_match_id is not None:
                self.object_history[best_match_id]['positions'].append(obj_center)
                self.object_history[best_match_id]['bbox'] = obj['bbox']
                self.object_history[best_match_id]['area'] = obj['area']
                current_objects[best_match_id] = self.object_history[best_match_id]
            else:
                new_id = self.next_object_id
                self.next_object_id += 1
                
                self.object_history[new_id] = {
                    'positions': [obj_center],
                    'bbox': obj['bbox'],
                    'area': obj['area'],
                    'color': tuple(np.random.randint(0, 255, 3).tolist())
                }
                current_objects[new_id] = self.object_history[new_id]
        
        for obj_id in list(self.object_history.keys()):
            if obj_id not in current_objects:
                if len(self.object_history[obj_id]['positions']) > 30:
                    del self.object_history[obj_id]
        
        return current_objects
    
    def calculate_object_movement_scores(self, tracked_objects):
        """Calculate movement scores for all tracked objects"""
        scores = {}
        
        for obj_id, obj_data in tracked_objects.items():
            positions = obj_data['positions']
            if len(positions) > 1:
                recent_positions = positions[-10:]
                movement_score = self.calculate_movement_score_from_positions(recent_positions)
                scores[obj_id] = movement_score
            else:
                scores[obj_id] = 0.0
        
        return scores
    
    def calculate_movement_score_from_positions(self, positions):
        """Calculate movement score from a list of positions"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance
        
        avg_distance = total_distance / (len(positions) - 1)
        max_expected_distance = 50
        movement_score = min(1.0, avg_distance / max_expected_distance)
        
        return movement_score
    
    def map_motion_to_audio(self, tracked_objects, movement_scores):
        """Map motion parameters to dance music synthesis parameters"""
        if not tracked_objects or not movement_scores:
            return
        
        # Calculate motion metrics
        avg_movement = np.mean(list(movement_scores.values())) if movement_scores else 0.0
        max_movement = np.max(list(movement_scores.values())) if movement_scores else 0.0
        object_count = len(tracked_objects)
        movement_variance = np.var(list(movement_scores.values())) if len(movement_scores) > 1 else 0.0
        
        # Map to dance music parameters
        current_time = time.time()
        
        # Update beat timing and musical structure
        if current_time - self.dance_params['last_beat_time'] >= self.dance_params['beat_interval']:
            self.dance_params['beat_count'] += 1
            self.dance_params['last_beat_time'] = current_time
            
            # Change chord every 4 beats (measure)
            if self.dance_params['beat_count'] % 4 == 0:
                self.dance_params['current_chord'] = (self.dance_params['current_chord'] + 1) % 4
                self.dance_params['measure_count'] += 1
                
                # Change key every 8 measures for variety
                if self.dance_params['measure_count'] % 8 == 0:
                    self.dance_params['current_key'] = (self.dance_params['current_key'] + 7) % 12  # Move to relative minor
                    self.dance_params['current_mode'] = 'minor' if self.dance_params['current_mode'] == 'major' else 'major'
                    self.current_scale = self.minor_scale if self.dance_params['current_mode'] == 'minor' else self.major_scale
        
        # Map motion to musical parameters with better musical logic
        energy = max_movement  # 0-1 energy level
        rhythm_complexity = min(1.0, movement_variance * 3)  # Rhythm variation
        
        # Lead melody based on movement
        lead_note = int(avg_movement * 7) % 7  # 7 notes in scale
        lead_freq = self.get_note_frequency(lead_note, 220)  # A3 base
        
        # Bass line based on object count and current chord
        chord_root = self.dance_params['chord_progression'][self.dance_params['current_chord']]
        bass_note = (chord_root + object_count % 3) % 7  # Chord tones
        bass_freq = self.get_note_frequency(bass_note, 55)  # A1 base
        
        # Harmony based on current chord
        chord_notes = self.get_chord_notes(chord_root, self.dance_params['current_mode'])
        harmony_freqs = [self.get_note_frequency(note, 110) for note in chord_notes]  # A2 base
        
        # Update dance parameters
        self.dance_params.update({
            'current_lead_freq': lead_freq,
            'current_bass_freq': bass_freq,
            'harmony_freqs': harmony_freqs,
            'energy': energy,
            'rhythm_complexity': rhythm_complexity,
            'current_time': current_time,
            'current_chord_root': chord_root
        })
        
        # Send to audio thread
        if self.audio_enabled:
            try:
                self.audio_queue.put_nowait({
                    'lead_freq': lead_freq,
                    'bass_freq': bass_freq,
                    'harmony_freqs': harmony_freqs,
                    'energy': energy,
                    'rhythm_complexity': rhythm_complexity,
                    'beat_count': self.dance_params['beat_count'],
                    'object_count': object_count,
                    'avg_movement': avg_movement,
                    'current_time': current_time,
                    'chord_root': chord_root
                })
            except queue.Full:
                pass  # Skip if queue is full
    
    def get_chord_notes(self, root, mode):
        """Get chord notes for a given root and mode"""
        if mode == 'major':
            # Major chord: root, major third, perfect fifth
            return [root, (root + 4) % 12, (root + 7) % 12]
        else:
            # Minor chord: root, minor third, perfect fifth
            return [root, (root + 3) % 12, (root + 7) % 12]
    
    def get_note_frequency(self, note_index, base_freq):
        """Convert note index to frequency using current scale"""
        if note_index >= len(self.current_scale):
            note_index = note_index % len(self.current_scale)
        
        semitone_offset = self.current_scale[int(note_index)]
        return base_freq * (2 ** (semitone_offset / 12))
    
    def audio_thread_func(self):
        """Dance music synthesis thread"""
        while self.running:
            try:
                # Get audio parameters from queue
                params = self.audio_queue.get(timeout=0.1)
                
                # Generate dance music elements
                sample_rate = 44100
                duration = 0.1
                samples = int(duration * sample_rate)
                t = np.linspace(0, duration, samples)
                
                # Initialize mix with proper levels
                mix = np.zeros(samples)
                
                # Get current beat position in 16-step pattern
                step = params['beat_count'] % 16
                
                # Generate kick drum using pattern (with volume control)
                if self.dance_patterns['kick_pattern'][step]:
                    kick = self.generate_kick_drum(samples, sample_rate, params['energy'])
                    mix += kick * self.volume_controls['kick']
                
                # Generate snare using pattern (with volume control)
                if self.dance_patterns['snare_pattern'][step]:
                    snare = self.generate_snare_drum(samples, sample_rate, params['energy'])
                    mix += snare * self.volume_controls['snare']
                
                # Generate hi-hats using pattern (with volume control)
                if self.dance_patterns['hihat_pattern'][step]:
                    hihat = self.generate_hihat(samples, sample_rate, params['rhythm_complexity'])
                    mix += hihat * self.volume_controls['hihat']
                
                # Generate bass line (every beat, with volume control)
                bass = self.generate_bass_line(samples, sample_rate, params['bass_freq'], params['energy'])
                mix += bass * self.volume_controls['bass']
                
                # Generate lead melody (every other beat, with volume control)
                if step % 2 == 0:
                    lead = self.generate_lead_melody(samples, sample_rate, params['lead_freq'], params['energy'])
                    mix += lead * self.volume_controls['lead']
                
                # Generate harmony (on chord changes, with volume control)
                if step % 4 == 0:
                    harmony = self.generate_harmony(samples, sample_rate, params['harmony_freqs'], params['energy'])
                    mix += harmony * self.volume_controls['harmony']
                
                # Generate pad (sustained chords, with volume control)
                pad = self.generate_pad(samples, sample_rate, params['harmony_freqs'], params['energy'])
                mix += pad * self.volume_controls['pad']
                
                # Apply gentle compression first
                mix = self.apply_gentle_compression(mix)
                
                # Apply EQ for dance music
                mix = self.apply_dance_eq(mix, sample_rate)
                
                # Apply final limiting to prevent clipping
                mix = self.apply_limiting(mix)
                
                # Convert to stereo with slight panning
                left_channel = mix * 0.9
                right_channel = mix * 0.9
                wave_stereo = np.column_stack((left_channel, right_channel))
                
                # Convert to 16-bit audio with proper scaling
                wave_stereo = (wave_stereo * 16383).astype(np.int16)  # Reduced from 32767
                
                # Play audio
                sound = pygame.sndarray.make_sound(wave_stereo)
                sound.play()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio error: {e}")
                continue
    
    def generate_kick_drum(self, samples, sample_rate, energy):
        """Generate kick drum sound"""
        t = np.linspace(0, samples/sample_rate, samples)
        
        # Pitch envelope (frequency sweep)
        start_freq = 80 + energy * 40
        end_freq = 40 + energy * 20
        freq_sweep = start_freq * np.exp(-t * 8)
        
        # Kick envelope (softer)
        envelope = np.exp(-t * 6) * (0.8 + energy * 0.4)
        
        # Generate kick sound with pitch sweep
        kick = np.sin(2 * np.pi * freq_sweep * t) * envelope
        kick += np.sin(2 * np.pi * freq_sweep * 0.5 * t) * envelope * 0.3  # Sub-harmonic
        
        # Add some click at the start
        click = np.exp(-t * 100) * 0.2
        kick += click
        
        return kick
    
    def generate_snare_drum(self, samples, sample_rate, energy):
        """Generate snare drum sound"""
        t = np.linspace(0, samples/sample_rate, samples)
        
        # Snare envelope (softer)
        envelope = np.exp(-t * 8) * (0.6 + energy * 0.3)
        
        # Generate snare sound (filtered noise + tone)
        noise = np.random.randn(samples)
        # Apply simple low-pass filter to noise
        filtered_noise = np.convolve(noise, np.ones(5)/5, mode='same')
        filtered_noise = filtered_noise * envelope
        
        # Tone component
        tone = np.sin(2 * np.pi * 200 * t) * envelope * 0.4
        tone += np.sin(2 * np.pi * 400 * t) * envelope * 0.2
        
        snare = filtered_noise * 0.6 + tone * 0.4
        return snare
    
    def generate_hihat(self, samples, sample_rate, complexity):
        """Generate hi-hat sound"""
        t = np.linspace(0, samples/sample_rate, samples)
        
        # Hi-hat envelope (softer)
        envelope = np.exp(-t * 30) * (0.4 + complexity * 0.3)
        
        # Generate hi-hat sound (filtered high-frequency noise)
        noise = np.random.randn(samples)
        # Apply high-pass filter effect
        filtered_noise = noise - np.convolve(noise, np.ones(10)/10, mode='same')
        hihat = filtered_noise * envelope * 0.3
        
        # Add some metallic character
        metallic = np.sin(2 * np.pi * 8000 * t) * envelope * 0.1
        hihat += metallic
        
        return hihat
    
    def generate_bass_line(self, samples, sample_rate, freq, energy):
        """Generate bass line"""
        t = np.linspace(0, samples/sample_rate, samples)
        
        # Bass envelope (softer)
        envelope = np.exp(-t * 3) * (0.7 + energy * 0.3)
        
        # Generate bass sound with harmonics
        bass = np.sin(2 * np.pi * freq * t) * envelope
        bass += np.sin(2 * np.pi * freq * 2 * t) * envelope * 0.2  # 2nd harmonic
        bass += np.sin(2 * np.pi * freq * 3 * t) * envelope * 0.1  # 3rd harmonic
        
        # Add some saturation for warmth
        bass = np.tanh(bass * 0.5) * 0.8
        
        return bass
    
    def generate_lead_melody(self, samples, sample_rate, freq, energy):
        """Generate lead melody"""
        t = np.linspace(0, samples/sample_rate, samples)
        
        # Lead envelope (softer)
        envelope = np.exp(-t * 4) * (0.6 + energy * 0.3)
        
        # Generate lead sound with harmonics
        lead = np.sin(2 * np.pi * freq * t) * envelope
        lead += np.sin(2 * np.pi * freq * 2 * t) * envelope * 0.3  # 2nd harmonic
        lead += np.sin(2 * np.pi * freq * 3 * t) * envelope * 0.15  # 3rd harmonic
        
        # Add some chorus effect
        chorus = np.sin(2 * np.pi * (freq + 2) * t) * envelope * 0.1
        lead += chorus
        
        # Apply gentle saturation
        lead = np.tanh(lead * 0.3) * 0.7
        
        return lead
    
    def generate_harmony(self, samples, sample_rate, harmony_freqs, energy):
        """Generate harmony from chord frequencies"""
        t = np.linspace(0, samples/sample_rate, samples)
        
        # Harmony envelope
        envelope = np.exp(-t * 3) * (1 + energy * 0.5)
        
        # Generate harmony by mixing chord tones
        harmony = np.zeros(samples)
        for freq in harmony_freqs:
            harmony += np.sin(2 * np.pi * freq * t) * envelope * 0.3
        
        return harmony
    
    def generate_pad(self, samples, sample_rate, harmony_freqs, energy):
        """Generate sustained pad sound"""
        t = np.linspace(0, samples/sample_rate, samples)
        
        # Pad envelope (longer sustain)
        envelope = np.exp(-t * 1) * (1 + energy * 0.3)
        
        # Generate pad with multiple harmonics
        pad = np.zeros(samples)
        for freq in harmony_freqs:
            # Add multiple harmonics for rich pad sound
            pad += np.sin(2 * np.pi * freq * t) * envelope * 0.2
            pad += np.sin(2 * np.pi * freq * 2 * t) * envelope * 0.1
            pad += np.sin(2 * np.pi * freq * 3 * t) * envelope * 0.05
        
        return pad
    
    def apply_gentle_compression(self, audio):
        """Apply gentle compression to control dynamics"""
        # Gentle compression settings
        threshold = 0.3
        ratio = 2.0
        
        # Apply compression
        compressed = np.where(np.abs(audio) > threshold, 
                            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
                            audio)
        
        return compressed
    
    def apply_limiting(self, audio):
        """Apply limiting to prevent clipping"""
        # Soft limiting
        limit = 0.8
        limited = np.where(np.abs(audio) > limit, 
                          np.sign(audio) * limit * (1 - np.exp(-(np.abs(audio) - limit))),
                          audio)
        
        # Final hard limit
        limited = np.clip(limited, -0.85, 0.85)
        
        return limited
    
    def apply_compression(self, audio, energy):
        """Apply compression and limiting"""
        # Simple compression
        threshold = 0.7
        ratio = 4.0
        
        # Apply compression
        compressed = np.where(np.abs(audio) > threshold, 
                            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
                            audio)
        
        # Apply limiting
        compressed = np.clip(compressed, -0.9, 0.9)
        
        # Normalize
        if np.max(np.abs(compressed)) > 0:
            compressed = compressed / np.max(np.abs(compressed)) * 0.8
        
        return compressed
    
    def apply_dance_eq(self, audio, sample_rate):
        """Apply EQ optimized for dance music"""
        # Simple EQ using FFT
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
        
        # Gentle bass boost (60-250 Hz)
        bass_mask = (np.abs(freqs) >= 60) & (np.abs(freqs) <= 250)
        fft[bass_mask] *= 1.2
        
        # Gentle kick boost (60-90 Hz)
        kick_mask = (np.abs(freqs) >= 60) & (np.abs(freqs) <= 90)
        fft[kick_mask] *= 1.3
        
        # Gentle high frequency boost (8k-16k Hz)
        high_mask = (np.abs(freqs) >= 8000) & (np.abs(freqs) <= 16000)
        fft[high_mask] *= 1.1
        
        # Gentle mud cut (200-400 Hz)
        mud_mask = (np.abs(freqs) >= 200) & (np.abs(freqs) <= 400)
        fft[mud_mask] *= 0.9
        
        # Convert back to time domain
        eq_audio = np.real(np.fft.ifft(fft))
        
        return eq_audio
    
    def draw_objects_and_scores(self, frame, tracked_objects, movement_scores):
        """Draw tracked objects and their movement scores"""
        for obj_id, obj_data in tracked_objects.items():
            if obj_id in movement_scores:
                score = movement_scores[obj_id]
                bbox = obj_data['bbox']
                color = obj_data['color']
                
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                label = f"ID:{obj_id} Score:{score:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if self.show_trajectories and len(obj_data['positions']) > 1:
                    positions = obj_data['positions']
                    for i in range(1, len(positions)):
                        cv2.line(frame, positions[i-1], positions[i], color, 2)
                
                if score > 0.5:
                    cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 255, 0), -1)
                elif score > 0.2:
                    cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 255, 255), -1)
                else:
                    cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 0, 255), -1)
    
    def draw_debug_info(self, frame, moving_objects, tracked_objects, movement_scores):
        """Draw debug information including audio parameters"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            debug_threshold = max(5, self.movement_threshold - 5)
            thresh = cv2.threshold(frame_diff, debug_threshold, 255, cv2.THRESH_BINARY)[1]
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            height, width = frame.shape[:2]
            debug_width = width // 2
            debug_height = height // 2
            
            motion_mask_resized = cv2.resize(thresh, (debug_width, debug_height))
            motion_mask_enhanced = cv2.cvtColor(motion_mask_resized, cv2.COLOR_GRAY2BGR)
            motion_mask_enhanced[motion_mask_resized == 255] = [0, 255, 255]
            motion_mask_enhanced[motion_mask_resized == 0] = [20, 20, 20]
            
            if np.sum(motion_mask_resized) == 0:
                cv2.circle(motion_mask_enhanced, (debug_width//2, debug_height//2), 20, [0, 255, 255], -1)
                cv2.putText(motion_mask_enhanced, "NO MOTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 255], 1)
            
            frame[10:10+debug_height, width-debug_width-10:width-10] = motion_mask_enhanced
            
            cv2.rectangle(frame, (width-debug_width-10, 10), (width-10, 10+debug_height), (255, 255, 255), 2)
            cv2.putText(frame, f"Motion Mask (Threshold: {self.movement_threshold})", (width-debug_width-10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw statistics including audio parameters
        current_width, current_height = self.screen_sizes[self.current_size_index]
        avg_movement = np.mean(list(movement_scores.values())) if movement_scores else 0.0
        object_count = len(tracked_objects)
        
        stats_text = [
            f"Objects tracked: {len(tracked_objects)}",
            f"Moving objects: {len(moving_objects)}",
            f"Avg movement: {avg_movement:.2f}",
            f"Motion threshold: {self.movement_threshold}",
            f"Min contour area: {self.min_contour_area}",
            f"Resolution: {current_width}x{current_height}",
            f"Audio: {'ON' if self.audio_enabled else 'OFF'}",
            f"BPM: {self.dance_params['bpm']}",
            f"Beat: {self.dance_params['beat_count'] % 4 + 1}/4",
            f"Energy: {self.dance_params.get('energy', 0):.2f}",
            f"Lead: {self.dance_params.get('current_lead_freq', 0):.0f} Hz",
            f"Bass: {self.dance_params.get('current_bass_freq', 0):.0f} Hz",
            f"Volume: {self.volume_control_mode} ({self.volume_controls[self.volume_control_mode]:.2f})",
            f"Press +/- to adjust sensitivity, v to cycle volume, arrows to adjust"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    def update_camera_resolution(self):
        """Update camera resolution based on current size index"""
        width, height = self.screen_sizes[self.current_size_index]
        
        self.cap.release()
        self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        time.sleep(0.2)
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if abs(actual_width - width) > 50 or abs(actual_height - height) > 50:
            print(f"Warning: Camera doesn't support {width}x{height}, using {int(actual_width)}x{int(actual_height)}")
            self.screen_sizes[self.current_size_index] = (int(actual_width), int(actual_height))
            width = int(actual_width)
        
        if width <= 320:
            self.min_contour_area = 30
            self.movement_threshold = 8
        elif width <= 640:
            self.min_contour_area = 100
            self.movement_threshold = 10
        elif width <= 800:
            self.min_contour_area = 150
            self.movement_threshold = 12
        else:
            self.min_contour_area = 200
            self.movement_threshold = 15
    
    def change_screen_size(self, size_index):
        """Change screen size"""
        if size_index in self.screen_sizes:
            try:
                self.current_size_index = size_index
                self.update_camera_resolution()
                self.prev_frame = None
                width, height = self.screen_sizes[size_index]
                print(f"Screen size changed to: {width}x{height}")
                print(f"Motion threshold: {self.movement_threshold}, Min contour area: {self.min_contour_area}")
            except Exception as e:
                print(f"Failed to change to size {size_index}: {e}")
                self.current_size_index = '2'
                self.update_camera_resolution()
                self.prev_frame = None
        else:
            print(f"Invalid size option. Available: 1-5")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen_mode = not self.fullscreen_mode
        if self.fullscreen_mode:
            cv2.setWindowProperty('Motion Controlled Synth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Fullscreen mode: ON")
        else:
            cv2.setWindowProperty('Motion Controlled Synth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("Fullscreen mode: OFF")
    
    def adjust_motion_sensitivity(self, direction):
        """Adjust motion detection sensitivity"""
        if direction == 'increase':
            self.movement_threshold = max(1, self.movement_threshold - 2)
            print(f"Motion threshold decreased to: {self.movement_threshold}")
        elif direction == 'decrease':
            self.movement_threshold = min(50, self.movement_threshold + 2)
            print(f"Motion threshold increased to: {self.movement_threshold}")
    
    def cycle_volume_parameter(self):
        """Cycle through volume parameters to adjust"""
        current_index = self.volume_modes.index(self.volume_control_mode)
        next_index = (current_index + 1) % len(self.volume_modes)
        self.volume_control_mode = self.volume_modes[next_index]
        print(f"Volume control: {self.volume_control_mode} (current: {self.volume_controls[self.volume_control_mode]:.2f})")
    
    def adjust_volume(self, direction):
        """Adjust volume of current parameter"""
        current_volume = self.volume_controls[self.volume_control_mode]
        
        if direction == 'up':
            new_volume = min(1.0, current_volume + 0.05)
        else:  # down
            new_volume = max(0.0, current_volume - 0.05)
        
        self.volume_controls[self.volume_control_mode] = new_volume
        print(f"{self.volume_control_mode} volume: {new_volume:.2f}")
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.object_history = {}
        self.next_object_id = 0
        self.prev_frame = None
        print("Tracking reset")
    
    def run(self):
        """Main loop"""
        print("Starting Motion Controlled Synth...")
        
        # Start audio thread
        if self.audio_enabled:
            self.audio_thread = threading.Thread(target=self.audio_thread_func, daemon=True)
            self.audio_thread.start()
        
        cv2.namedWindow('Motion Controlled Synth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Motion Controlled Synth', 640, 480)
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame - trying to reconnect...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(0)
                    self.update_camera_resolution()
                    continue
            except Exception as e:
                print(f"Camera read error: {e} - trying to reconnect...")
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
                self.update_camera_resolution()
                continue
            
            moving_objects = self.detect_moving_objects(frame)
            tracked_objects = self.track_objects(moving_objects)
            movement_scores = self.calculate_object_movement_scores(tracked_objects)
            
            # Map motion to audio
            self.map_motion_to_audio(tracked_objects, movement_scores)
            
            if self.debug_mode:
                self.draw_debug_info(frame, moving_objects, tracked_objects, movement_scores)
            
            self.draw_objects_and_scores(frame, tracked_objects, movement_scores)
            
            cv2.putText(frame, "Press 'd' for debug, 't' for trajectories, '+/-' for sensitivity, '1-5' for size, 'f' for fullscreen, 'a' for audio, 'v' for volume, arrows to adjust, 'r' to reset, 'q' to quit", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Motion Controlled Synth', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('t'):
                self.show_trajectories = not self.show_trajectories
                print(f"Trajectory display: {'ON' if self.show_trajectories else 'OFF'}")
            elif key == ord('f'):
                self.toggle_fullscreen()
            elif key == ord('a'):
                self.audio_enabled = not self.audio_enabled
                print(f"Audio: {'ON' if self.audio_enabled else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.adjust_motion_sensitivity('increase')
            elif key == ord('-') or key == ord('_'):
                self.adjust_motion_sensitivity('decrease')
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                self.change_screen_size(chr(key))
            elif key == ord('v'):
                self.cycle_volume_parameter()
            elif key == ord('r'):
                self.reset_tracking()
            # Arrow key handling for volume control
            elif key == 82:  # Up arrow
                self.adjust_volume('up')
            elif key == 84:  # Down arrow
                self.adjust_volume('down')
        
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    synth = MotionControlledSynth()
    synth.run() 