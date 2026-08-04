"""
Alert Engine — DrowsyGuard
CoderAxo Internship CAX-OL-2026-290
Author: Muhammad Azeen Waqas
Institution: COMSATS University Islamabad, Wah Campus
"""
import pygame
import threading
import wave
import struct
import math
import os

def generate_tone(filename, freq=1000, duration=0.3, volume=0.5, rate=44100):
    if os.path.exists(filename):
        return
    with wave.open(filename, 'w') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        for i in range(int(rate * duration)):
            val = int(volume * 32767 * math.sin(2 * math.pi * freq * i / rate))
            f.writeframes(struct.pack('<h', val))

class AlertEngine:
    def __init__(self, beep_path='beep.wav', alarm_path='alarm.wav'):
        pygame.mixer.init()
        generate_tone(beep_path,  freq=1000, duration=0.3, volume=0.6)
        generate_tone(alarm_path, freq=1800, duration=0.9, volume=0.9)
        self.beep_path  = beep_path
        self.alarm_path = alarm_path
        self.alerting   = False
        self._thread    = None

    def trigger(self, level='SOFT'):
        if not self.alerting:
            self.alerting = True
            path = self.beep_path if level == 'SOFT' else self.alarm_path
            self._thread = threading.Thread(target=self._play, args=(path,), daemon=True)
            self._thread.start()

    def stop(self):
        self.alerting = False
        pygame.mixer.stop()

    def _play(self, path):
        sound = pygame.mixer.Sound(path)
        while self.alerting:
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
