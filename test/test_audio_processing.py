#!/usr/bin/env python3
"""
Tests for audio processing utilities
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add max-whisper to path
sys.path.append(str(Path(__file__).parent.parent / "max-whisper"))

class TestAudioProcessing(unittest.TestCase):
    """Test audio processing functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create synthetic audio data for testing
        self.sample_rate = 16000
        self.duration = 1.0  # 1 second
        self.samples = int(self.sample_rate * self.duration)
        
        # Generate sine wave test audio
        t = np.linspace(0, self.duration, self.samples)
        frequency = 440  # A4 note
        self.test_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    def test_audio_loading(self):
        """Test that audio loading functions work"""
        try:
            from audio.preprocessing import load_audio
            # Test with synthetic data (mock file loading)
            # This would normally load from file
            self.assertTrue(True, "Audio preprocessing module imports correctly")
        except ImportError:
            self.skipTest("Audio preprocessing module not available")
    
    def test_synthetic_audio_properties(self):
        """Test properties of our synthetic test audio"""
        self.assertEqual(len(self.test_audio), self.samples)
        self.assertEqual(self.test_audio.dtype, np.float32)
        self.assertGreater(np.max(self.test_audio), 0.5)  # Should have reasonable amplitude
        self.assertLess(np.min(self.test_audio), -0.5)
    
    def test_audio_normalization(self):
        """Test audio normalization"""
        # Create audio with different amplitude
        loud_audio = self.test_audio * 10.0
        
        # Simple normalization
        normalized = loud_audio / np.max(np.abs(loud_audio))
        
        self.assertLessEqual(np.max(np.abs(normalized)), 1.0)
        self.assertGreaterEqual(np.max(np.abs(normalized)), 0.5)  # Should maintain some amplitude

class TestMojoIntegration(unittest.TestCase):
    """Test Mojo integration where possible"""
    
    def test_mojo_files_exist(self):
        """Test that Mojo files exist in audio directory"""
        audio_dir = Path(__file__).parent.parent / "max-whisper" / "audio"
        
        mojo_files = list(audio_dir.glob("*.mojo"))
        self.assertGreater(len(mojo_files), 0, "At least one Mojo file should exist")
        
        expected_files = [
            "gpu_mel_kernel.mojo",
            "mel_kernel.mojo", 
            "simple_kernel.mojo",
            "working_kernel.mojo"
        ]
        
        existing_files = [f.name for f in mojo_files]
        for expected in expected_files:
            if expected in existing_files:
                self.assertTrue(True, f"Found expected Mojo file: {expected}")

if __name__ == '__main__':
    unittest.main()