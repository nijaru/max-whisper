#!/usr/bin/env python3
"""
Unit tests for Whisper implementations
Tests basic functionality without requiring heavy model loading
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add max-whisper to path
sys.path.append(str(Path(__file__).parent.parent / "max-whisper"))

class TestWhisperImplementations(unittest.TestCase):
    """Test basic functionality of all implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_audio_file = "audio_samples/modular_video.wav"
        
    def test_cpu_import(self):
        """Test that CPU implementation can be imported"""
        try:
            from whisper_cpu import WhisperCPU
            self.assertTrue(True, "CPU implementation imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import CPU implementation: {e}")
    
    def test_gpu_import(self):
        """Test that GPU implementation can be imported"""
        try:
            from whisper_gpu import WhisperGPU
            self.assertTrue(True, "GPU implementation imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import GPU implementation: {e}")
    
    def test_max_import(self):
        """Test that MAX Graph implementation can be imported"""
        try:
            from whisper_max import WhisperMAX
            self.assertTrue(True, "MAX Graph implementation imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import MAX Graph implementation: {e}")
    
    def test_cpu_initialization(self):
        """Test CPU model initialization"""
        from whisper_cpu import WhisperCPU
        
        # Test with mock to avoid loading actual model
        with patch('whisper_cpu.whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            cpu_model = WhisperCPU(model_size="tiny")
            self.assertIsNotNone(cpu_model)
            mock_load.assert_called_once_with("tiny")
    
    def test_model_size_validation(self):
        """Test that invalid model sizes are handled"""
        from whisper_cpu import WhisperCPU
        
        with self.assertRaises(ValueError):
            WhisperCPU(model_size="invalid_size")
    
    @patch('torch.cuda.is_available')
    def test_gpu_availability_check(self, mock_cuda):
        """Test GPU availability checking"""
        from whisper_gpu import WhisperGPU
        
        # Test when CUDA is available
        mock_cuda.return_value = True
        with patch('whisper_gpu.whisper.load_model') as mock_load:
            mock_load.return_value = Mock()
            gpu_model = WhisperGPU(model_size="tiny", use_gpu=True)
            self.assertTrue(gpu_model.use_gpu)
        
        # Test when CUDA is not available
        mock_cuda.return_value = False
        with patch('whisper_gpu.whisper.load_model') as mock_load:
            mock_load.return_value = Mock()
            gpu_model = WhisperGPU(model_size="tiny", use_gpu=True)
            self.assertFalse(gpu_model.use_gpu)  # Should fall back to CPU

class TestLoggingUtilities(unittest.TestCase):
    """Test logging utilities"""
    
    def test_logger_setup(self):
        """Test logger setup"""
        from utils.logging import setup_logger, get_logger
        
        logger = setup_logger("test", level="DEBUG")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test")
        
        # Test singleton behavior
        logger2 = get_logger()
        self.assertEqual(logger, logger2)
    
    def test_benchmark_logger(self):
        """Test benchmark logger functionality"""
        from utils.logging import BenchmarkLogger, setup_logger
        
        logger = setup_logger("test_benchmark", json_output=True)
        benchmark_logger = BenchmarkLogger(logger)
        
        # Test that logging doesn't raise exceptions
        try:
            benchmark_logger.log_benchmark_result(
                implementation="test",
                model_size="tiny",
                audio_file="test.wav",
                execution_time=1.23,
                result_text="test transcription"
            )
            self.assertTrue(True, "Benchmark logging completed without error")
        except Exception as e:
            self.fail(f"Benchmark logging failed: {e}")

class TestBenchmarkRunner(unittest.TestCase):
    """Test benchmark runner functionality"""
    
    def test_benchmark_runner_init(self):
        """Test benchmark runner initialization"""
        from benchmarks.benchmark_runner import WhisperBenchmarkRunner
        
        runner = WhisperBenchmarkRunner(json_output=True, log_level="DEBUG")
        self.assertIsNotNone(runner)
        self.assertIsNotNone(runner.logger)
        self.assertIsNotNone(runner.benchmark_logger)
        self.assertEqual(runner.results, [])

if __name__ == '__main__':
    unittest.main()