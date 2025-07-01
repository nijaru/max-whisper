#!/usr/bin/env python3
"""
Structured logging utilities for max-whisper project
Provides consistent logging across all components with JSON output support
"""

import logging
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import contextlib

# Global logger instance
_logger = None

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry)

def setup_logger(name: str = "max-whisper", level: str = "INFO", json_output: bool = False) -> logging.Logger:
    """Setup structured logger for max-whisper components"""
    global _logger
    
    if _logger is not None:
        return _logger
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    _logger = logger
    return logger

def get_logger() -> logging.Logger:
    """Get the global logger instance"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger

@contextlib.contextmanager
def log_execution_time(operation: str, logger: Optional[logging.Logger] = None, **kwargs):
    """Context manager to log execution time of operations"""
    if logger is None:
        logger = get_logger()
    
    start_time = time.time()
    logger.info(f"Starting {operation}", extra={'extra_fields': {'operation': operation, **kwargs}})
    
    try:
        yield
        execution_time = time.time() - start_time
        logger.info(
            f"Completed {operation} in {execution_time:.3f}s", 
            extra={'extra_fields': {'operation': operation, 'execution_time': execution_time, **kwargs}}
        )
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            f"Failed {operation} after {execution_time:.3f}s: {e}", 
            extra={'extra_fields': {'operation': operation, 'execution_time': execution_time, 'error': str(e), **kwargs}}
        )
        raise

class BenchmarkLogger:
    """Specialized logger for benchmark results"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger()
        
    def log_benchmark_result(self, 
                           implementation: str,
                           model_size: str,
                           audio_file: str,
                           execution_time: float,
                           memory_usage: Optional[Dict[str, Any]] = None,
                           result_text: Optional[str] = None,
                           error: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Log structured benchmark result"""
        
        result = {
            'benchmark_type': 'whisper_implementation',
            'implementation': implementation,
            'model_size': model_size,
            'audio_file': audio_file,
            'execution_time': execution_time,
            'status': 'success' if error is None else 'error'
        }
        
        if memory_usage:
            result['memory_usage'] = memory_usage
        if result_text:
            result['result_length'] = len(result_text)
            result['result_preview'] = result_text[:100] + "..." if len(result_text) > 100 else result_text
        if error:
            result['error'] = error
        if metadata:
            result['metadata'] = metadata
            
        self.logger.info(
            f"Benchmark result: {implementation} ({model_size}) - {execution_time:.3f}s",
            extra={'extra_fields': result}
        )
        
    def log_system_info(self, gpu_available: bool = False, cuda_version: Optional[str] = None):
        """Log system information for benchmark context"""
        info = {
            'log_type': 'system_info',
            'gpu_available': gpu_available,
            'cuda_version': cuda_version
        }
        
        self.logger.info("System info", extra={'extra_fields': info})

def log_test_result(test_name: str, status: str, duration: float, 
                   details: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
    """Log test execution result"""
    if logger is None:
        logger = get_logger()
        
    result = {
        'log_type': 'test_result',
        'test_name': test_name,
        'status': status,
        'duration': duration
    }
    
    if details:
        result['details'] = details
        
    logger.info(f"Test {test_name}: {status} ({duration:.3f}s)", extra={'extra_fields': result})

# Export commonly used functions
__all__ = ['setup_logger', 'get_logger', 'log_execution_time', 'BenchmarkLogger', 'log_test_result']