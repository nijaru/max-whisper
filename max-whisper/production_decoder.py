"""
Production-Ready MAX Graph Decoder with Enterprise Error Handling

Implements robust error recovery, monitoring, and production safeguards:
- Automatic retry with exponential backoff
- Graceful degradation on errors
- Performance monitoring hooks
- Memory overflow protection
- Token generation timeout handling
"""

import time
import logging
import warnings
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from functools import wraps

import numpy as np
from max import engine
from max.dtype import DType
from max.graph import Graph, ops, Symbol, TensorType

from .max_graph_decoder import MAX_TextDecoder

# Configure production logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionError(Exception):
    """Base exception for production decoder errors"""
    pass

class TokenGenerationTimeout(ProductionError):
    """Raised when token generation exceeds timeout"""
    pass

class MemoryOverflowError(ProductionError):
    """Raised when memory usage exceeds limits"""
    pass

class ProductionDecoder(MAX_TextDecoder):
    """Production-ready decoder with enterprise error handling"""
    
    def __init__(self, 
                 model_state_dict,
                 tokenizer,
                 device_ref,
                 driver_device,
                 max_seq_len: int = 448,
                 retry_attempts: int = 3,
                 timeout_per_token: float = 0.5,
                 memory_limit_mb: float = 1024,
                 enable_monitoring: bool = True):
        """
        Initialize production decoder with safeguards.
        
        Args:
            retry_attempts: Number of retry attempts on failure
            timeout_per_token: Maximum time per token generation (seconds)
            memory_limit_mb: Maximum memory usage in MB
            enable_monitoring: Enable performance monitoring
        """
        super().__init__(model_state_dict, tokenizer, device_ref, driver_device, max_seq_len)
        
        self.retry_attempts = retry_attempts
        self.timeout_per_token = timeout_per_token
        self.memory_limit_mb = memory_limit_mb
        self.enable_monitoring = enable_monitoring
        
        # Performance metrics
        self.metrics = {
            'total_tokens_generated': 0,
            'total_errors': 0,
            'retry_count': 0,
            'timeout_count': 0,
            'memory_overflow_count': 0,
            'generation_times': []
        }
        
        # Error recovery state
        self.last_successful_state = None
        
    @contextmanager
    def error_recovery(self, operation_name: str):
        """Context manager for error recovery"""
        try:
            yield
        except Exception as e:
            logger.error(f"Error in {operation_name}: {str(e)}")
            self.metrics['total_errors'] += 1
            
            # Attempt recovery
            if hasattr(self, 'reset_cache'):
                logger.info("Attempting cache reset for recovery...")
                self.reset_cache()
            
            raise ProductionError(f"Operation {operation_name} failed: {str(e)}") from e
    
    def retry_with_backoff(self, func):
        """Decorator for retry with exponential backoff"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.retry_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    self.metrics['retry_count'] += 1
                    
                    if attempt < self.retry_attempts - 1:
                        backoff_time = 2 ** attempt * 0.1  # 0.1s, 0.2s, 0.4s...
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {backoff_time}s...")
                        time.sleep(backoff_time)
                    else:
                        logger.error(f"All {self.retry_attempts} attempts failed")
                        
            raise last_exception
        return wrapper
    
    def check_memory_usage(self):
        """Check if memory usage is within limits"""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            current_mb = usage.ru_maxrss / 1024  # KB to MB on Linux
            
            if current_mb > self.memory_limit_mb:
                self.metrics['memory_overflow_count'] += 1
                raise MemoryOverflowError(
                    f"Memory usage ({current_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)"
                )
        except ImportError:
            pass  # Skip memory check if resource module not available
    
    def generate_with_timeout(self, 
                            tokens: np.ndarray,
                            encoder_output: np.ndarray,
                            temperature: float = 0.0,
                            max_tokens: Optional[int] = None) -> Tuple[List[int], Dict[str, Any]]:
        """
        Generate tokens with timeout protection.
        
        Returns:
            Tuple of (generated_tokens, generation_stats)
        """
        if max_tokens is None:
            max_tokens = self.max_seq_len
            
        generated_tokens = []
        generation_stats = {
            'tokens_per_second': [],
            'cumulative_time': 0,
            'errors_recovered': 0
        }
        
        start_time = time.time()
        
        for i in range(max_tokens):
            try:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.timeout_per_token * (i + 1):
                    self.metrics['timeout_count'] += 1
                    raise TokenGenerationTimeout(
                        f"Token generation timeout after {i} tokens ({elapsed:.1f}s)"
                    )
                
                # Check memory
                if i % 10 == 0:  # Check every 10 tokens
                    self.check_memory_usage()
                
                # Generate next token with retry
                token_start = time.time()
                
                with self.error_recovery(f"token_generation_{i}"):
                    next_token = self._generate_single_token(
                        tokens, encoder_output, temperature
                    )
                
                token_time = time.time() - token_start
                
                # Update metrics
                generated_tokens.append(next_token)
                generation_stats['tokens_per_second'].append(1.0 / token_time)
                
                # Check for end token
                if next_token == self.tokenizer.eot:
                    break
                    
                # Update tokens for next iteration
                tokens = np.append(tokens, next_token)
                
            except (TokenGenerationTimeout, MemoryOverflowError):
                logger.warning(f"Generation stopped at token {i} due to limits")
                break
            except Exception as e:
                logger.error(f"Error at token {i}: {str(e)}")
                generation_stats['errors_recovered'] += 1
                
                # Try to recover with last good state
                if self.last_successful_state is not None:
                    logger.info("Attempting recovery from last successful state...")
                    tokens = self.last_successful_state
                else:
                    break
        
        # Final stats
        generation_stats['cumulative_time'] = time.time() - start_time
        generation_stats['total_tokens'] = len(generated_tokens)
        
        if self.enable_monitoring:
            self.update_metrics(generation_stats)
            
        return generated_tokens, generation_stats
    
    @retry_with_backoff
    def _generate_single_token(self, 
                              tokens: np.ndarray,
                              encoder_output: np.ndarray,
                              temperature: float) -> int:
        """Generate a single token with error handling"""
        # Save state before generation
        self.last_successful_state = tokens.copy()
        
        # Get logits from parent class
        logits = self.forward(tokens, encoder_output)
        
        # Apply temperature and sampling
        if temperature > 0:
            # Simplified sampling for robustness
            probs = self._apply_temperature_sampling(logits[-1], temperature)
            next_token = np.random.choice(len(probs), p=probs)
        else:
            next_token = np.argmax(logits[-1])
            
        return int(next_token)
    
    def _apply_temperature_sampling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling with numerical stability"""
        # Subtract max for numerical stability
        logits = logits - np.max(logits)
        
        # Apply temperature
        scaled_logits = logits / temperature
        
        # Compute probabilities
        exp_logits = np.exp(scaled_logits)
        probs = exp_logits / np.sum(exp_logits)
        
        # Handle numerical issues
        if np.any(np.isnan(probs)):
            logger.warning("NaN in probabilities, using uniform distribution")
            probs = np.ones_like(probs) / len(probs)
            
        return probs
    
    def update_metrics(self, generation_stats: Dict[str, Any]):
        """Update performance metrics"""
        self.metrics['total_tokens_generated'] += generation_stats['total_tokens']
        self.metrics['generation_times'].append(generation_stats['cumulative_time'])
        
        # Log performance summary
        if len(self.metrics['generation_times']) % 10 == 0:
            avg_time = np.mean(self.metrics['generation_times'][-10:])
            avg_tokens = self.metrics['total_tokens_generated'] / len(self.metrics['generation_times'])
            
            logger.info(f"Performance Summary: "
                       f"Avg time: {avg_time:.2f}s, "
                       f"Avg tokens: {avg_tokens:.1f}, "
                       f"Errors: {self.metrics['total_errors']}, "
                       f"Retries: {self.metrics['retry_count']}")
    
    def reset_cache(self):
        """Reset KV cache to recover from errors"""
        if hasattr(self, 'k_cache') and hasattr(self, 'v_cache'):
            # Reset to zeros
            for layer_idx in range(len(self.k_cache)):
                self.k_cache[layer_idx].fill(0)
                self.v_cache[layer_idx].fill(0)
            logger.info("KV cache reset completed")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        status = {
            'healthy': True,
            'metrics': self.metrics,
            'warnings': [],
            'cache_status': 'unknown'
        }
        
        # Check error rate
        if self.metrics['total_tokens_generated'] > 0:
            error_rate = self.metrics['total_errors'] / self.metrics['total_tokens_generated']
            if error_rate > 0.1:  # More than 10% errors
                status['warnings'].append(f"High error rate: {error_rate:.2%}")
                status['healthy'] = False
        
        # Check timeout rate
        if self.metrics['timeout_count'] > 5:
            status['warnings'].append(f"High timeout count: {self.metrics['timeout_count']}")
            
        # Check cache
        try:
            if hasattr(self, 'k_cache'):
                status['cache_status'] = 'initialized'
        except:
            status['cache_status'] = 'error'
            status['healthy'] = False
            
        return status
    
    def shutdown(self):
        """Graceful shutdown with cleanup"""
        logger.info("Shutting down production decoder...")
        
        # Log final metrics
        logger.info(f"Final metrics: {self.metrics}")
        
        # Clear caches
        if hasattr(self, 'reset_cache'):
            self.reset_cache()
            
        # Clear session if exists
        if hasattr(self, 'session'):
            del self.session
            
        logger.info("Shutdown complete")


def create_production_decoder(model_state_dict,
                            tokenizer,
                            device_ref,
                            driver_device,
                            config: Optional[Dict[str, Any]] = None) -> ProductionDecoder:
    """
    Factory function to create production decoder with config.
    
    Args:
        config: Optional configuration dict with keys:
            - max_seq_len: Maximum sequence length (default: 448)
            - retry_attempts: Number of retries (default: 3)
            - timeout_per_token: Timeout per token in seconds (default: 0.5)
            - memory_limit_mb: Memory limit in MB (default: 1024)
            - enable_monitoring: Enable monitoring (default: True)
    """
    default_config = {
        'max_seq_len': 448,
        'retry_attempts': 3,
        'timeout_per_token': 0.5,
        'memory_limit_mb': 1024,
        'enable_monitoring': True
    }
    
    if config:
        default_config.update(config)
        
    return ProductionDecoder(
        model_state_dict=model_state_dict,
        tokenizer=tokenizer,
        device_ref=device_ref,
        driver_device=driver_device,
        **default_config
    )