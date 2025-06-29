#!/usr/bin/env python3
"""
Modular TUI Demo - Clean Terminal Interface
Shows test boxes with progress and results for configurable implementations
"""

import argparse
import time
import subprocess
import sys
import os
import re
import platform
import psutil
from typing import Optional, Dict, List

class ModularDemo:
    """Modular TUI Demo with clean boxes"""
    
    # Available test definitions
    AVAILABLE_TESTS = {
        "cpu": {"name": "CPU Baseline", "script": "whisper_cpu", "desc": "OpenAI Whisper"},
        "gpu": {"name": "GPU Accelerated", "script": "whisper_gpu", "desc": "CUDA + PyTorch"},
        "max": {"name": "MAX Graph", "script": "whisper_max", "desc": "MAX Graph Hybrid"},
        "fast": {"name": "MAX Graph Fast", "script": "whisper_max_fast", "desc": "Ultra-Optimized"}
    }
    
    def __init__(self, model_size: str = "tiny", audio_file: str = None, tests: List[str] = None):
        self.model_size = model_size
        self.audio_file = audio_file or "audio_samples/modular_video.wav"
        
        # Configure which tests to run
        if tests is None:
            tests = ["cpu", "gpu", "max", "fast"]  # Default: all tests
        
        # Build test list
        self.tests = []
        for test_key in tests:
            if test_key in self.AVAILABLE_TESTS:
                test_def = self.AVAILABLE_TESTS[test_key].copy()
                test_def["status"] = "waiting"
                test_def["time"] = None
                self.tests.append(test_def)
        
        self.baseline_time = None
        self.hw_specs = self._detect_hardware()
    
    def _detect_hardware(self) -> Dict:
        """Detect hardware specifications"""
        specs = {}
        
        # CPU info - try to get actual model name
        try:
            specs['cpu_count'] = psutil.cpu_count()
            specs['cpu_cores'] = psutil.cpu_count(logical=False)
            
            # Try to get CPU model from /proc/cpuinfo on Linux
            specs['cpu'] = "Unknown"
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            cpu_model = line.split(':')[1].strip()
                            # Clean up common suffixes and duplicates
                            cpu_model = cpu_model.replace(' CPU', '').replace(' @ ', '@')
                            specs['cpu'] = cpu_model
                            break
            except:
                # Fallback to platform info
                specs['cpu'] = platform.processor() or platform.machine()
        except:
            specs['cpu'] = "Unknown"
            specs['cpu_count'] = "Unknown"
            specs['cpu_cores'] = "Unknown"
        
        # Memory info
        try:
            mem = psutil.virtual_memory()
            specs['memory_gb'] = round(mem.total / (1024**3), 1)
        except:
            specs['memory_gb'] = "Unknown"
        
        # GPU info (try multiple approaches)
        specs['gpu'] = "Not detected"
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                specs['gpu'] = result.stdout.strip().split('\n')[0]
        except:
            try:
                # Try lspci as fallback
                result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line or '3D controller' in line:
                            # Extract GPU name from lspci output
                            if 'NVIDIA' in line:
                                specs['gpu'] = line.split(': ')[-1] if ': ' in line else "NVIDIA GPU"
                            elif 'AMD' in line or 'Radeon' in line:
                                specs['gpu'] = line.split(': ')[-1] if ': ' in line else "AMD GPU"
                            break
            except:
                pass
        
        # Platform info
        specs['platform'] = platform.system()
        specs['arch'] = platform.machine()
        
        return specs
    
    def _format_hw_info(self) -> List[str]:
        """Format hardware info for display as multiple lines"""
        hw = self.hw_specs
        lines = []
        
        # CPU line - show model and cores/threads
        if hw.get('cpu') and hw['cpu'] not in ['Unknown', '']:
            cpu_model = hw['cpu'][:50] + "..." if len(hw['cpu']) > 50 else hw['cpu']
            cpu_info = f"CPU: {cpu_model}"
            if hw.get('cpu_cores') and hw.get('cpu_count'):
                cpu_info += f" ({hw['cpu_cores']}C/{hw['cpu_count']}T)"
            lines.append(cpu_info)
        elif hw.get('cpu_cores') and hw.get('cpu_count'):
            lines.append(f"CPU: {hw['cpu_cores']}C/{hw['cpu_count']}T")
        
        # RAM line
        if hw.get('memory_gb') != "Unknown":
            lines.append(f"RAM: {hw['memory_gb']}GB")
        
        # GPU line
        if hw.get('gpu') and hw['gpu'] != "Not detected":
            gpu_name = hw['gpu'][:50] + "..." if len(hw['gpu']) > 50 else hw['gpu']
            lines.append(f"GPU: {gpu_name}")
        
        return lines
    
    def clear_screen(self):
        """Clear terminal"""
        # Use cursor positioning to move to top and clear from there
        import sys
        sys.stdout.write('\033[2J\033[H')
        sys.stdout.flush()
    
    def draw_box(self, test: Dict, width: int = 60) -> str:
        """Draw a box for one test"""
        name = test["name"]
        desc = test["desc"]
        status = test["status"]
        test_time = test["time"]
        
        # Status symbols and colors
        if status == "waiting":
            symbol = "⏳"
            status_text = "Waiting..."
            time_text = ""
        elif status == "running":
            symbol = "🚀"
            status_text = "Running..."
            time_text = ""
        elif status == "complete" and test_time is not None:
            symbol = "✅"
            status_text = "Complete"
            time_text = f"{test_time:.2f}s"
            if self.baseline_time and test_time != self.baseline_time:
                speedup = self.baseline_time / test_time
                time_text += f" ({speedup:.1f}x)"
        else:
            symbol = "❌"
            status_text = "Failed"
            time_text = "ERROR"
        
        # Build box
        top = "┌" + "─" * (width - 2) + "┐"
        bottom = "└" + "─" * (width - 2) + "┘"
        
        # Fixed approach: build each line to exact width
        inner_width = width - 2  # Account for │ on both sides
        
        # Helper to get visual width (emojis are 2 chars wide visually)
        def visual_width(text):
            # Common emojis take 2 display characters
            emoji_count = 0
            for char in text:
                code = ord(char)
                # Common emoji ranges
                if (0x1F600 <= code <= 0x1F64F or  # Emoticons
                    0x1F300 <= code <= 0x1F5FF or  # Misc Symbols and Pictographs
                    0x1F680 <= code <= 0x1F6FF or  # Transport and Map
                    0x1F1E0 <= code <= 0x1F1FF or  # Regional indicators
                    0x2600 <= code <= 0x26FF or    # Misc symbols
                    0x2700 <= code <= 0x27BF or    # Dingbats
                    0x23E9 <= code <= 0x23FA):     # More symbols
                    emoji_count += 1
            return len(text) + emoji_count
        
        # Title line - manually pad to exact width accounting for emoji width
        title_content = f" {symbol} {name} ({self.model_size})"
        title_visual_width = visual_width(title_content)
        title_padding = " " * (inner_width - title_visual_width)
        title_line = f"│{title_content}{title_padding}│"
        
        # Description line - manually pad to exact width  
        desc_content = f"   {desc}"
        desc_padding = " " * (inner_width - len(desc_content))
        desc_line = f"│{desc_content}{desc_padding}│"
        
        # Status line - manually pad to exact width
        status_content = f"   {status_text}"
        if time_text:
            # Right align time, left align status
            total_content_len = len(status_content) + len(time_text)
            middle_padding = " " * (inner_width - total_content_len)
            status_line = f"│{status_content}{middle_padding}{time_text}│"
        else:
            # Just status, no time
            status_padding = " " * (inner_width - len(status_content))
            status_line = f"│{status_content}{status_padding}│"
        
        return f"{top}\n{title_line}\n{desc_line}\n{status_line}\n{bottom}"
    
    def render_all(self):
        """Render complete interface"""
        self.clear_screen()
        
        # Header box
        width = 60
        outer_top = "┌" + "─" * (width - 2) + "┐"
        outer_bottom = "└" + "─" * (width - 2) + "┘"
        
        print(outer_top)
        self._print_boxed_line("🎪 MAX Graph Performance Demo", width, center=True, has_emoji=True)
        self._print_boxed_line("=" * (width - 4), width, center=True)
        self._print_boxed_line("", width)  # Empty line
        
        # Hardware specs (multiple lines)
        hw_lines = self._format_hw_info()
        for line in hw_lines:
            self._print_boxed_line(line, width)
        
        # Tests and audio info
        self._print_boxed_line(f"Tests: {len(self.tests)}", width)
        
        # Audio file info (truncated to fit)
        audio_name = os.path.basename(self.audio_file)
        if len(audio_name) > 50:
            audio_name = audio_name[:47] + "..."
        self._print_boxed_line(f"Audio: {audio_name}", width)
        
        print(outer_bottom)
        print()
        
        # Test boxes with bold headers
        for i, test in enumerate(self.tests):
            # Bold header for each test
            print(f"\033[1m{test['name']} ({self.model_size.upper()})\033[0m")
            print(self.draw_box(test))
            print()
        
        # Summary if tests are done
        completed = [t for t in self.tests if t["status"] == "complete" and t["time"] is not None]
        if len(completed) > 0:
            fastest = min(completed, key=lambda x: x["time"])
            print(f"🏆 Status: {len(completed)}/{len(self.tests)} complete")
            if len(completed) == len(self.tests) and self.baseline_time:
                print(f"⚡ Fastest: {fastest['name']} - {fastest['time']:.2f}s ({self.baseline_time/fastest['time']:.1f}x speedup)")
    
    def _print_boxed_line(self, content: str, width: int, center: bool = False, has_emoji: bool = False):
        """Print a line inside the header box"""
        inner_width = width - 2  # Account for │ on both sides
        
        if center:
            # Account for emoji visual width (emojis take 2 display chars but count as 1 in len())
            if has_emoji:
                # Count emojis in the content
                emoji_count = 0
                for char in content:
                    code = ord(char)
                    if (0x1F600 <= code <= 0x1F64F or  # Emoticons
                        0x1F300 <= code <= 0x1F5FF or  # Misc Symbols and Pictographs
                        0x1F680 <= code <= 0x1F6FF or  # Transport and Map
                        0x1F1E0 <= code <= 0x1F1FF or  # Regional indicators
                        0x2600 <= code <= 0x26FF or    # Misc symbols
                        0x2700 <= code <= 0x27BF or    # Dingbats
                        0x23E9 <= code <= 0x23FA):     # More symbols
                        emoji_count += 1
                
                # Adjust padding calculation for emoji visual width
                visual_width = len(content) + emoji_count
                padding_total = inner_width - visual_width
            else:
                padding_total = inner_width - len(content)
            
            left_pad = padding_total // 2
            right_pad = padding_total - left_pad
            line = f"│{' ' * left_pad}{content}{' ' * right_pad}│"
        else:
            padding = " " * (inner_width - len(content))
            line = f"│ {content}{padding[1:]}│"  # Start with space, adjust padding
        
        print(line)
    
    def run_test(self, test_index: int):
        """Run a single test"""
        test = self.tests[test_index]
        
        try:
            # Mark as running
            test["status"] = "running"
            self.render_all()
            
            # Build command
            cmd = [
                "pixi", "run", "-e", "benchmark", "python",
                f"src/model/{test['script']}.py",
                "--model-size", self.model_size
            ]
            
            if self.audio_file:
                cmd.extend(["--audio-file", self.audio_file])
            
            # Run test
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            end_time = time.time()
            
            if result.returncode == 0:
                # Try to extract timing from output, fallback to measured time
                extracted_time = self.extract_timing(result.stdout)
                final_time = extracted_time if extracted_time else (end_time - start_time)
                
                test["status"] = "complete"
                test["time"] = final_time
                
                # Set baseline from CPU test
                if test["script"] == "whisper_cpu":
                    self.baseline_time = final_time
            else:
                test["status"] = "failed"
                test["time"] = None
            
            self.render_all()
            
        except subprocess.TimeoutExpired:
            test["status"] = "failed"
            test["time"] = None
            self.render_all()
        except Exception as e:
            test["status"] = "failed"
            test["time"] = None
            self.render_all()
    
    def extract_timing(self, output: str) -> Optional[float]:
        """Extract timing from output"""
        lines = output.split('\n')
        for line in lines:
            if "Total" in line and ("ms" in line or "s" in line):
                # Look for patterns like "1234.5ms" or "1.23s"
                time_match = re.search(r'(\d+\.?\d*)(ms|s)', line)
                if time_match:
                    time_val = float(time_match.group(1))
                    unit = time_match.group(2)
                    
                    # Convert to seconds
                    if unit == "ms":
                        time_val = time_val / 1000
                    
                    return time_val
        return None
    
    def run_all_tests(self):
        """Run all tests sequentially"""
        self.render_all()
        
        for i in range(len(self.tests)):
            self.run_test(i)
            time.sleep(0.5)  # Brief pause between tests
        
        # Final render
        self.render_all()
        
        # Final summary
        print("\n🎯 Demo Complete!")
        completed = [t for t in self.tests if t["status"] == "complete" and t["time"] is not None]
        if completed:
            print(f"✅ {len(completed)}/{len(self.tests)} implementations successful")
        else:
            print("❌ No implementations completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Modular TUI Demo")
    
    # Positional arguments
    parser.add_argument('model_size', nargs='?', default='small', choices=['tiny', 'small', 'base'],
                       help='Model size to test (default: small)')
    parser.add_argument('audio_file', nargs='?',
                       help='Audio file to use (default: audio_samples/modular_video.wav)')
    
    # Optional arguments
    parser.add_argument('--tests', nargs='+', choices=['cpu', 'gpu', 'max', 'fast'],
                       help='Which tests to run (default: all)')
    parser.add_argument('--demo-type', choices=['quick', 'judge', 'full'], default='full',
                       help='Predefined demo configurations')
    
    args = parser.parse_args()
    
    # Handle predefined demo types
    if args.demo_type == 'quick':
        tests = ['cpu', 'gpu']  # Quick demo - just CPU and GPU
    elif args.demo_type == 'judge':
        tests = ['cpu', 'gpu', 'max', 'fast']  # Full demo for judges
    else:
        tests = args.tests  # Use specified tests or default to all
    
    # Create and run demo
    demo = ModularDemo(model_size=args.model_size, audio_file=args.audio_file, tests=tests)
    
    try:
        demo.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()