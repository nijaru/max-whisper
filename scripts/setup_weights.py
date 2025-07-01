#!/usr/bin/env python3
"""
Weight Setup Script for MAX Graph Whisper
Extracts and caches pretrained Whisper weights for MAX Graph usage
"""

import os
import sys
from pathlib import Path

# Add max-whisper to path for imports
sys.path.append(str(Path(__file__).parent.parent / "max-whisper"))

try:
    from whisper_weight_extractor import WhisperWeightExtractor
    print("✅ Weight extractor available")
except ImportError as e:
    print(f"❌ Cannot import weight extractor: {e}")
    print("Make sure you're running from the project root with dependencies installed")
    sys.exit(1)


def setup_weights(model_size="tiny"):
    """Extract and cache weights for the specified model size"""
    print(f"🔧 Setting up weights for Whisper {model_size} model...")
    
    try:
        # Create weight extractor
        extractor = WhisperWeightExtractor(model_size)
        
        # Extract weights from pretrained model
        weights = extractor.extract_openai_whisper_weights()
        
        if not weights:
            print(f"❌ Failed to extract weights for {model_size} model")
            return False
        
        # Save weights to cache files
        weights_file = f"whisper_{model_size}_weights.npz"
        if extractor.save_weights(weights_file):
            print(f"✅ Weights cached to {weights_file}")
            print(f"📊 Extracted {len(weights)} weight tensors")
            
            # Print weight summary
            summary = extractor.get_weight_summary()
            total_params = summary.get('TOTAL_PARAMETERS', ('', 0))[1]
            print(f"🔢 Total parameters: {total_params:,}")
            
            return True
        else:
            print(f"❌ Failed to save weights")
            return False
            
    except Exception as e:
        print(f"❌ Weight setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function"""
    print("🎯 MAX Graph Whisper Weight Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("max-whisper/whisper_weight_extractor.py"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Setup weights for available model sizes
    model_sizes = ["tiny"]  # Start with tiny, can add others
    
    success_count = 0
    for model_size in model_sizes:
        print(f"\n📦 Processing {model_size} model:")
        if setup_weights(model_size):
            success_count += 1
        print()
    
    print("=" * 50)
    if success_count == len(model_sizes):
        print(f"🎉 Weight setup complete! Successfully processed {success_count} models")
        print("\n🚀 You can now run the MAX Graph Whisper implementations:")
        print("   python max-whisper/whisper_max.py --model-size tiny")
        print("   make benchmark")
    else:
        print(f"⚠️ Partial success: {success_count}/{len(model_sizes)} models processed")
        print("Some MAX Graph implementations may not work without cached weights")


if __name__ == "__main__":
    main()