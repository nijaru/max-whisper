#!/usr/bin/env python3
"""
Test Segmented Audio Processing for Full-Length Transcription
Process long audio files in segments to achieve full 1800+ character transcription
"""

import sys
import time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "max-whisper"))

from whisper_max import WhisperMAX
import whisper


def segment_audio(audio, segment_length_sec=30, overlap_sec=2, sample_rate=16000):
    """
    Segment long audio into overlapping chunks for processing
    
    Args:
        audio: Audio array from whisper.load_audio()
        segment_length_sec: Length of each segment in seconds
        overlap_sec: Overlap between segments in seconds  
        sample_rate: Audio sample rate (Whisper uses 16kHz)
    
    Returns:
        List of audio segments with metadata
    """
    samples_per_segment = segment_length_sec * sample_rate
    overlap_samples = overlap_sec * sample_rate
    
    segments = []
    start_sample = 0
    segment_idx = 0
    
    while start_sample < len(audio):
        end_sample = min(start_sample + samples_per_segment, len(audio))
        segment_audio = audio[start_sample:end_sample]
        
        # Only process segments with sufficient audio (>5 seconds)
        if len(segment_audio) >= 5 * sample_rate:
            segments.append({
                'audio': segment_audio,
                'start_time': start_sample / sample_rate,
                'end_time': end_sample / sample_rate,
                'duration': len(segment_audio) / sample_rate,
                'segment_idx': segment_idx,
                'overlap_start': segment_idx > 0,
                'overlap_end': end_sample < len(audio)
            })
        
        # Move to next segment with overlap
        start_sample = end_sample - overlap_samples
        segment_idx += 1
        
        # Break if we've covered the full audio
        if end_sample >= len(audio):
            break
    
    return segments


def process_audio_segments(encoder, segments, merge_overlap=True):
    """
    Process each audio segment and merge results
    
    Args:
        encoder: WhisperMAX encoder instance
        segments: List of audio segments from segment_audio()
        merge_overlap: Whether to merge overlapping content
    
    Returns:
        Combined transcription and performance metrics
    """
    segment_results = []
    total_encoder_time = 0
    total_decoder_time = 0
    
    print(f"🎯 Processing {len(segments)} audio segments...")
    
    for i, segment in enumerate(segments):
        print(f"\n📊 Segment {i+1}/{len(segments)}: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s ({segment['duration']:.1f}s)")
        
        # Process segment
        start_time = time.time()
        
        # Convert audio to mel spectrogram
        mel_features = whisper.log_mel_spectrogram(segment['audio'])
        mel_np = mel_features.cpu().numpy()
        
        # Run MAX Graph encoder
        encoder_start = time.time()
        encoder_features = encoder._encode_with_max_graph(mel_np)
        encoder_time = time.time() - encoder_start
        
        # Run PyTorch decoder
        decoder_start = time.time()
        transcript = encoder._decode_with_pytorch(encoder_features)
        decoder_time = time.time() - decoder_start
        
        segment_time = time.time() - start_time
        
        segment_results.append({
            'segment_idx': i,
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'transcript': transcript,
            'encoder_time': encoder_time,
            'decoder_time': decoder_time,
            'total_time': segment_time,
            'char_count': len(transcript)
        })
        
        total_encoder_time += encoder_time
        total_decoder_time += decoder_time
        
        print(f"   ⚡ Encoder: {encoder_time:.3f}s")
        print(f"   📝 Decoder: {decoder_time:.3f}s") 
        print(f"   📏 Output: {len(transcript)} chars")
        print(f"   📄 Text: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
    
    # Merge segments
    if merge_overlap:
        merged_transcript = merge_overlapping_segments(segment_results)
    else:
        merged_transcript = " ".join([result['transcript'] for result in segment_results])
    
    return {
        'segments': segment_results,
        'merged_transcript': merged_transcript,
        'total_char_count': len(merged_transcript),
        'total_encoder_time': total_encoder_time,
        'total_decoder_time': total_decoder_time,
        'avg_encoder_time': total_encoder_time / len(segments),
        'avg_decoder_time': total_decoder_time / len(segments),
        'num_segments': len(segments)
    }


def merge_overlapping_segments(segment_results, overlap_threshold=0.7):
    """
    Merge transcripts from overlapping segments by removing duplicate content
    
    Args:
        segment_results: List of segment processing results
        overlap_threshold: Similarity threshold for detecting overlaps
    
    Returns:
        Merged transcript string
    """
    if not segment_results:
        return ""
    
    merged_parts = []
    
    for i, result in enumerate(segment_results):
        transcript = result['transcript'].strip()
        
        if i == 0:
            # First segment - add completely
            merged_parts.append(transcript)
        else:
            # Find overlap with previous segment
            prev_transcript = merged_parts[-1] if merged_parts else ""
            
            # Simple overlap detection: check if start of current matches end of previous
            overlap_found = False
            min_overlap_len = 20  # Minimum characters to consider overlap
            
            for overlap_len in range(min_overlap_len, min(len(transcript), len(prev_transcript)) // 2):
                current_start = transcript[:overlap_len]
                prev_end = prev_transcript[-overlap_len:]
                
                # Check similarity (simple approach)
                if current_start.lower().replace(" ", "") == prev_end.lower().replace(" ", ""):
                    # Found overlap - add only the non-overlapping part
                    non_overlapping_part = transcript[overlap_len:].strip()
                    if non_overlapping_part:
                        merged_parts.append(" " + non_overlapping_part)
                    overlap_found = True
                    break
            
            if not overlap_found:
                # No overlap found - add full transcript with space
                merged_parts.append(" " + transcript)
    
    return "".join(merged_parts)


def test_segmented_processing():
    """Test segmented audio processing for full-length transcription"""
    print("🚀 Testing Segmented Audio Processing for Full-Length Transcription")
    print("=" * 80)
    
    try:
        # Initialize encoder
        print("🔧 Step 1: Initializing MAX Graph encoder...")
        encoder = WhisperMAX(model_size='tiny', full_max_graph=False)
        
        # Load full audio file
        print("🎵 Step 2: Loading full audio file...")
        audio = whisper.load_audio("audio_samples/modular_video.wav")
        audio_duration = len(audio) / 16000  # Whisper uses 16kHz
        
        print(f"   📊 Audio duration: {audio_duration:.1f} seconds ({len(audio):,} samples)")
        
        # Test CPU baseline for comparison
        print("🔍 Step 3: Getting CPU baseline for comparison...")
        baseline_start = time.time()
        
        # Run CPU Whisper for quality reference
        cpu_model = whisper.load_model("tiny")
        cpu_result = whisper.transcribe(cpu_model, audio)
        cpu_transcript = cpu_result["text"].strip()
        cpu_time = time.time() - baseline_start
        
        print(f"   📏 CPU baseline: {len(cpu_transcript)} characters in {cpu_time:.2f}s")
        print(f"   📄 CPU text: '{cpu_transcript[:100]}{'...' if len(cpu_transcript) > 100 else ''}'")
        
        # Segment audio 
        print("✂️ Step 4: Segmenting audio...")
        segments = segment_audio(audio, segment_length_sec=30, overlap_sec=3)
        
        print(f"   📊 Created {len(segments)} segments")
        for i, seg in enumerate(segments[:3]):  # Show first 3 segments
            print(f"   - Segment {i+1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s ({seg['duration']:.1f}s)")
        if len(segments) > 3:
            print(f"   - ... and {len(segments)-3} more segments")
        
        # Process segments
        print("🎯 Step 5: Processing segments with hybrid approach...")
        processing_start = time.time()
        
        results = process_audio_segments(encoder, segments)
        total_processing_time = time.time() - processing_start
        
        # Results analysis
        print("\n" + "="*80)
        print("📊 SEGMENTED PROCESSING RESULTS")
        print("="*80)
        
        print(f"⚡ Performance Metrics:")
        print(f"   🔢 Total encoder time: {results['total_encoder_time']:.3f}s")
        print(f"   🔤 Total decoder time: {results['total_decoder_time']:.3f}s")
        print(f"   ⏱️ Total processing: {total_processing_time:.3f}s")
        print(f"   📊 Average per segment: {total_processing_time/results['num_segments']:.3f}s")
        
        print(f"\n📝 Content Metrics:")
        print(f"   📏 Merged transcript: {results['total_char_count']} characters")
        print(f"   📊 CPU baseline: {len(cpu_transcript)} characters")
        print(f"   📈 Length ratio: {results['total_char_count']/len(cpu_transcript):.2f}x")
        
        print(f"\n🎯 Quality Analysis:")
        # Show first 200 chars of merged transcript
        merged_preview = results['merged_transcript'][:200]
        cpu_preview = cpu_transcript[:200] 
        
        print(f"   📄 Hybrid segments: '{merged_preview}{'...' if len(results['merged_transcript']) > 200 else ''}'")
        print(f"   📄 CPU baseline:   '{cpu_preview}{'...' if len(cpu_transcript) > 200 else ''}'")
        
        # Success criteria
        print(f"\n✅ SUCCESS CRITERIA:")
        length_success = results['total_char_count'] >= len(cpu_transcript) * 0.8  # 80% of baseline
        speed_success = total_processing_time < cpu_time * 2  # Within 2x of CPU time
        content_success = len(results['merged_transcript']) > 1000  # Substantial content
        
        print(f"   {'✅' if length_success else '❌'} Length: {'SUCCESS' if length_success else 'INSUFFICIENT'} ({results['total_char_count']} vs {len(cpu_transcript)} target)")
        print(f"   {'✅' if speed_success else '❌'} Speed: {'SUCCESS' if speed_success else 'SLOW'} ({total_processing_time:.2f}s vs {cpu_time:.2f}s baseline)")
        print(f"   {'✅' if content_success else '❌'} Content: {'SUCCESS' if content_success else 'INSUFFICIENT'} ({len(results['merged_transcript'])} chars)")
        
        overall_success = length_success and speed_success and content_success
        
        return {
            "success": overall_success,
            "results": results,
            "cpu_baseline": {"text": cpu_transcript, "time": cpu_time},
            "metrics": {
                "length_ratio": results['total_char_count']/len(cpu_transcript),
                "speed_ratio": total_processing_time/cpu_time,
                "total_chars": results['total_char_count'],
                "processing_time": total_processing_time
            }
        }
        
    except Exception as e:
        print(f"❌ Segmented processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = test_segmented_processing()
    
    if result and result.get("success"):
        print(f"\n🎉 SEGMENTED PROCESSING: SUCCESS!")
        print(f"   🎯 Achieved {result['metrics']['total_chars']} character transcription")
        print(f"   ⚡ Speed: {result['metrics']['speed_ratio']:.2f}x vs CPU baseline")
        print(f"   📈 Length: {result['metrics']['length_ratio']:.2f}x vs CPU baseline")
    else:
        print(f"\n❌ SEGMENTED PROCESSING: FAILED")
        if result and "error" in result:
            print(f"   🐛 Error: {result['error']}")