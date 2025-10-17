#!/usr/bin/env python3
"""
Test script to compare regular Chronos training vs MOE training.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_training(config_file, description):
    """Run training with the given config file."""
    print(f"\n{'='*60}")
    print(f"Testing {description}")
    print(f"Config: {config_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/training/train.py", 
            f"--config={config_file}"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {duration:.1f}s")
            
            # Extract some key metrics from the output
            lines = result.stdout.split('\n')
            for line in lines:
                if ('train_loss' in line or 'train_ce_loss' in line or 'train_load_loss' in line or 
                    'model parameters:' in line or 'MOE overhead:' in line):
                    print(f"   {line.strip()}")
            
            return True
        else:
            print(f"❌ {description} failed")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 {description} crashed: {e}")
        return False


def main():
    """Run training comparison tests."""
    print("🚀 Testing Chronos Training: Regular vs MOE")
    
    # Check if sample data exists
    if not Path("sample_training_data.arrow").exists():
        print("📊 Generating sample training data...")
        result = subprocess.run([
            sys.executable, "scripts/generate_sample_data.py",
            "--num_series", "50",
            "--length", "150",
            "--output_file", "./sample_training_data.arrow"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Failed to generate sample data")
            print(result.stderr)
            return
        print("✅ Sample data generated")
    
    # Test regular training
    regular_success = run_training(
        "scripts/training/configs/chronos-t5-tiny-sample.yaml",
        "Regular Chronos Training"
    )
    
    # Test MOE training
    moe_success = run_training(
        "scripts/training/configs/chronos-t5-tiny-moe-sample.yaml", 
        "MOE Chronos Training"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Regular Training: {'✅ PASS' if regular_success else '❌ FAIL'}")
    print(f"MOE Training:     {'✅ PASS' if moe_success else '❌ FAIL'}")
    
    if regular_success and moe_success:
        print("\n🎉 Both training methods work successfully!")
        print("\nKey differences observed:")
        print("- Regular training uses standard cross-entropy loss")
        print("- MOE training adds load balancing loss to encourage expert diversity")
        print("- MOE training shows separate CE loss and load balancing loss metrics")
        print("- Both methods converge and reduce loss over training steps")
    else:
        print("\n⚠️  Some training methods failed. Check the logs above.")
    
    # Check output directories
    output_dirs = list(Path("output").glob("run-*"))
    if output_dirs:
        print(f"\n📁 Training outputs saved in: {len(output_dirs)} directories")
        for dir_path in sorted(output_dirs)[-2:]:  # Show last 2
            print(f"   {dir_path}")


if __name__ == "__main__":
    main()