#!/usr/bin/env python3
"""
Generate sample training data for Chronos model training.
Creates diverse synthetic time series data in GluonTS Arrow format.
"""

import numpy as np
from pathlib import Path
import argparse
from gluonts.dataset.arrow import ArrowWriter


def create_synthetic_time_series(num_series=100, length=200, output_file="./sample_training_data.arrow"):
    """Create diverse synthetic time series data in Arrow format."""
    print(f"Creating {num_series} synthetic time series...")
    
    time_series_data = []
    
    # Set an arbitrary start time (use pandas timestamp for compatibility)
    import pandas as pd
    start = pd.Timestamp("2020-01-01 00:00:00")
    
    # Create different pattern types to encourage diversity
    patterns = ['trend', 'seasonal', 'noise', 'mixed']
    series_per_pattern = num_series // len(patterns)
    
    for pattern_idx, pattern_type in enumerate(patterns):
        for i in range(series_per_pattern):
            t = np.linspace(0, 4*np.pi, length)
            
            if pattern_type == 'trend':
                # Strong trend with minimal noise
                ts = 0.5 * t + np.random.normal(0, 0.1, length)
                
            elif pattern_type == 'seasonal':
                # Strong seasonal pattern
                ts = 3 * np.sin(t) + 2 * np.sin(2*t) + np.random.normal(0, 0.2, length)
                
            elif pattern_type == 'noise':
                # Mostly noise with weak signal
                ts = 0.1 * np.sin(t) + np.random.normal(0, 1.0, length)
                
            else:  # mixed
                # Complex mixed pattern
                trend = 0.2 * t
                seasonal = 2 * np.sin(t) + np.sin(3*t)
                noise = np.random.normal(0, 0.3, length)
                ts = trend + seasonal + noise
            
            # Random offset for diversity
            offset = np.random.normal(0, 2)
            ts = ts + offset
            
            # Create GluonTS format entry
            time_series_data.append({
                "start": start,
                "target": ts.astype(np.float32)  # Ensure float32 for efficiency
            })
    
    # Add remaining series as mixed
    remaining = num_series - len(time_series_data)
    for i in range(remaining):
        t = np.linspace(0, 4*np.pi, length)
        ts = np.random.normal(0, 1, length) + 0.1 * t
        offset = np.random.normal(0, 2)
        ts = ts + offset
        
        time_series_data.append({
            "start": start,
            "target": ts.astype(np.float32)
        })
    
    # Save as Arrow format
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ArrowWriter(compression="lz4").write_to_file(
        time_series_data,
        path=output_path,
    )
    
    print(f"Created {num_series} time series in {output_path}")
    print(f"Pattern distribution: {dict(zip(patterns, [series_per_pattern] * len(patterns)))}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample training data")
    parser.add_argument("--num_series", type=int, default=100, help="Number of time series to generate")
    parser.add_argument("--length", type=int, default=200, help="Length of each time series")
    parser.add_argument("--output_file", type=str, default="./sample_training_data.arrow", help="Output arrow file")
    
    args = parser.parse_args()
    
    create_synthetic_time_series(
        num_series=args.num_series,
        length=args.length,
        output_file=args.output_file
    )