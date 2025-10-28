#!/usr/bin/env python3
"""Data preparation script for Antarctic whale vocalization dataset."""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DatasetBuilder, LabelEncoder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare Antarctic whale vocalization dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to raw data directory (overrides config)'
    )
    parser.add_argument(
        '--use_merged_labels',
        action='store_true',
        help='Use merged label categories (ABZ, DDswp, 20Hz20Plus)'
    )
    
    return parser.parse_args()


def main():
    """Main data preparation pipeline."""
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.data_path:
        config['data']['raw_data_path'] = args.data_path
    
    print("="*60)
    print("Antarctic Whale Vocalization - Data Preparation")
    print("="*60)
    
    dataset_builder = DatasetBuilder(config)
    label_encoder = LabelEncoder(config)
    
    for split in ['train', 'test']:
        print(f"\nProcessing {split} split...")
        
        try:
            X, y = dataset_builder.build_dataset(split)
            
            print(f"Raw data shape: X={X.shape}, y={y.shape}")
            print(f"Unique labels: {np.unique(y)}")
            
            if args.use_merged_labels:
                print("Merging labels into broader categories...")
                y = label_encoder.merge_labels(y)
                print(f"Merged labels: {np.unique(y)}")
            
            y_encoded, label_mapping = label_encoder.encode_labels(y)
            print(f"Label mapping: {label_mapping}")
            
            dataset_builder.save_processed_data(X, y_encoded, split)
            
            print(f"\n{split.upper()} data statistics:")
            print(f"  Total samples: {len(X)}")
            print(f"  Image shape: {X[0].shape}")
            print(f"  Number of classes: {len(np.unique(y_encoded))}")
            
            unique, counts = np.unique(y_encoded, return_counts=True)
            print(f"\n  Class distribution:")
            for label_idx, count in zip(unique, counts):
                label_name = [k for k, v in label_mapping.items() if v == label_idx][0]
                print(f"    {label_name}: {count} ({count/len(y_encoded)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("Data preparation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
