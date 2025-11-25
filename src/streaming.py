# Streaming job placeholder
"""
Streaming Data Simulator
Converts static CSV into real-time stream
"""

import time
import random
from datetime import datetime
import pandas as pd
import os


class TransactionSimulator:
    """Simulates real-time shopping transactions"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} transactions from {csv_path}")
    
    def stream_to_directory(self, output_dir='data/streaming_input', 
                           batch_size=50, num_batches=20, delay=2):
        """
        Create streaming batches as CSV files
        
        Args:
            output_dir: Where to save batch files
            batch_size: Transactions per batch
            num_batches: Total batches to create
            delay: Seconds between batches
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"STREAMING SIMULATION STARTED")
        print(f"{'='*70}")
        print(f"  Output: {output_dir}")
        print(f"  Batch size: {batch_size} transactions")
        print(f"  Total batches: {num_batches}")
        print(f"  Delay: {delay} seconds")
        print(f"  Total duration: ~{num_batches * delay} seconds")
        print(f"{'='*70}\n")
        
        for batch_num in range(num_batches):
            # Random sample from data
            batch_df = self.df.sample(n=batch_size, replace=True).copy()
            
            # Add timestamp (event time)
            batch_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save batch
            filename = f"batch_{batch_num:04d}_{int(time.time())}.csv"
            filepath = os.path.join(output_dir, filename)
            batch_df.to_csv(filepath, index=False)
            
            print(f"✓ Batch {batch_num + 1}/{num_batches}: {filename} ({batch_size} transactions)")
            
            # Wait before next batch
            if batch_num < num_batches - 1:
                time.sleep(delay)
        
        print(f"\n{'='*70}")
        print(f"✓ STREAMING SIMULATION COMPLETED")
        print(f"  Created {num_batches} batches")
        print(f"  Total transactions: {num_batches * batch_size}")
        print(f"{'='*70}\n")


def main():
    """Run simulator"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║            Shopping Trends - Streaming Simulator              ║
    ║            ITCS 6190 Big Data Analytics                       ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    simulator = TransactionSimulator('data/shopping.csv')
    
    print("\nConfiguration:")
    print("  1. Quick Demo (10 batches, 30 trans each, 2s delay) - 20 seconds")
    print("  2. Standard (20 batches, 50 trans each, 2s delay) - 40 seconds")
    print("  3. Extended (30 batches, 50 trans each, 3s delay) - 90 seconds")
    print("  4. Custom")
    
    choice = input("\nSelect configuration (1-4): ")
    
    if choice == '1':
        simulator.stream_to_directory(batch_size=30, num_batches=10, delay=2)
    elif choice == '2':
        simulator.stream_to_directory(batch_size=50, num_batches=20, delay=2)
    elif choice == '3':
        simulator.stream_to_directory(batch_size=50, num_batches=30, delay=3)
    elif choice == '4':
        batch_size = int(input("Batch size: "))
        num_batches = int(input("Number of batches: "))
        delay = int(input("Delay (seconds): "))
        simulator.stream_to_directory(batch_size=batch_size, num_batches=num_batches, delay=delay)
    else:
        print("Invalid choice, using standard configuration")
        simulator.stream_to_directory()


if __name__ == "__main__":
    main()