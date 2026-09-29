import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from src.probes import ProbeTrainer

def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark")
    args = parser.parse_args()
    
    print("Running benchmark...")
    # trainer = ProbeTrainer()
    # ... benchmark logic ...

if __name__ == "__main__":
    main()
