import argparse
from src.hook_utils import load_model_bundle, PolygraphFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract concept vectors")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()
    
    print(f"Extracting vectors for {args.model}...")
    # Initialize and run extraction...

if __name__ == "__main__":
    main()
