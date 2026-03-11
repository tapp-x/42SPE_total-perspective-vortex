import argparse
from preprocessing import load_subject_epochs
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from pipeline_config import build_pipeline, parse_runs

def main():
    parser = argparse.ArgumentParser(description="BCI Pipeline using Machine Learning.")
    parser.add_argument("subject", type=int, help="Subject ID (e.g., 1 for S001)")
    parser.add_argument("runs", type=str, nargs='+', help="List of runs to process (e.g., 4 8 12) or 'all'")
    parser.add_argument("--path", type=str, default=None, help="Base path to the dataset")
    parser.add_argument("--plot", action="store_true", help="Visualize raw and filtered data for the first run")
    
    args = parser.parse_args()
    
    try:
        target_runs = parse_runs(args.runs)
    except ValueError:
        print("Error : Invalid run numbers.")
        return
            
    X, y = load_subject_epochs(subject_id=args.subject, runs=target_runs, base_path=args.path, plot=args.plot)
    
    if X is not None:
        print("\n--- PREPROCESSING COMPLETE ---")
        print(f"Shape of X (3D) : {X.shape}")
        print(f"Shape of y      : {y.shape}")
        
        print("\n--- 2. CREATION OF THE PIPELINE ---")
        
        pipeline = build_pipeline()
        
        extractor = pipeline.named_steps['feature_extraction']
        X_2D = extractor.fit_transform(X)
        
        print(f"Shape of X after Fourier (2D) : {X_2D.shape}") 

if __name__ == "__main__":
    main()
