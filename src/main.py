import argparse
from preprocessing import preprocessing
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from features import PowerBandExtractor

def main():
    parser = argparse.ArgumentParser(description="BCI Pipeline using Machine Learning.")
    parser.add_argument("subject", type=int, help="Subject ID (e.g., 1 for S001)")
    parser.add_argument("runs", type=str, nargs='+', help="List of runs to process (e.g., 4 8 12) or 'all'")
    parser.add_argument("--path", type=str, default=None, help="Base path to the dataset")
    parser.add_argument("--plot", action="store_true", help="Visualize raw and filtered data for the first run")
    
    args = parser.parse_args()
    
    if 'all' in [r.lower() for r in args.runs]:
        target_runs = list(range(1, 15))
    else:
        try:
            target_runs = [int(r) for r in args.runs]
        except ValueError:
            print("Error : Invalid run numbers.")
            return
            
    X, y = preprocessing(subject_id=args.subject, runs=target_runs, base_path=args.path, plot=args.plot)
    
    if X is not None:
        print("\n--- PREPROCESSING COMPLETE ---")
        print(f"Shape of X (3D) : {X.shape}")
        print(f"Shape of y      : {y.shape}")
        
        print("\n--- 2. CREATION OF THE PIPELINE ---")
        
        pipeline = Pipeline([
            ('feature_extraction', PowerBandExtractor()),
            # ('dimensionality_reduction', ...), #Todo: : add my CSP here
            ('classifier', SVC(kernel='linear'))
        ])
        
        extractor = pipeline.named_steps['feature_extraction']
        X_2D = extractor.fit_transform(X)
        
        print(f"Shape of X after Fourier (2D) : {X_2D.shape}") 

if __name__ == "__main__":
    main()