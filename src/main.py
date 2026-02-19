import argparse
from preprocessing import preprocessing
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="BCI Pipeline using Machine Learning.")
    parser.add_argument("subject", type=int, help="Subject ID (e.g., 1 for S001)")
    parser.add_argument("runs", type=str, nargs='+', help="List of runs to process (e.g., 4 8 12) or 'all'")
    
    parser.add_argument("--path", type=str, default=None, help="Base path to the dataset")
    parser.add_argument("--plot", action="store_true", help="Visualize raw and filtered data for the first run")
    
    args = parser.parse_args()
    
    if 'all' in [r.lower() for r in args.runs]:
        target_runs = list(range(1, 15))
        print("Option 'all' détectée : traitement des 14 runs.")
    else:
        try:
            target_runs = [int(r) for r in args.runs]
        except ValueError:
            print("Erreur : Les runs doivent être des nombres entiers ou 'all'.")
            return
            
    X, y = preprocessing(subject_id=args.subject, runs=target_runs, base_path=args.path, plot=args.plot)
    
    if X is not None:
        print("\n--- PREPROCESSING COMPLETE ---")
        print(f"Final X shape : {X.shape}")
        print(f"Final y shape : {y.shape}")
        

if __name__ == "__main__":
    main()