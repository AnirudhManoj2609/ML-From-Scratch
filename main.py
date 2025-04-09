import os
import sys

sys.path.append("slr")
sys.path.append("mlr")

def run_slr():
    #Import
    from regression.slr.slr_model import LinearRegression
    from regression.slr.slr_utils import load_csv,plot_regression

    #Add datasets
    X,Y = load_csv("data/slr_data.csv")

    model = LinearRegression()
    model.train(X,Y,learning_rate=0.01,iters=1000)

    predictions = model.predicting(X)

    plot_regression(X,Y,predictions)

def main():
    choice = input("Enter the model: ").strip().lower()
    if choice == "slr":
        run_slr()
    else:
        print("Invalid choice\n")

if __name__ == "__main__":
    main()