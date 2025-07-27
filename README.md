# msc-degree-scripts

Scripts for FOREX (FOReign EXchange) rate data preprocessing, model training and verification.

## Requirements

- python 3
- jupyter
- numpy
- pandas
- scikit-learn
- statsmodels

## Models

- Neural Network (NN)
- Support Vector Machine (SVM)
- Random Forest (RF)

## How to run

### Preprocessing

1. Run 1-preprocessing/generate-datasets.py in the 1-preprocessing directory to generate each dataset variation.
2. In the 1-preprocessing/all-window-selection directory run mlp.py, rf.py and svr.py to run an initial evaluation of each dataset preprocessing variation of each of the default or near default models.
3. Run rename-files.py to rename all files in the result to their abbreviated forms
4. In the 1-preprocessing/all-window-selection/optimal directory run mlp.py, rf.py and svr.py in order to find the optimal results.
5. Run the any script in the 1-preprocessing/all-window-selection/plot and 1-preprocessing/all-window-selection/optimal/plot directories to get visualizations of the data.

### Model training and testing

Run any combination of the param.py, reverse-transform.py and param-plot.py scripts for each parameter or all in order to get their respective visualizations.

