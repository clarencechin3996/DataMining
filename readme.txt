Phase 2: Data-Oriented Project

Description
Brief description of what the project does and what it's used for.

Environment and Tools
Operating System: MacOS
Programming Language: Python 3.11
Development Environment: PyCharm

Additional Packages
scikit-learn
pandas
NumPy

To reproduce the results reported in this project, follow these steps:

- Make sure you've set up your environment as described above and download s4761962 folder.
- Set your folder path to desktop or any intended path as you prefer.
- You may move your train.csv or test.csv in this folder for convenience.
- (Set your own path to load the train and test csv data) in main.py
   Path: ../(your path)/train.csv

- Run the script main.py.
- The results will be saved in the Python output/ directory.

Additional Information
- For initial hyperparameter tuning, we used GridSearchCV with a 5-fold cross-validation strategy.
- To further mitigate the risk of overfitting, additional manual tuning was performed on each classification technique. This involved iteratively adjusting hyperparameters based on cross-validation results and re-evaluating the model.


References
Scikit-learn Documentation
Pandas Documentation
