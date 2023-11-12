Phase 2: Data-Oriented Project

Description
In this data-oriented project, our dataset named train.csv is designed to closely simulate real-world scenarios, reflecting the inherent complexities found in natural data. It originates from the CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html) dataset, but we have deliberately introduced various realistic challenges, such as missing values, a diverse range of data scales, and outliers. To create this dataset, we employed a neural network to extract features from the original CIFAR-10 data and made certain modifications to the resultant features. As a result, the dataset exhibits a compelling resemblance to naturally occurring data, offering an excellent opportunity to study and develop robust solutions applicable to real-world data analysis.

In the "train.csv" file, each row, except for the first one, represents a single data point, and there are a total of 2180 data points in this dataset. The first 100 columns (Num_Col_0 to Num_Col_99) contain numerical features, while the subsequent 28 columns (Cat_Col_100 to Cat_Col_127) contain nominal features. The last column "Label" indicates the corresponding label for each data point. The dataset includes a total of 10 classes, which are denoted by numbers 0 to 9, representing the following categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck, respectively. The main objective based on the provided labeled data is to develop a classifier capable of accurately classifying a data point into one of the ten classes for unseen data. 

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
