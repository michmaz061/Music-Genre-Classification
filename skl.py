import sys
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
import numpy as np
import classic as cl


data = pd.read_csv('../Music-Genre-Classification/Data/features_3_sec.csv')
data = data.iloc[0:, 1:]
encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])
X = data.iloc[0:, 0:58]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64)

file_path = 'outputclass3.txt'
sys.stdout = open(file_path, "w")
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
ss= ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
for train_idx, val_idx in ss.split(X, y):
    X_tr = np.array(X)[train_idx]
    y_tr = np.array(y)[train_idx]

    X_val = np.array(X)[val_idx]
    y_val = np.array(y)[val_idx]
    cl.main(X_tr, y_tr, X_val, y_val)

