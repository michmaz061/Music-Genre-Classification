import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def model_assess(model, title="Default"):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print('Accuracy', title, ':', round(accuracy_score(y_test, pred), 5), '\n')


data = pd.read_csv('../Music-Genre-Classification/Data/features_3_sec.csv')
data = data.iloc[0:, 1:]
encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])
X = data.iloc[0:, 0:58]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64)

nb = GaussianNB()
model_assess(nb, "Naive Bayes")

sgd = SGDClassifier(max_iter=5000, random_state=0)
model_assess(sgd, "Stochastic Gradient Descent")

knn = KNeighborsClassifier(n_neighbors=64)
model_assess(knn, "KNN")

tree = DecisionTreeClassifier()
model_assess(tree, "Decission trees")

randforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_assess(randforest, "Random Forest")

xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model_assess(xgb, "Cross Gradient Booster")
