from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import  QuadraticDiscriminantAnalysis
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

models = {
    "KNN": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "TREE": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(n_estimators=20),
    "SVM": SVC(gamma='scale'),
    "PERC": Perceptron(max_iter=2000),
    # "GB": GradientBoostingClassifier(),
    "GNB": GaussianNB(),
    "SVC": SVC(),
    "RFC": RandomForestClassifier(),

    "LR": LogisticRegression(solver='lbfgs', max_iter=1000),
    "MLP": MLPClassifier(alpha=1, max_iter=1000),
    # "GPC": GaussianProcessClassifier(1.0 * RBF(1.0)),
    # "ABC": AdaBoostClassifier(n_estimators=100, random_state=0),
    "QDA": QuadraticDiscriminantAnalysis(),

    "BEST": Perceptron(penalty=None, n_iter_no_change=10, alpha=0.1, l1_ratio=0.415, fit_intercept=True,
                       max_iter=5000, tol=0.01, shuffle=True, verbose=0, eta0=2.0, n_jobs=None, random_state=0,
                       early_stopping=False, validation_fraction=0.5
                       , class_weight=None, warm_start=False)

}
def klasyf(klasyfikator,trainSamples,trainLabels,testSamples,testLabels):
  model = models[klasyfikator]
  model.fit(trainSamples, trainLabels)
  predictedLabels = model.predict(testSamples)

  print(confusion_matrix(testLabels, predictedLabels))
  print(classification_report(testLabels, predictedLabels))  # precision    recall  f1-score   support
  accuracy = accuracy_score(testLabels, predictedLabels)
  print("Accuracy: {:.2f}".format(accuracy))
  c_kappa = cohen_kappa_score(testLabels, predictedLabels)
  print("Cohen's Kappa: {:.2f}".format(c_kappa))

def main(trainSamples,trainLabels,testSamples,testLabels):
  for x in models:
    print(x)
    print(time.strftime("%H:%M:%S"))
    time_pocz = time.process_time()
    klasyf(x,trainSamples,trainLabels,testSamples,testLabels)
    time_kon = time.process_time()
    print("czas wykoniania [s]: ", time_kon - time_pocz, "\n\n")