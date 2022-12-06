import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import pandas as pd
plt.style.use('ggplot')
dataframe = pd.read_csv('dataset.csv', sep=';')
dataframe.info()
X = dataframe.copy()
y = X.pop('label')
classes = ['kaleng minuman', 'gelas plastik', 'botol plastik besar', 'botol plastik kecil', 'botol kaca', 'kardus']

y_bin = label_binarize(y, classes=classes)
n_classes = y_bin.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size= 0.3, random_state=101)
X_train

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=101))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
 fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
 roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'purple'])
for i, color in zip(range(n_classes), colors):
 plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=101)

classifier_svc = svm.SVC(kernel='linear',random_state=0)
classifier_svc.fit(X_train, y_train)
y_pred = classifier_svc.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

acc = accuracy_score(y_test, y_pred)
print('Accuracy Score: ', acc)
print(classification_report(y_test, y_pred))