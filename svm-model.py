# Imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load data
racialProf = pd.read_csv ("filePath")

# Train Test Split
from sklearn.model_selection import train_test_split
train, test, = train_test_split(racialProf,
test_size=0.2)

# Dropping irrelevant columns from Training Set
train = train.drop('Unnamed: 0', axis=1)
train = train.drop('Stop_Key', axis=1)
train = train.drop('Location', axis=1)
train = train.drop('Zip_Code_1', axis=1)


# Dropping irrelevant columns from Test Set
test = test.drop('Unnamed: 0', axis=1)
test = test.drop('Stop_Key', axis=1)
test = test.drop('Location', axis=1)
test = test.drop('Zip_Code_1', axis=1)

# Building Data Imputation and Encoding Pipeline
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
numeric_features = train.select_dtypes(include=['int64','float64']).columns
categorical_features = train.select_dtypes(include=['object']).drop(['TCOLE_RACE_ETHNICITY'], axis=1).columns
preprocessor = ColumnTransformer( transformers=[('num', numeric_transformer, numeric_features),
                                                ('cat', categorical_transformer, categorical_features)])
# Fitting SVM Model
svm_classifier = Pipeline(steps=[('preprocessor', preprocessor),
('classifier', SVC(kernel="poly", degree=3))])
X_train = train.drop('TCOLE_RACE_ETHNICITY', axis=1)
y_train = train['TCOLE_RACE_ETHNICITY']
svm_classifier.fit(X_train, y_train)

# Fitting Training Set
train_predSVM = svm_classifier.predict (X_train)
train_precnSVM = precision_score (y_train, train_predSVM, average=None)
train_recalSVM = recall_score (y_train, train_predSVM, average=None)
train_f1_scoreSVM  = f1_score (y_train, train_predSVM, average=None)

# Training Set Evaluation
conf_matrix = confusion_matrix (y_train, train_predSVM)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print ("Training precision", np.round (train_precnSVM, 4), ". Mean =", np.round (np.mean (train_precnSVM), 4))
print ("Training recall", np.round (train_recalSVM, 4), ". Mean =", np.round (np.mean (train_recalSVM), 4))
print ("Training f1_score", np.round (train_f1_scoreSVM, 4), ". Mean =", np.round (np.mean (train_f1_scoreSVM), 4))

# Learning curves|cross validation Functions
def plot_curve (pltid, x, y1, y2, ylab):
    pltid.plot(x, y1, "r-+", linewidth=2, label="train")
    pltid.plot(x, y2, "b-", linewidth=3, label="val")
    pltid.legend()
    pltid.set_title (ylab) 

def plot_learning_curves(model, X, y):
  startobs = 999
  increment = 1000
  train_accur, val_accur, train_precn, val_precn, train_recall, val_recall, n_obs = [], [], [], [], [], [], []
  for m in range(startobs, len(X_train), increment):
      print ("nobs=", m)
      model.fit (X[:m], y[:m])

      y_train_predict = model.predict (X[:m])
      y_val_predict = model.predict (X)

      train_accur.append (accuracy_score(y [:m], y_train_predict))
      train_precn.append (precision_score(y [:m], y_train_predict, average='macro'))
      train_recall.append (recall_score(y [:m], y_train_predict, average='macro'))

      val_accur.append (accuracy_score(y, y_val_predict))
      val_precn.append (precision_score(y, y_val_predict, average='macro'))
      val_recall.append (recall_score(y, y_val_predict, average='macro'))
      n_obs.append (m)

    fig, axs = plt.subplots(1,3, figsize=(15,5))
    plot_curve (axs[0], n_obs, train_accur, val_accur, "Accuracy")
    plot_curve (axs[1], n_obs, train_precn, val_precn, "Precision")
    plot_curve (axs[2], n_obs, train_recall, val_recall, "Recall")
 
# Plotting Learning Curves for Training Set
plot_learning_curves(svm_classifier, X_train, y_train)

# Fitting Test Set
X_test = test.drop('TCOLE_RACE_ETHNICITY', axis=1)
y_test = test['TCOLE_RACE_ETHNICITY']
test_predSVM = svm_classifier.predict (X_test)
test_precnSVM = precision_score (y_test, test_predSVM, average=None)
test_recalSVM = recall_score (y_test, test_predSVM, average=None)
test_f1_scoreSVM  = f1_score (y_test, test_predSVM, average=None)

# Test Set Evaluation | Confusion Matrix
conf_matrixSVM = confusion_matrix (y_test, test_predSVM)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrixSVM, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print ("testSVM precision", np.round (test_precnSVM, 4), ". Mean =", np.round (np.mean (test_precnSVM), 4))
print ("testSVM recall", np.round (test_recalSVM, 4), ". Mean =", np.round (np.mean (test_recalSVM), 4))
print ("testSVM f1_score", np.round (test_f1_scoreSVM, 4), ". Mean =", np.round (np.mean (test_f1_scoreSVM), 4))

# Learning curves|cross validation Functions
def plot_curve (pltid, x, y1, y2, ylab):
    pltid.plot(x, y1, "r-+", linewidth=2, label="train")
    pltid.plot(x, y2, "b-", linewidth=3, label="test")
    pltid.legend()
    pltid.set_title (ylab) 

def plot_learning_curves(model, X, y, X_test, y_test):
  startobs = 999
  increment = 1000
   
  train_accur, test_accur, train_precn, test_precn, train_recall, test_recall, n_obs = [], [], [], [], [], [], []
  for m in range(startobs, len(X_train), increment):
      print ("nobs=", m)
      model.fit (X[:m], y[:m])
      
      y_train_predict = model.predict (X[:m])
      y_test_predict = model.predict (X_test)
      
      train_accur.append (accuracy_score(y [:m], y_train_predict))
      train_precn.append (precision_score(y [:m], y_train_predict, average='macro'))
      train_recall.append (recall_score(y [:m], y_train_predict, average='macro'))
      
      test_accur.append (accuracy_score(y_test, y_test_predict))
      test_precn.append (precision_score(y_test, y_test_predict, average='macro'))
      test_recall.append (recall_score(y_test, y_test_predict, average='macro'))
      n_obs.append (m)
  
  fig, axs = plt.subplots(1,3, figsize=(15,5))
  plot_curve (axs[0], n_obs, train_accur, test_accur, "Accuracy")
  plot_curve (axs[1], n_obs, train_precn, test_precn, "Precision")
  plot_curve (axs[2], n_obs, train_recall, test_recall, "Recall")
 
# Plotting Learning Curves for Test Set
plot_learning_curves(svm_classifier, X_train, y_train, X_test, y_test)

# Grid Search for SVM
param_grid = {
'classifier__kernel': ['poly'],
'classifier__degree': [1,2,3],
'classifier__decision_function_shape': ['ovr','ovo']}
from sklearn.model_selection import GridSearchCV
CV1 = GridSearchCV(svm_classifier, param_grid, n_jobs= 1)
CV1.fit(X_train, y_train)
print(CV1.best_params_)
print(CV1.best_score_)

# Re-fit SVM with Grid Search suggestions
svm_classifierGridSearch = Pipeline(steps=[('preprocessor', preprocessor),
('classifier', SVC(decision_function_shape= 'ovr', degree= 1, kernel= 'poly'
))])

# Fit Training Set
svm_classifierGridSearch.fit(X_train, y_train)
train_predSVMGRID = svm_classifierGridSearch.predict (X_train)
train_precnSVMGRID  = precision_score (y_train, train_predSVMGRID, average=None)
train_recalSVMGRID  = recall_score (y_train, train_predSVMGRID, average=None)
train_f1_scoreSVMGRID  = f1_score (y_train, train_predSVMGRID, average=None)

# Training Set Evaluation
conf_matrixSVMGRID = confusion_matrix (y_train, train_predSVMGRID)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrixSVMGRID, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print ("Training precision", np.round (train_precnSVMGRID, 4), ". Mean =", np.round (np.mean (train_precnSVMGRID), 4))
print ("Training recall", np.round (train_recalSVMGRID, 4), ". Mean =", np.round (np.mean (train_recalSVMGRID), 4))
print ("Training f1_score", np.round (train_f1_scoreSVMGRID, 4),
       ". Mean =", np.round (np.mean (train_f1_scoreSVMGRID), 4))

# Fit Test Set
test_predSVMGRID = svm_classifierGridSearch.predict (X_test)
test_precnSVMGRID = precision_score (y_test, test_predSVMGRID, average=None)
test_recalSVMGRID = recall_score (y_test, test_predSVMGRID, average=None)
test_f1_scoreSVMGRID  = f1_score (y_test, test_predSVMGRID, average=None)

# Test Set Evaluation
conf_matrixGRID = confusion_matrix (y_test, test_predSVMGRID)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrixGRID, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print ("testSVM precision", np.round (test_precnSVMGRID, 4), ". Mean =", np.round (np.mean (test_precnSVMGRID), 4))
print ("testSVM recall", np.round (test_recalSVMGRID, 4), ". Mean =", np.round (np.mean (test_recalSVMGRID), 4))
print ("testSVM f1_score", np.round (test_f1_scoreSVMGRID, 4),
       ". Mean =", np.round (np.mean (test_f1_scoreSVMGRID), 4))
