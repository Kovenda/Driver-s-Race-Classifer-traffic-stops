# Imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
# Fitting randomForest Model Training Set
X_train = train.drop('TCOLE_RACE_ETHNICITY', axis=1)
y_train = train['TCOLE_RACE_ETHNICITY']
rf.fit(X_train ,y_train)
train_pred = rf.predict (X_train)
train_precn = precision_score (y_train, train_pred, average=None)
train_recal = recall_score (y_train, train_pred, average=None)
train_f1_score  = f1_score (y_train, train_pred, average=None)

# Training Set Evaluation | Confusion Matrix
conf_matrix = confusion_matrix (y_train, train_pred)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print ("Training precision", np.round (train_precn, 4), ". Mean =", np.round (np.mean (train_precn), 4))
print ("Training recall", np.round (train_recal, 4), ". Mean =", np.round (np.mean (train_recal), 4))
print ("Training f1_score", np.round (train_f1_score, 4), ". Mean =", np.round (np.mean (train_f1_score), 4))

# Learning Curves | Cross Validation Functions
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

# Plot Learning Curves for Training Set
plot_learning_curves(rf, X_train, y_train)

# Fitting Test Set
X_test = test.drop('TCOLE_RACE_ETHNICITY', axis=1)
y_test = test['TCOLE_RACE_ETHNICITY']
test_predRF = rf.predict (X_test)
test_precnRF = precision_score (y_test, test_predRF, average=None)
test_recalRF = recall_score (y_test, test_predRF, average=None)
test_f1_scoreRF  = f1_score (y_test, test_predRF, average=None)

# Test Set Evaluation
conf_matrixRF = confusion_matrix (y_test, test_predRF)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrixRF, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print ("testRF precision", np.round (test_precnRF, 4), ". Mean =", np.round (np.mean (test_precnRF), 4))
print ("testRF recall", np.round (test_recalRF, 4), ". Mean =", np.round (np.mean (test_recalRF), 4))
print ("testRF f1_score", np.round (test_f1_scoreRF, 4), ". Mean =", np.round (np.mean (test_f1_scoreRF), 4))

# Learning Curves | Cross Validation Test Set Functions
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
    
# Plot Learning Curves
plot_learning_curves(rf, X_train, y_train, X_test, y_test)

# Grid Search
param_grid = {
'classifier__n_estimators': [200, 500],
'classifier__max_features': ['auto', 'sqrt', 'log2'],
'classifier__max_depth' : [4,5,6,7,8],
'classifier__criterion' :['gini', 'entropy']}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(rf, param_grid, n_jobs= 1)
CV.fit(X_train, y_train)
print(CV.best_params_)
print(CV.best_score_)

# Re-fit with Grid Search Parameters
from sklearn.ensemble import RandomForestClassifier
rf_GridSearch = Pipeline(steps=[('preprocessor', preprocessor),
('classifier', RandomForestClassifier(criterion = 'entropy', 
                                      max_depth = 8, max_features= 'sqrt',
                                        n_estimators= 200))])

# Fit Training Set
rf_GridSearch.fit(X_train, y_train)
train_predRFGRID = rf_GridSearch.predict (X_train)
train_precnRFGRID = precision_score (y_train, train_predRFGRID, average=None)
train_recalRFGRID = recall_score (y_train, train_predRFGRID, average=None)
train_F1RFGRID = f1_score (y_train, train_predRFGRID, average=None)

# Training Set Validation
conf_matrixRFGRID = confusion_matrix (y_train, train_predRFGRID)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrixRFGRID, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print ("Training precision", np.round (train_precnRFGRID, 4), ". Mean =", np.round (np.mean (train_precnRFGRID), 4))
print ("Training recall", np.round (train_recalRFGRID, 4), ". Mean =", np.round (np.mean (train_recalRFGRID), 4))
print ("Training F1 score", np.round (train_F1RFGRID, 4), ". Mean =", np.round (np.mean (train_F1RFGRID), 4))

# Fit Test Set
test_predRFGRID = rf_GridSearch.predict (X_test)
test_precnRFGRID = precision_score (y_test, test_predRFGRID, average=None)
test_recalRFGRID = recall_score (y_test, test_predRFGRID, average=None)
test_f1_scoreRFGRID  = f1_score (y_test, test_predRFGRID, average=None)

# Test Evaluation
conf_matrixRFGRID = confusion_matrix (y_test, test_predRFGRID)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrixRFGRID, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print ("testSVM precision", np.round (test_precnRFGRID, 4), ". Mean =", np.round (np.mean (test_precnRFGRID), 4))
print ("testSVM recall", np.round (test_recalRFGRID, 4), ". Mean =", np.round (np.mean (test_recalRFGRID), 4))
print ("testSVM f1_score", np.round (test_f1_scoreRFGRID, 4),
       ". Mean =", np.round (np.mean (test_f1_scoreRFGRID), 4))
