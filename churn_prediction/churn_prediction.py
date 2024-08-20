from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, TunedThresholdClassifierCV



class ChurnPrediction:

    def __init__(self, df, target, x_train = None, x_test = None, y_train = None, y_test = None, model_hyperparams = None, threshold = None, f1 = None):
        
        self.df = df
        self.target = target
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.model_hyperparams = model_hyperparams

        self.threshold = threshold

        self.f1 = f1


    
    def split_data(self):
        '''Splits out data for training and final test set'''

        try:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df, self.target, test_size=0.2, random_state=0)
        
        except:
            raise NotImplementedError("Unable to split data")
        

    def hyperparameter_tuning(self):
        '''Uses cross-validated grid search to tune hyperparameters of model'''

        try:
            parameters = {'n_estimators' : [100, 500, 1000, 2000], 'max_depth' : [3, 6, 9]}
            model = GradientBoostingClassifier()

            gs = GridSearchCV(estimator= model, param_grid= parameters, scoring='f1')
            gs.fit(self.x_train, self.y_train)

            self.model_hyperparams = gs.best_params_
        
        except:
            raise NotImplementedError("Unable to tune hyperparameters")

    


    
    def threshold_tuning(self):
        '''Tunes probability cutoff threshold by maximising f1 score'''

        try:
            model = GradientBoostingClassifier(n_estimators=self.model_hyperparams['n_estimators'], max_depth=self.model_hyperparams['max_depth'], random_state=0)
            classifier = TunedThresholdClassifierCV(model, scoring = 'f1')

            classifier.fit(self.x_train, self.y_train)

            self.threshold = classifier.best_threshold_
        
        except:
            raise NotImplementedError("Unable to tune classification threshold")

    
    
    def model_evaluation(self):
        '''Fits model to test set to determine F1 Score'''

        try:
            model = GradientBoostingClassifier(n_estimators=self.model_hyperparams['n_estimators'], max_depth=self.model_hyperparams['max_depth'], random_state=0)
            model.fit(self.x_train, self.y_train) 

            y_pred = (model.predict_proba(self.x_test)[:,1] >= self.threshold).astype(bool)
            self.f1 = f1_score(self.y_test, y_pred)
        
        except:
            raise NotImplementedError("Unable to evaluate model")




