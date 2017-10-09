# Developed by Roberto Spadim - SPAEmpresarial - Brazil - roberto@spadim.com.br
# 2017-10-09 - first version
#
#
#
import time
import pandas as pd
import numpy as np
from math import factorial
from itertools import permutations
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import roc_auc_score,log_loss,mean_absolute_error,mean_squared_error,r2_score

# small black magic
class CategoricalReorders(TransformerMixin):
    def __init__(self,classifier=None,
                max_iterations=721,verbose=False,random_permutation=None,
                tree_seed=19870425,random_seed=19870425):
        self.classifier =classifier
        self.max_iterations=max_iterations
        self.verbose    =verbose
        self.random_permutation=random_permutation
        self.tree_seed  =tree_seed
        self.random_seed=random_seed
        self.transformers={}
    def fit(self,X,y):
        if not (isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
            raise Exception("X isn't pandas.DataFrame, nor NumPy.ndarray, type(X):",type(X))
        if not (isinstance(y, pd.Series) or isinstance(y, np.ndarray)):
            raise Exception("y isn't pandas.Series, nor NumPy.ndarray, type(y):",type(y))
        if(y.ndim!=1):
            raise Exception("Dimension of Y isn't 1, shape:",np.shape(Y))

        if(self.verbose):
            print("Fitting columns: ",X.columns)
        for X1 in X.columns:
            if(self.verbose):
                print("Fitting: ",X1)
            self.transformers[X1]=CategoricalReorder(
                    classifier=self.classifier,
                    max_iterations=self.max_iterations,
                    verbose=self.verbose,
                    random_permutation=self.random_permutation,
                    tree_seed=self.tree_seed,
                    random_seed=self.random_seed)
            self.transformers[X1].fit(X[X1],y)
            if(self.verbose):
                print("\n")

    def transform(self,X):
        if not (isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
            raise Exception("X isn't pandas.DataFrame, nor NumPy.ndarray, type(X):",type(X))
        ret=X.copy()
        for X1 in X.columns:
            if(self.verbose):
                print("Transforming: ",X1)
            ret[X1]=self.transformers[X1].transform(X[X1])
        return ret

    def inverse_transform(self,X):
        if not (isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
            raise Exception("X isn't pandas.DataFrame, nor NumPy.ndarray, type(X):",type(X))
        ret=X.copy()
        for X1 in X.columns:
            if(self.verbose):
                print("Inverse Transforming: ",X1)
            ret[X1]=self.transformers[X1].inverse_transform(X[X1])
        return ret

    
    
# small black magic
class CategoricalReorder(TransformerMixin):
    def __init__(self,classifier=None,
                max_iterations=721,verbose=False,random_permutation=None,
                tree_seed=19870425,random_seed=19870425):
        self.classifier =classifier
        self.max_iterations=max_iterations
        self.verbose    =verbose
        self.random_permutation=random_permutation
        self.tree_seed  =tree_seed
        self.random_seed=random_seed
        self.dicts      =None
        self.optimized  =False
        
    def fit(self,X,y):
        if not (isinstance(X, pd.Series) or isinstance(X, np.ndarray)):
            raise Exception("X isn't pandas.DataFrame, nor NumPy.ndarray, type(X):",type(X))
        if not (isinstance(y, pd.Series) or isinstance(y, np.ndarray)):
            raise Exception("y isn't pandas.Series, nor NumPy.ndarray, type(y):",type(y))
        if(X.ndim!=1):
            raise Exception("Dimension of X isn't 1, use CategoricalReorders instead, shape:",np.shape(X))
        if(y.ndim!=1):
            raise Exception("Dimension of Y isn't 1, shape:",np.shape(Y))
        self._reorder(X,y)
        
    def transform(self,X):
        if not (isinstance(X, pd.Series) or isinstance(X, np.ndarray)):
            raise Exception("X isn't pandas.Series, nor NumPy.ndarray, type(X):",type(X))
        if(X.ndim!=1):
            raise Exception("Dimension of X isn't 1, use CategoricalReorders instead, shape:",np.shape(X))
        if(self.dicts is None):
            raise Exception("Dict not created, fit transformer first!")
        if(self.optimized):
            return X.replace(self.dicts)
        return X
    
    def inverse_transform(self,X):
        if not (isinstance(X, pd.Series) or isinstance(X, np.ndarray)):
            raise Exception("X isn't pandas.Series, nor NumPy.ndarray, type(X):",type(X))
        if(X.ndim!=1):
            raise Exception("Dimension of X isn't 1, use CategoricalReorders instead, shape:",np.shape(X))
        if(self.dicts is None):
            raise Exception("Dict not created, fit transformer first!")
        
        if(self.optimized):
            reverse={}
            for k,v in self.dicts:
                reverse[v]=k
            return X.replace(reverse)
        return X
    
    def _printError(self,model,X,y):
        if(self.classifier):
            print('ROC_AUC/LogLoss: ',
                      roc_auc_score(y,model.predict_proba(X)[:,1]),'/',
                      log_loss(     y,model.predict_proba(X)[:,1]))
        else:
            print('MAE/MSE/R²: ',
                      mean_absolute_error(y,model.predict(X)[:,1]),'/',
                      mean_squared_error( y,model.predict(X)[:,1]),'/',
                      r2_score(           y,model.predict(X)[:,1]))

    def _reorder(self,X,y):
        #time it
        start     =time.time()
        
        values    =X.sort_values().unique() #nd array, since df[col] is a series
        len_values=len(values)

        #min dictionary (l<=>l)
        self.optimized=False
        self.dicts  ={l:l for l in values}
        if(len_values<3):
            if(self.verbose):
                print('Unique=',len_values,', values=',values)
                print('LESS THAN 3 UNIQUE VALUES, Time spent (seconds):',time.time() - start)
            return

        #classifier or regressor?
        if(self.classifier is None):
            values_y=y.unique() #nd array, since df[col] is a series
            len_values_y=len(values_y)
            self.classifier=False
            if(len_values_y!=2):
                if(self.verbose):
                    print("Problem identification: Regression")
            elif((values_y[0]==0 or values_y[0]==1) and
                 (values_y[1]==0 or values_y[1]==1)):
                if(self.verbose):
                    print("Problem identification: Classification")
                self.classifier=True

        
        #Current Values
        if(self.classifier):
            model=DecisionTreeClassifier(max_depth=None,presort=True,criterion='entropy',class_weight='balanced',random_state=self.tree_seed)
        else:
            model=DecisionTreeRegressor(max_depth=None,presort=True,random_state=self.tree_seed)
        model.fit(X.values.reshape(-1,1),y)
        min_depth_count=model.tree_.max_depth
        if(self.verbose):
            print('Unique=',len_values,', depth=',min_depth_count,', values=',values)
            self._printError(model,X.values.reshape(-1,1),y)
        if(min_depth_count==1):
            if(self.verbose):
                print('DEPTH=1, Time spent (seconds):',time.time() - start)
            return

        #Naive order by count
        if(self.classifier):
            first_try=X[y==0].value_counts(sort=True,ascending=True)
        else:
            #maybe a median/mean order? for example, target_col>mean(target) ?
            first_try=X.value_counts(sort=True,ascending=True)
        l,values_dict=0,{}
        for i in first_try.index:
            values_dict[values[l]]=i
            l+=1

        if(self.classifier):
            model=DecisionTreeClassifier(max_depth=None,presort=True,criterion='entropy',class_weight='balanced',random_state=self.tree_seed)
        else:
            model=DecisionTreeRegressor(max_depth=None,presort=True,random_state=self.tree_seed)
        model.fit(X.replace(values_dict).values.reshape(-1,1),y)
        # better than l<=>l ?
        if(min_depth_count>model.tree_.max_depth):
            self.optimized=True
            self.dicts=values_dict
            if(self.verbose):
                print('Naive order by count: from ',min_depth_count,' to ',model.tree_.max_depth,', dict:',values_dict)
                self._printError(model,X.replace(values_dict).values.reshape(-1,1),y)
            min_depth_count=model.tree_.max_depth
            if(min_depth_count==1):
                if(self.verbose):
                    print('DEPTH=1, Time spent (seconds):',time.time() - start)
                return
        elif(self.verbose):
            print('=[ No optimization using naive order by Count')

        # Search Space:
        # maybe random_permutatition isn't the best method... 
        #     if len(permutations)~=factorial(len_values) < max_iterations, we can use permutatition (real brute force)
        random_permutation=self.random_permutation
        if(random_permutation==None):
            random_permutation=False
            if(factorial(len_values)>self.max_iterations):
                random_permutation=True
                if(self.verbose):
                    print('Using RANDOM SAMPLING, search space is too big')
            elif(self.verbose):
                print('Factorial(length) (',factorial(len_values),') <= max_iterations (',self.max_iterations,'), USING PERMUTATION')

        # TODO: maybe we can do better with GA ?!
        if(random_permutation):
            # random permutation ( good lucky =] )
            np.random.seed(self.random_seed)
            space=range(self.max_iterations)
        else:
            # default itertools permutation
            space=permutations(values)

        count=0
        for perm in space:
            if(count>self.max_iterations):
                break
            # random permutation
            if(random_permutation):
                perm=np.random.permutation(values)

            values_dict={values[i]:perm[i] for i in range(0,len_values)}
            if(self.classifier):
                model=DecisionTreeClassifier(max_depth=None,presort=True,criterion='entropy',class_weight='balanced',random_state=self.tree_seed)
            else:
                model=DecisionTreeRegressor(max_depth=None,presort=True,random_state=self.tree_seed)
            model.fit(X.replace(values_dict).values.reshape(-1,1),y)
            if(min_depth_count>model.tree_.max_depth):
                self.optimized=True
                self.dicts=values_dict
                if(self.verbose):
                    print(count,'/',self.max_iterations,'NEW!!! from',min_depth_count,' to ',model.tree_.max_depth,' dict:',values_dict)
                    self._printError(model,X.replace(values_dict).values.reshape(-1,1),y)
                min_depth_count=model.tree_.max_depth
                if(min_depth_count==1):
                    if(self.verbose):
                        print('DEPTH=1')
                    break
            count+=1
        if(self.verbose):
            print('Time spent (seconds):',time.time() - start)
        return
