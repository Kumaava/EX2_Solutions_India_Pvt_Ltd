import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn import ensemble
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split

class Exploration :
    
    def __init__(self,path):
        self.path = path
        self.data = pd.read_csv(path)
    
    def shapeandhead(self):
        data = pd.read_csv(self.path)
        train_data = self.data.drop(["Target_1","Target_2"],axis=1)
        print "test_data shape : ",train_data.shape
        print "\n---------------------------------Train-Data---------------------------------------"
        print data.head()
        return
    
    
    def types(self):
        
        pd.options.display.max_rows = 15
        dtype_df = self.data.dtypes.reset_index()
        dtype_df.columns = ["Count", "Column Type"]
        print dtype.groupby("Column Type").aggregate('count').reset_index()
        return dtype_df
    
    def scatterplot(self,scattermat = False):
        
        if scattermat == True:
            print "will take some time___ have patience"
            pd.plotting.scatter_matrix(self.data, alpha = 0.3, figsize = (10,6), diagonal = 'kde');
        else : 
            for i in self.data.columns:
                plt.scatter(range(self.data.shape[0]),np.sort(self.data[str(i)].values))
                plt.xlabel('index', fontsize=12)
                plt.ylabel(str(i), fontsize=12)
                plt.show()
            
        return
    
    def missingvalplot(self):
        #data = "input/train.csv"
        missing_df=self.data.isnull().sum(axis=0).reset_index()
        missing_df.columns = ['column_name', 'missing_count']
        missing_df = missing_df.ix[missing_df['missing_count']>0]
        missing_df = missing_df.sort_values(by='missing_count')
        print missing_df
        ind = np.arange(missing_df.shape[0])
        width = 0.9
        fig, ax = plt.subplots(figsize=(7,7))
        rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
        ax.set_yticks(ind)
        ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
        ax.set_xlabel("Count of missing values")
        ax.set_title("Number of missing values in each column")
        plt.show()
        return
        
    def corrmapplot(self):
        corrmat = self.data.corr(method='spearman')
        f, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(corrmat, vmax=1., square=True)
        plt.title("Important variables correlation map", fontsize=20)
        plt.show()
        return
    
    def corrCoefplot(self, Target_1 = False, Target_2 = False):
        # Let us just impute the missing values with mean values to compute correlation coefficients
        mean_values = self.data.mean(axis=0)
        train_df_new = self.data.fillna(mean_values, inplace=True)
        
        # Now let us look at the correlation coefficient of each of these variables
        if Target_1 == True:
            x_cols = [col for col in train_df_new.columns if col not in ['Target_1'] if train_df_new[col].dtype=='float64']
        elif Target_2 == True:
            x_cols = [col for col in train_df_new.columns if col not in ['Target_2'] if train_df_new[col].dtype=='float64']
        else :
            raise Exception("Set_anyone : Target_1 == True or Target_2 == True")
            
        labels = []
        values = []
        for col in x_cols:
            labels.append(col)
            values.append(np.corrcoef(train_df_new[col].values, train_df_new.Target_1.values)[0,1])
        corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
        corr_df = corr_df.sort_values(by='corr_values')
        print corr_df
        ind = np.arange(len(labels))
        width = 0.9
        fig, ax = plt.subplots(figsize=(7,7))
        rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='blue')
        ax.set_yticks(ind)
        ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
        ax.set_xlabel("Correlation coefficient")
        ax.set_title("Correlation coefficient of the variables")
        #autolabel(rects)
        plt.show()
        return



class Scoreandevaluate(Exploration):
    
    def preprocessing_multi(self, minusone_sub = False , mean_sub = False):
        print( "\nProcessing self.data")
        for c in self.data.columns:
            if minusone_sub == True:
                self.data[c]=self.data[c].fillna(-1)                              #imputing NA with -1
            
            elif mean_sub == True :
                print np.mean(self.data[c])
                self.data[c] = self.data[c].fillna(np.mean(self.data[c]))              #imputing NA with mean
            
            if self.data[c].dtype == 'object':
                lbl = LabelEncoder()
                self.data[c] = lbl.fit_transform(list(self.data[c].values))   # Label encoding object type self.data
        return self.data

    def minmaxscaling(self):
        """does min-max scaling to normalise the feature influence"""
        sc = MinMaxScaler()
        for i in self.data.columns:
            self.data[i] = sc.fit_transform(self.data[i])
        return self.data
    
    def preprocessing(self, minusone_sub = False ,mean_sub = False):
        if mean_sub == True:
            train = self.preprocessing_multi(mean_sub = True)
        elif minusone_sub == True:
            train = self.preprocessing_multi(minusone_sub=True)
        print train.head()
        train = self.minmaxscaling()
        Target_1 = train.Target_1
        Target_2 = train.Target_2
        train = train.drop(["Target_1","Target_2"],axis=1)
        return train,Target_1,Target_2

    def result(self,Tar1 = False, Tar2= False,minusone_sub = False ,mean_sub = False,GBM= False, XGB=False):
        """gives the result of regression with score using any algo provided """
        if GBM == True:
            reg = ensemble.GradientBoostingRegressor()
        elif XGB == True:
            reg = XGBRegressor()
        
        if minusone_sub == True:
            train,Target_1,Target_2 = self.preprocessing(minusone_sub=True)
        elif mean_sub == True:
            train,Target_1,Target_2 = self.preprocessing(mean_sub=True)
        
        if Tar1 == True:
            X_train, X_test, y_train, y_test = train_test_split(train,Target_1, test_size = 0.8, random_state = 1)
        elif Tar2 == True:
            X_train, X_test, y_train, y_test = train_test_split(train,Target_2, test_size = 0.8, random_state = 1)
        
        reg.fit(X_train,y_train)
        pred = reg.predict(X_test)
        evaluation = explained_variance_score(pred,y_test)
        score = reg.score(X_test,y_test)
        if hasattr(reg, "feature_importances_") :
            x = reg.feature_importances_
            print x
        else :
            print "no feature_importances_ function in the model"
        
        print {"score": score}
        print {"evaluation":evaluation}
        #print {"featureimp":fi}
        return score,evaluation







