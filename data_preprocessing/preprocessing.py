import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
        This class will be used to clean and transform the data before training.

        Written By: Aman Sharma
        Version: 1.0
        Revisions: None
    """
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self,data,columns):
        """
            Method Name: remove_columns
            Description: This method remove the given columns from a pandas dataframe.
            Output: A pandas DataFrame after removing the specified columns.
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,"Entered the remove_columnc method of the Preprocessor class")
        self.data = data
        self.columns = columns
        try:
            # drop the lables specified in the columns
            self.useful_data = self.data.drop(labels=self.columns,axis =1)
            self.logger_object.log(self.file_object,
                "Column removal Successful. Exited the remove_columns method of the Preprocessor class")
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occured in remove_columns method of the Preprocessor class. Exception message: "+str(e))
            self.logger_object.log(self.file_object,"Column removal Unsuccessful. Exited the remove_columns method of the Proeprocessor class")
            raise Exception()


    def separate_label_feature(self,data,label_column_name):
        """
            Method Name: separate_label_feature
            Description: This method separates the features and a Label columns.
            Output:  Returns two separate Dataframes, one containing features and the other containing Labels.
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """

        self.logger_object.log(self.file_object,"Entered the separate_label_feature method of the Preprocessor class")
        try:
            # Dropped the columns specified and separate the feature columns
            self.X = data.drop(labels = label_column_name,axis = 1)
            # Filter the Label columns
            self.Y = data[label_column_name]
            self.logger_object.log(self.file_object,"Label Sepration Operation Successful. Exited the separate_label_feature method of the Preprocessor class")
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occurred in separate_lable_feature method of the Preprocessor class. Exception message: '+ str(e))
            self.logger_object.log(self.file_object,
                                   "Label Sepration Operation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class")
            raise e
    
    def dropUnnecessaryColumns(self,data,columnNameList):
        """
            Method Name: dropUnnecessaryColumns
            Description: This method drops the unwanted columns as discussed in EDA section.
            Output: Data after removing the Unnecessary Columns
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None 
        """
        self.logger_object.log(self.file_object,"Entered the dropUnnecessaryColumns method of the Preprocessor class")
        try:
            data = data.drop(columnNameList,axis =1)
            self.logger_object.log(self.file_object,"Unnecessary columns remove operation Successful. Exited the dropUnnecessaryColumns method of the Preprocessor class")
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occurred in dropUnnecessaryColumns method of the Preprocessor class. Exception message: '+ str(e))
            self.logger_object.log(self.file_object,
                                   "Unnecessary columns remove operation Unsuccessful. Exited the dropUnnecessaryColumns method of the Preprocessor class")
            raise e
    
    def replaceInvalidValuesWithNull(self,data):
        """
            Method Name: replaceInvalidValuesWithNull
            Description: This method replaces invalid values i.e. "?" with null,
                         as discussed in EDA.
            Output: Data after replacing Invalid values with Null
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None 
        """
        self.logger_object.log(self.file_object,"Entered the replaceInvalidValuesWithNull method of the Preprocessor class")
        try:
            for column in data.columns:
                count = data[column][data[column]=="?"].count()
                if count != 0:
                    data[column] = data[column].replace('?',np.nan)
            self.logger_object.log(self.file_object,"Replacing Invalid values with Null operation Successful. Exited the replaceInvalidValuesWithNull method of the Preprocessor class")
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occurred in replaceInvalidValuesWithNull method of the Preprocessor class. Exception message: '+ str(e))
            self.logger_object.log(self.file_object,
                                   "Replacing Invalid values with Null operation Unsuccessful. Exited the replaceInvalidValuesWithNull method of the Preprocessor class")
            raise e

    def is_null_present(self,data):
        """
            Method Name: in_null_present
            Description: This method checks whether there are null values present in the pandas Dataframe or not.
            Output:  Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present and 
                    returns the list of columns for which null values are present.
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,"Entered the is_null_present method in Preprocessor class")
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            # Checks for the count of null values per column
            self.null_counts = data.isna().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present):
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null["missing values count"] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv("preprocessing_data/null_values.csv") # storing the null column information to file
            self.logger_object.log(self.file_object,'Finding missing values is a success. Data written to null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise e

    def encodeCategoricalValues(self,data):
        """
            Method Name: encodeCategoricalValues
            Description: This method encodes all the categorical values in the training set.
            Output: A dataframe which has all the categorical values encoded
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None 
        """
        self.logger_object.log(self.file_object,"Entered the encodeCategoricalValues method of the Preprocessor class")
        try:
            data["class"] = data["class"].map({'p':1,"e":2})
            for column in data.drop(['class'],axis=1).columns:
                data = pd.get_dummies(data,columns=[column])
            self.logger_object.log(self.file_object,"encodes all the categorical values operation Successful. Exited the encodeCategoricalValues method of the Preprocessor class")
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occurred in encodeCategoricalValues method of the Preprocessor class. Exception message: '+ str(e))
            self.logger_object.log(self.file_object,
                                   "encodes all the categorical values operation Unsuccessful. Exited the encodeCategoricalValues method of the Preprocessor class")
            raise e
    
    def encodeCategoricalValuesPrediction(self,data):
        """
            Method Name: encodeCategoricalValuesPrediction
            Description: This method encodes all the categorical values in the prediction set.
            Output: A dataframe which has all the categorical values encoded
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None 
        """
        self.logger_object.log(self.file_object,"Entered the encodeCategoricalValues method of the Preprocessor class")
        try:
            for column in data.columns:
                data = pd.get_dummies(data,columns=[column])
            self.logger_object.log(self.file_object,"encodes all the categorical values operation Successful. Exited the encodeCategoricalValues method of the Preprocessor class")
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occurred in encodeCategoricalValues method of the Preprocessor class. Exception message: '+ str(e))
            self.logger_object.log(self.file_object,
                                   "encodes all the categorical values operation Unsuccessful. Exited the encodeCategoricalValues method of the Preprocessor class")
            raise e
    
    def standardScalingData(self,X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    def logTransformation(self,X):
        for column in X.columns:
            X[column] += 1
            X[column] = np.log(X[column])

        return X


    def impute_missing_values(self, data):
        """
            Method Name: impute_missing_values
            Description: This method replaces all the missing values in the Dataframe using KNN imputer.
            Output: A Dataframe  which has all the missing values imputed.
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,'Entered the impute_missing_values method of the Preprocessor class')
        self.data = data
        try:
            imputer = KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            # Impute the missing values
            self.new_array = imputer.fit_transform(self.data)
            # Convert the nd-array returned in the step above to a Dataframe
            self.new_data = pd.DataFrame(data=self.new_array,columns=self.data.columns)
            self.logger_object.log(self.file_object,'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occurred in impute_missing_values method of the Preprocessor class. Exception message: "+str(e))
            self.logger_object.log(self.file_object,"Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class")
            raise Exception()

    def get_columns_with_zero_std_deviation(self,data):
        """
            Method Name: get_columns_with_zero_std_deviation
            Description: This method finds out the columns which have a standard deviation of zero.
            Output: List of the columns with standard deviation of zero
            On Failure: Raise Exception

            Written By: Aman Sharma
            Version: 1.0
            Revisions: None

        """
        self.logger_object.log(self.file_object,"Entered the get_columns_with_zero_std_deviation method of the Preprocessor class")
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop =[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # Check if standard deviation is zero
                    self.col_to_drop.append(x) # prepare the list of columns with standard deviation zero
            self.logger_object.log(self.file_object,'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message: "+str(e))
            self.logger_object.log(self.file_object,"Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class")
            raise Exception()