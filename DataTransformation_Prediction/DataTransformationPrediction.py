from os import listdir
import pandas as pd
from application_logging.logger import App_Logger

class dataTransformPredict:
    """
        This class will be used for transforming the Good Raw Predict Data before loading it in Database!!

        Written By: Aman Sharma
        Version: 1.0
        Revisions: None
    """
    def __init__(self):
        self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
        self.logger = App_Logger()

    def addQuotesToStringValuesInColumn(self):
        """
            Method Name: addQuotesToStringValuesInColumn
            Description: This method replces the missing values in columns with "NULL" to
                         store in the table. We are using substring in the first column to
                         keep only "Integer" data for ease up the loading.
                         This column in anyways going to be removed during prediction.
            
            Written By: Aman Sharma
            Version: 1.0
            Revisions: None
        """
        try:
            log_file = open("Prediction_logs/dataTransformLog.txt","a+")
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                data = pd.read_csv(self.goodDataPath+"/"+file)
                data['DATE'] = data["DATE"].apply(lambda x: "'"+str(x)+"'")
                data.to_csv(self.goodDataPath+"/"+file,index =None,header=True)
                self.logger.log(log_file, "%s: Quotes added successfully!!" % file)
        except Exception as e:
            log_file = open("Prediction_logs/dataTransformLog.txt","a+")
            self.logger.log(log_file,"Data Transformation failed because:: %s" % e)
            log_file.close()
            raise e
        log_file.close()