from Training_Raw_Data_Validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from DataTransform_Training.DataTransformation import dataTransform
from application_logging.logger import App_Logger

class train_validation:
    def __init__(self,path):
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()
        self.dBOperation = dBOperation()
        self.file_object  = open("Training_Logs/Training_Main_Log.txt","a+")
        self.log_writer = App_Logger()

    def train_validation(self):
        try:
            self.log_writer.log(self.file_object,"Start of Validation on files for training !!")
            # extracting values from training schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()
            # getting the regex defined to validate filename
            regex = self.raw_data.manualRegexCreation()
            # validating filename of training files
            self.raw_data.validationFileNameRaw(regex,LengthOfDateStampInFile,LengthOfTimeStampInFile)
            # validating columnc length in the file
            self.raw_data.validateColumnLength(noofcolumns)
            # validating if any column has all values mising
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object,"Raw Data Validation Complete!!")

            self.log_writer.log(self.file_object,"Creating Training Database and tables on the basis of given schema !!")
            # create database with given name, if present open the connection. Create table with columnc given in schema
            self.dBOperation.createTableDB("Training",column_names)
            self.log_writer.log(self.file_object,"Table Creation Completed!!")
            self.log_writer.log(self.file_object,"Insertion of Data into Table started!!")
            # insert csv files in the table
            self.dBOperation.insertIntoTableGoodData("Training")
            self.log_writer.log(self.file_object,"Insertion in Table Completed!!")
            self.log_writer.log(self.file_object,"Deleting Good Data Folder!!")
            # Delete the good data folder after loading files in table
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")
            self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            # Move the bad files to archive folder
            self.raw_data.moveBadFilesToArchiveBad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object, "Validation Operation completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")
            # export data in table to csvfile
            self.dBOperation.selectingDatafromtableintocsv('Training')
            self.file_object.close()

        except Exception as e:
            raise e

            
