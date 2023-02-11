import pandas
from file_operations.file_methods import File_Operation
from data_preprocessing.preprocessing import Preprocessor
from data_ingestion.data_loader_prediction import Data_Getter_Pred
from application_logging.logger import App_Logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation

class prediction:
    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt","a+")
        self.log_writer = App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):
        try:
            # delete the existing prediction file
            self.pred_data_val.deletePredictionFile()
            self.log_writer.log(self.file_object,"Start of Prediction")
            data_getter = Data_Getter_Pred(self.file_object,self.log_writer)
            data = data_getter.get_data()
            preprocessor = Preprocessor(self.file_object,self.log_writer)

            is_null_present,cols_with_missing_values = preprocessor.is_null_present(data)
            if(is_null_present):
                data = preprocessor.impute_missing_values(data)

            data= preprocessor.logTransformation(data)

            # scale the prediction data
            data_scaled = pandas.DataFrame(preprocessor.standardScalingData(data),columns=data.columns)
            file_loader = File_Operation(self.file_object,self.log_writer)
            kmeans = file_loader.load_model("KMeans")

            clusters = kmeans.predict(data_scaled)
            data_scaled['clusters'] = clusters
            clusters = data_scaled['clusters'].unique()
            result = []

            for i in clusters:
                cluster_data = data_scaled[data_scaled['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                for val in (model.predict(cluster_data.values)):
                    result.append(val)
            result = pandas.DataFrame(result,columns=["Predictions"])
            path = "Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True)
            self.log_writer.log(self.file_object,"End of Prediction")
        except Exception as ex:
            self.log_writer.log(self.file_object,"Error occured while running the prediction!! Error:: %s" % ex)
            raise ex
        return path