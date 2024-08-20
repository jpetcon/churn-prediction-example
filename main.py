import churn_prediction.feature_engineering as fe
import churn_prediction.churn_prediction as cp


def main():

    # Create initial data object and clean up raw data
    churn_data = fe.DataObject()

    churn_data.load_csv('data/customer-churn.csv')
    churn_data.drop_null_values()

    churn_data.fix_data_types('SeniorCitizen', object)
    churn_data.fix_data_types('TotalCharges', float)

    churn_data.map_target_variable()

    churn_data.remove_outliers()


    # Preprocess data to generate model input
    processed_data = fe.PreProcessData(df=churn_data.df)

    processed_data.yeo_johnson_transform('tenure')
    processed_data.yeo_johnson_transform('MonthlyCharges')
    processed_data.yeo_johnson_transform('TotalCharges')

    processed_data.scale_data()

    processed_data.one_hot_encode_data()


    # Select Predictive Features

    model_input_data = fe.FeatureSelection(df=processed_data.df, target=churn_data.df.Churn)

    model_input_data.rfe_selection()


    # Train model and get F1 Score

    prediction = cp.ChurnPrediction(df = model_input_data.df, target = model_input_data.target)

    prediction.split_data()

    prediction.hyperparameter_tuning()

    prediction.threshold_tuning()

    prediction.model_evaluation()

    print(prediction.f1)


if __name__ == "__main__":
    main()