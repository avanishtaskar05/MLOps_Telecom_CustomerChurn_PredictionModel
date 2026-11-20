from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from imblearn.over_sampling import SMOTE
def data_preprocessing(df):
    '''
    # Step1: Drop duplicated values
    # Step2: Feature Engineering and Feature Selection (Optional)
    # Step3: Split the data into X and y
    # Step4 Split the data into train and test sets
    # Step5 Segregate the data into categorical and numerical columns
    # Step6: Build pipelines for numerical_pipeline: imputation and scaling
    # Step7: Build pipelines for categorical_pipeline: imputation and encoding
    # Step8: Concat numerical and categorical pipelines using ColumnTransformer
    # Step9: fit the transformer on train and test data
    # Step10: fit_resample so that the target column balance
    # Step11: Return X_train, X_test, y_train, y_test
    # ----------------------------------------------------------------------------
    '''
    # Step1: Drop duplicated values
    df = df.drop_duplicates()

    # Step2: Feature Engineering and Feature Selection (Optional)

    # Step3: Split the data into X and y
    X = df.drop(columns = ['customerID', 'Churn'])
    y = df['Churn']

    # Step4 Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.3, 
                                                        random_state = 1)

    # Step5 Segregate the data into categorical and numerical columns
    numerical_cols = X_train.select_dtypes(exclude = 'object').columns
    categorical_cols = X_train.select_dtypes(include = 'object').columns

    # Step6: Build pipelines for numerical_pipeline: imputation and scaling
    numerical_pipeline = Pipeline([('imputation', SimpleImputer(strategy = 'median')),
                                  ('scaling', MinMaxScaler())])

    # Step7: Build pipelines for categorical_pipeline: imputation and encoding
    categorical_pipeline = Pipeline([('imputation', SimpleImputer(strategy = 'most_frequent')),
                                     ('encoding', OneHotEncoder(handle_unknown = 'ignore'))])

    # Step8: Concat numerical and categorical pipelines using ColumnTransformer
    transformer = ColumnTransformer([('numerical_pipeline', numerical_pipeline, numerical_cols),
                                     ('categorical', categorical_pipeline, categorical_cols)
    ])

    # Step9: fit the transformer on train and test data
    X_train = transformer.fit_transform(X_train)  # Seen data
    X_test = transformer.transform(X_test)        # Unseen data

    # Step10: Use target column balanace technique
    '''
    * Generally used with classification model
    * target column balance technique basically classified into two...

        - over_sampling technique

        - under_sampling technique
    '''
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    # Step11: Return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test