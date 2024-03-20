import argparse
import pandas as pd
import tomli
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from pipeline import main
from sklearn.preprocessing import LabelEncoder
import json


def load_config(path) :
    '''
    Function to read the toml file that contains all configurations
    Args:
    path: directory path
    '''
    with open(path, 'rb') as f:
        return tomli.load(f)

def wrapper(df,feature_names, label_name, test_size,parent_dir,real_data_path):
    '''
    This function allows to use tabddpm in an easier way

    Args:
        df: Dataframe to create the difussion model 
        feature_names: This var should specify the features that we select to build the diffusion model
        label_name: We should specify the name of the label to build the difussion model
        test_size: specifies the testing size 
        parent_dir: this path should go to the results of the tabddpm data generation
        real_data_path: path were we will store all the input data necessary to run the program. In this path, we will create info.json, and numpy files with the data
    '''        
     
    raw_config =load_config(real_data_path+'/config.toml')  #raw config load
   
    # Code to execute if no exception occurred
    #importing csv file
    length_features_col=len(feature_names)  #length of the features
    features_original=df[feature_names]   #select the features according to file config    
    label_original=df[label_name]   #select the label according to file config
    
     
    num_classes = label_original.nunique()   #get number of classes
    task_type='multiclass'  #multiclass or binclass
    label_encoder = LabelEncoder()  #define label encoder class
    # Fit and transform the 'Category' column
    label_encoded_original = pd.DataFrame(label_encoder.fit_transform(label_original))  
    print("label enconded")
    correspondence = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(correspondence)

    # Check if the DataFrame contains categorical columns
    categorical_columns = features_original.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        raise ValueError("DataFrame contains categorical columns: {}".format(categorical_columns))
    else:
        #create .numpy files and info.json file

        #CREATE NUMPY FILES
        # Split in train, test and validation
        X_num_train, X_num_temp, y_train, y_temp = train_test_split(features_original, label_encoded_original , test_size=test_size, random_state=42)
     
        X_num_train_pd=pd.DataFrame( X_num_train)
       
        y_train_pd=pd.DataFrame(y_train)
        X_num_test, X_num_val, y_test, y_val = train_test_split(X_num_temp, y_temp, test_size=0.5, random_state=42)

        #create numpy files for the main program
        X_num_train_np =  X_num_train.to_numpy()
        X_num_test_np =  X_num_test.to_numpy()
        X_num_val_np =  X_num_val.to_numpy()

        y_train_np =  y_train.to_numpy()
        y_test_np =  y_test.to_numpy()
        y_val_np =  y_val.to_numpy()
        
        length_train=len(y_train_np)
        length_test=len(y_test_np)
        length_val=len(y_val_np)

        # Save the NumPy array to a .npy file
        np.save(real_data_path+'/X_num_train.npy', X_num_train_np)
        np.save(real_data_path+'/X_num_test.npy', X_num_test_np)
        np.save(real_data_path+'/X_num_val.npy', X_num_val_np)
        np.save(real_data_path+'/y_train.npy', y_train_np)
        np.save(real_data_path+'/y_test.npy', y_test_np)
        np.save(real_data_path+'/y_val.npy', y_val_np)

        #CREATE INFO.JSON  
        data = {
        "name": "Experiment",
        "id": "experiment-",
        "task_type": task_type,
        "n_num_features": length_features_col,
        "n_cat_features": 0,
        "train_size": length_train,
        "val_size": length_val,
        "test_size": length_test,
        "n_classes": num_classes
        }
        # Specify the file path
        file_path = real_data_path+'/info.json'

        # Write the JSON data to the file
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=2)
    
    #PASS CONFIGURATION FILE TO PIPELINE.PY MAIN PROGRAM
    num_samples_to_generate=1000
    main(raw_config,parent_dir,real_data_path,num_samples_to_generate)
    features_gen=pd.read_csv(parent_dir+'/X_gen.csv', index_col=0)
    label_gen=pd.read_csv(parent_dir+'/y_gen.csv')['0']
    verify_datasamples_rf(features_original,features_gen)
    
def verify_datasamples_rf(features_original,features_gen):
    '''
    This function helps to verify that the samples generated are similar to the original datasamples
    Args:
    features_original: original features of the dataset that we use to create augmented data
    features_gen: generated features with the difussion model

    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    features_gen['class_identifier']=1 #identify original features 
    features_original['class_identifier']=0 #identify original features 
    name_columns=features_original.columns  #get name columns of the original dataset
    features_gen.columns=name_columns #reaname the columns of the dataset generated to be the same of the original dataset
   
    all_dataset=pd.concat([features_gen,features_original])  #concatenate original and generated dataset
    print(all_dataset)

    y=all_dataset['class_identifier']   #extract labels from the dataset
    X=all_dataset.drop(columns=['class_identifier'])  #extract features from the dataset

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
 

    cm = confusion_matrix(y_test, y_pred)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(cm)
    report = classification_report(y_test, y_pred)

    # Print the classification report
    print("Classification Report:")
    print(report)

    print("asd")

        
        