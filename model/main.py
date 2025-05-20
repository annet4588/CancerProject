import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

# Train model
def create_model(data):
    # Divide the data in the predictors and the target variables
   X = data.drop(['diagnosis'], axis=1)
   y = data['diagnosis']
   
   # Scale the data
   scaler = StandardScaler() 
   # Scale all the training predictors
   X = scaler.fit_transform(X)
   
   # Split the data into testing and training sets
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   
   # Train model
   model = LogisticRegression()
   model.fit(X_train, y_train) # Fit model to our data. Train on the training part of the predictors
   
   # Test model
   y_pred = model.predict(X_test) # Test it on the testing part of the predictors
   print('Accuracy of our model: ', accuracy_score(y_test, y_pred)) # 2 params - actual and predicted
   print("Classification report: \n", classification_report(y_test, y_pred)) # more in-depth report with 2 params - actual and predicted
   
   # Once we trained the model - we return (return the scalar to use it to make predictions)
   return model, scaler

# Read the data from th file
def get_clean_data():
    data = pd.read_csv("data/data.csv")
   
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Encode the diagnosis column with map() function. Assign M to 1 and B to 0.
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})    
    return data
    
def main():   
    data = get_clean_data()
    # print(data.info())
    
    model, scaler = create_model(data)
   
    # Write binary file 'wb' with the module pickle
    with open('model/model.pkl', 'wb') as f:
       pickle.dump(model, f)
    
    # Same with scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f) 
if __name__ == '__main__':
  main()