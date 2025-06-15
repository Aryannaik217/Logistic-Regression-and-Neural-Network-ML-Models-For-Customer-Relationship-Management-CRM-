import pandas as pd
import numpy as np

df=pd.read_csv("Customer_Churn.csv")

df.head()

df.info

df.dtypes

df.describe()

df.isnull().sum()

df.duplicated().sum()

df.dropna(how="any",inplace=True)

df.shape

df.isnull().sum()

df.columns.values

df.columns

df.tail

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')

df.isnull().sum()

df.dropna(how="any",inplace=True)

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Churn'] = encoder.fit_transform(df['Churn'])
Churn = {index : label for index, label in enumerate(encoder.classes_)}
Churn

df['PaymentMethod'] = encoder.fit_transform(df['PaymentMethod'])
PaymentMethod = {index : label for index, label in enumerate(encoder.classes_)}
PaymentMethod

df['customerID'] = encoder.fit_transform(df['customerID'])
customerID = {index : label for index, label in enumerate(encoder.classes_)}
customerID

df['gender'] = encoder.fit_transform(df['gender'])
gender = {index : label for index, label in enumerate(encoder.classes_)}
gender

df['Partner'] = encoder.fit_transform(df['Partner'])
Partner = {index : label for index, label in enumerate(encoder.classes_)}
Partner

df['Dependents'] = encoder.fit_transform(df['Dependents'])
Dependents = {index : label for index, label in enumerate(encoder.classes_)}
Dependents

df['PhoneService'] = encoder.fit_transform(df['PhoneService'])
PhoneService = {index : label for index, label in enumerate(encoder.classes_)}
PhoneService

df['MultipleLines'] = encoder.fit_transform(df['MultipleLines'])
MultipleLines = {index : label for index, label in enumerate(encoder.classes_)}
MultipleLines

df['InternetService'] = encoder.fit_transform(df['InternetService'])
InternetService = {index : label for index, label in enumerate(encoder.classes_)}
InternetService

df['OnlineSecurity'] = encoder.fit_transform(df['OnlineSecurity'])
OnlineSecurity = {index : label for index, label in enumerate(encoder.classes_)}
OnlineSecurity

df['OnlineBackup'] = encoder.fit_transform(df['OnlineSecurity'])
OnlineSecurity = {index : label for index, label in enumerate(encoder.classes_)}
OnlineSecurity

df['DeviceProtection'] = encoder.fit_transform(df['DeviceProtection'])
DeviceProtection = {index : label for index, label in enumerate(encoder.classes_)}
DeviceProtection

df['TechSupport'] = encoder.fit_transform(df['TechSupport'])
TechSupport = {index : label for index, label in enumerate(encoder.classes_)}
TechSupport

df['StreamingTV'] = encoder.fit_transform(df['StreamingTV'])
StreamingTV = {index : label for index, label in enumerate(encoder.classes_)}
StreamingTV

df['StreamingMovies'] = encoder.fit_transform(df['StreamingMovies'])
StreamingMovies = {index : label for index, label in enumerate(encoder.classes_)}
StreamingMovies

df['Contract'] = encoder.fit_transform(df['Contract'])
Contract = {index : label for index, label in enumerate(encoder.classes_)}
Contract

df['PaperlessBilling'] = encoder.fit_transform(df['PaperlessBilling'])
PaperlessBilling = {index : label for index, label in enumerate(encoder.classes_)}
PaperlessBilling

y = df['Churn'].values
X = df.drop(columns = ['Churn'])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)

#showing data
print('x \n' , X[:10])
print('y \n' , y[:10])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=30,random_state=0)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# logistic regression model using PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)

# Instantiate the model
input_dim = X_train.shape[1]
output_dim = len(np.unique(y_train))
model = LogisticRegressionModel(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluating the model
model.eval()
with torch.no_grad():
    train_accuracy = (model(X_train_tensor).argmax(dim=1) == y_train_tensor).float().mean()
    test_accuracy = (model(X_test_tensor).argmax(dim=1) == y_test_tensor).float().mean()

    print('LogisticRegressionModel Train Accuracy is : ', train_accuracy.item())
    print('LogisticRegressionModel Test Accuracy is : ', test_accuracy.item())
    print('LogisticRegressionModel Classes are : ', np.unique(y_train))
    print('----------------------------------------------------')

    # Calculating Prediction
    y_pred = model(X_test_tensor).argmax(dim=1)
    y_pred_prob = model(X_test_tensor)

    print('Predicted Value for LogisticRegressionModel is : ', y_pred[:10].numpy())
    print('Prediction Probabilities Value for LogisticRegressionModel is : ', y_pred_prob[:10].numpy())

""" **Model accuracy**"""

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting Accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x=['Train Accuracy', 'Test Accuracy'], y=[train_accuracy.item(), test_accuracy.item()])
plt.title('Train and Test Accuracy of Logistic Regression Model')
plt.ylabel('Accuracy')
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import numpy as np

conf_matrix = confusion_matrix(y_test_tensor.numpy(), y_pred.numpy())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test_tensor.numpy(), y_pred_prob.numpy()[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have already executed the code provided in the question to prepare the data and train the Logistic Regression model

# Create a DataFrame for the test data
df_test = pd.DataFrame(X_test, columns=df.columns[:-1])
df_test['Churn'] = y_test

# Visualizing Churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=df_test)
plt.title('Churn Distribution')
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob[:,1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()