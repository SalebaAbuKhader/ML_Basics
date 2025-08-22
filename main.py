#url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset from the URL

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

#define features and target variable

features = df[["total_bill","size"]]
target = df["tip"]

print("Features : \n ", features.head())
print("Target : \n", target.head())

# Split tha data into training and testing sets 

X_train , X_test , y_train , y_test = train_test_split(features , target , test_size = 0.2 , random_state = 42)

print("Training data set :\n", X_train.shape)
print("Testing data set :\n", X_test.shape)

#visualize the data 

sns.pairplot(df , x_vars = ["total_bill","size"] , y_vars ="tip", height = 5 , aspect = 0.8 , kind ="scatter")
plt.title("Feature vs Target Relationship")
plt.show()







