import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

creditData = pd.read_csv("credit_data.csv")

features = creditData[["income","age","loan"]]
target = creditData.default

model = LogisticRegression()
predicted = cross_val_predict(model,features,target, cv=10)

print(accuracy_score(target,predicted))