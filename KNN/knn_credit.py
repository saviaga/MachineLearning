import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split

data = pd.read_csv("credit_data.csv")

# Logistic regression accuracy: 93%
# we do better with knn: 98.5%


print(data.corr()) #print the correlation of features

data.features = data[["income","age","loan"]] #LTI-loan to income ratio
data.target = data.default #0 or 1

data.features = preprocessing.MinMaxScaler().fit_transform(data.features) #min-max normalization

feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.target, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=28)  # k value !!!
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

cross_valid_scores = []

for k in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=k) #k value
    scores = cross_val_score(knn,data.features,data.target,cv=10,scoring='accuracy')
    cross_valid_scores.append(scores.mean())
    

print("Optimal k with cross-validation: ", np.argmax(cross_valid_scores))    
    
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))