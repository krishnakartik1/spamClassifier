from sklearn import datasets
iris = datasets.load_iris()
x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split 
xTrain, xTest,yTrain,yTest = train_test_split(x,y,test_size = 0.5)

from sklearn.neighbors import KNeighborsClassifier
myClassifier=KNeighborsClassifier()
myClassifier.fit(xTrain,yTrain)
predictions = myClassifier.predict(xTest)
print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(yTest,predictions)

