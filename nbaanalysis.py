import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def loadData(datafile):
    with open(datafile,"r", encoding = "latin1") as csvfile:
        data=pd.read_csv(csvfile)
        
        print(data.columns.values)
        
        return data
        
        
def runKNN(dataset, prediction, ignore):
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1, stratify =Y)
    
    
    knn = KNeighborsClassifier(n_neighbors=5)
    
               
    knn.fit(X_train, Y_train)

    score = knn.score(X_test, Y_test)
    
    print ("Predictions " +prediction+"with"+ str(score)+" accuracy")
    print ("Chance is: "+ str(1.0/len(dataset.groupby(prediction))))
    
    return knn
    
def classifyPlayer(targetRow, data, model,  prediction, ignore):
    X = targetRow.drop (columns=[prediction, ignore])

    neighbors = model.kneighbors(X, n_neighbors = 5,return_distance = False)
    
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])
nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData,"pos", "player")
classifyPlayer(nbaData.loc[nbaData['player']=="Harold Chang"], nbaData, knnModel, "pos", "player")