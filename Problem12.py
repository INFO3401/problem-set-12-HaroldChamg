import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans


#Worked with Steve, Andrew, Luke, Anastasiya


def loadData(dataFile):
	with open(dataFile, 'r', encoding='latin1') as csvFile:
		data = pd.read_csv(csvFile)

	#Inspect data
	print(data.columns.values)
	return data

def runKNN(dataset, prediction, ignore, neighbors):
	
	X = dataset.drop(columns=[prediction, ignore])
	Y = dataset[prediction].values

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1, stratify = Y)


	knn = KNeighborsClassifier(n_neighbors = neighbors)


	knn.fit(X_train, Y_train)

	
	score = knn.score(X_test, Y_test)
	#Y_predict = knn.predict(X_test)
	#F1 = f1_score(Y_test, Y_predict, average = 'macro')
	print("Predicts " + prediction + " with " + str(score) + " accuracy")
	print('Chance is ' + str(1.0/len(dataset.groupby(prediction))))
	#print('F1: ', F1)

	return knn
    


    
    
def classifyPlayer(targetRow, dataset, model, prediction, ignore):
	X = dataset.drop(columns=[prediction, ignore])

	neighbors = model.kneighbors(X, n_neighbors = 5, return_distance = False)


	for neighbor in neighbors[0]:
		print(dataset.iloc[neighbor])
    
def runKNNCrossfold(dataset, kVal, neighbors):
	dataset = dataset.sample(frac = 1)
	test_value = dataset.shape[0]//kVal 
	start = 0
	end = test_value

	for k in range(kVal):
		test = dataset.iloc[start:end]
		train1 = dataset.iloc[0:start]
		train2 = dataset.iloc[end:]
		start += test_value
		end += test_value
		train = pd.concat([train1, train2])


		X_train = train.drop(columns=['pos', 'player'])
		Y_train = train['pos'].values

		X_test = test.drop(columns=['pos', 'player'])
		Y_test = test['pos'].values

		
	
		knn = KNeighborsClassifier(n_neighbors = neighbors)

		knn.fit(X_train, Y_train)

		score = knn.score(X_test, Y_test)
		print('Fold {} of {} predicts position with an accuracy of {}'.format(k+1, kVal, score))

        
        
def determineK(dataset):
	scores = {}

	for val in range(2,11):
		dataset = dataset.sample(frac = 1)
		test_value = dataset.shape[0]//5 
		start = 0
		end = test_value
		k_scores = []
		print('val: ', val)

		for k in range(5):
			test = dataset.iloc[start:end]
			train1 = dataset.iloc[0:start]
			train2 = dataset.iloc[end:]
			start += test_value
			end += test_value
			train = pd.concat([train1, train2])

		
			X_train = train.drop(columns=['pos', 'player'])
			Y_train = train['pos'].values

			X_test = test.drop(columns=['pos', 'player'])
			Y_test = test['pos'].values

		
			knn = KNeighborsClassifier(n_neighbors = val)

			knn.fit(X_train, Y_train)

			score = knn.score(X_test, Y_test)
			k_scores.append(score)
			print('Fold {} of {} predicts position with an accuracy of {}'.format(k+1, 5, score))

		scores[val] = sum(k_scores)/len(k_scores)
		print(scores[val])
	max_accuracy = max(scores.values())
	
	for k in scores.keys():
		if scores[k] == max_accuracy:
			print(k)
			maxK = k
	print(maxK, max_accuracy)
	print(scores)
    
    
def runKMeans(dataset, ignore):
    dropped= dataset.drop(columns = ignore)
    
    kmeans = KMeans(n_clusters = 5)
    
    kmeans.fit(dropped)
	
    dataset['cluster'] = pd.Series(kmeans.predict(dropped), index = dataset.index)
	
    scatterMatrix = sns.pairplot(dataset.drop(columns = ignore), hue = 'cluster', palette = 'Set2')
	
    scatterMatrix.savefig('kmeansClusters.png')
    return kmeans


def findClusterK(dataset, ignore):
    mean_distances = {}
    X = dataset.drop(columns=ignore)
    for n in np.arange(4,12):
        model = runKMeans(dataset, ignore, n) 
        mean_distances[n] = np.mean([np.min(x) for x in model.transform(X)])

    print("Best k by average distance: " + str(min(mean_distances, key=mean_distances.get)))
    

nbaData = loadData('nba_2013_clean.csv')
knnModel = runKNN(nbaData, 'pos', 'player', 5)
print('')
print('')
    
    
print('Classifying player:')
classifyPlayer(nbaData.loc[nbaData['player']=='Kobe Bryant'], nbaData, knnModel, 'pos', 'player')
print('')
print('')

print('Running crossfold validation:')
for kval in [5, 7, 10]:
	runKNNCrossfold(nbaData, kval, 5)
print('')
print('')

print('Determine K:')
determineK(nbaData)
print('')
print('')


#print('Finding the best number of clusters')
#converge = findClusterK(nbaData, ['pos', 'player'])

#Problem 3: The results tell me that kn is not ver reliable becuz they accuracy is below 50%



    
