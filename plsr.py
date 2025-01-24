#code from following  tutorial https://nirpyresearch.com/variable-selection-method-pls-python/
from sys import stdout
 
import pandas as pd
import numpy as np

import sys
from datetime import datetime
 
from scipy.signal import savgol_filter
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import random
import argparse

import math
import spatial 
#nSamplesPerClient: number of samples to samples from clients to make sure each client has at most nSamplesPerClient samples, or -1 to include all samples
def runFeatureSelection(inPath,outResPath,outCoeffPath,trials,nFolds,nSamplesPerClient):
	
	random.seed(123) #always sample from dataset and search for hyperparameters the same way for reproducibility
	
	start_time = datetime.now()

	start_timeStr = start_time.strftime("%Y-%m-%d %H:%M:%S")
	print("starting feature selection of "+inPath+" at "+start_timeStr)
	
		
	
	
		
	data,clientID,xCoord,yCoord,X,y =readDataset(inPath,nSamplesPerClient)
	
	#pls_variable_selection(xCoord,yCoord,X, y, trials,nFolds=5):
	#dataSubset,numFeatures,mse,r2,sorted_ind,trials= pls_variable_selection(X, y, trials)
	#bestNumberFeatures,bestMSE,bestR2, sorted_ind,coeffs,trials=pls_variable_selection(t,X, y, trials,temporalCVFlag,nFolds,blockClusterSize)
	bestNumberFeatures,bestMSE,bestR2, sorted_ind,coeffs,trials=pls_variable_selection(clientID,xCoord,yCoord,X, y, trials,nFolds)
	
	predVarCols = data.columns[3:-1] #predictor variable col names (ignore clientid,eastin/northing coordinates as first 3 columns, and target variable is last column)
	
	#create map that indicates if a column is selected
	selectedColNameMap={}
	for i in range(len(sorted_ind)):
		
		colIx = sorted_ind[i]
		colName = predVarCols[colIx]
		#column selected?
		if i <bestNumberFeatures:
		
			selectedColNameMap[colName]=True
		else:
			#not selected
			selectedColNameMap[colName]=False
			

	end_time = datetime.now()

	end_timeStr = end_time.strftime("%Y-%m-%d %H:%M:%S")	
	
	secondsPerHour=60.0*60.0
	
	timeDiff=end_time-start_time
	
	ellapsedTimeHours = float(timeDiff.seconds)/secondsPerHour
	
	#RESCALE the MSE back to orignal scale
	targetVarCol = data[data.columns[-1]] #target variable is last column
	std = targetVarCol.std()
	m = targetVarCol.mean()	
		
	if std== 0:
		std= 0.000001 #avoid divisions by 0
	bestMSE = (bestMSE*std)+m
	#write following to outPath
	#year, input dataset name, start time,end time, total duration (hours),trials,numberSelectedFeatures,MSE,R2,feature1,feature2,feature3,...featuren
	#,where year is growth season year, input dataset name is a friendly name of input dataset (e.g., no wind sensor) featurei is flag if feature i is selected, and 
	#trials is number of different random hyperparameters (number of components) we try
	
	
	header="start time,end time,total duration (hours),# trials,number selected features,number of folds, number of samples per client,mean of target variable, standard deviation of target variable,MSE,RMSE,R2"
	#append all non-location, non client id, and non-target vairable column names to header
	for cName in predVarCols:
		header = header +","+cName
	
	header = header + "\n"
	
	bestRMSE = math.sqrt(bestMSE)
	resultRow= start_timeStr+","+end_timeStr+","+str(ellapsedTimeHours)+","+str(trials)+","+str(bestNumberFeatures)+","+str(nFolds)+","+str(nSamplesPerClient)+","+str(m)+","+str(std)+","+str(bestMSE)+","+str(bestRMSE)+","+str(bestR2)
	
	#add indication of 1 when column selected, 0 when not selected
	for cName in predVarCols:
		#column selected?
		if selectedColNameMap[cName]:
			resultRow = resultRow +",1"
		else:
			resultRow = resultRow +",0"
	
	
	
	#write results to file
	with open(outResPath,"w") as file:
		file.write(header)
		file.write(resultRow)
		
	#we also write a 2nd file that has coefficients values for each column
	coeffHeader = ""
	for i  in range(len(predVarCols)):
		cName = predVarCols[i]
		if i ==0:
			coeffHeader =cName
		else:
			coeffHeader = coeffHeader + "," + cName
			
	coeffHeader = coeffHeader+"\n"
	coeffRow = ""
	#write coefficients of each column
	for i  in range(len(predVarCols)):		
		coeff = coeffs[i]
		coeff=coeff[0]
		if i ==0:
			coeffRow = str(coeff)
		else:
			coeffRow = coeffRow + "," + str(coeff)
		
	
	#write coeef results to file
	with open(outCoeffPath,"w") as file:
		file.write(coeffHeader)
		file.write(coeffRow)
		
	print("finished feature selection on "+inPath+" at "+end_timeStr+", writing results to "+outResPath+" and "+outCoeffPath)
	
def readDataset(inPath,nSamplesPerClient):
	rawDF = pd.read_csv(inPath)
	
	
	
	#make sure all the clients have same number of samples
	#to a avoid feature selection favboring one client
	unscaledSampledDF = spatial.clientSampler(rawDF,nSamplesPerClient)
	
	
	#feed a deep copy of sampled dataset to scale data so that original dataset isn't changed
	sampledDF = scaleData(unscaledSampledDF.copy(deep=True))
	
	clientID=sampledDF[spatial.CLIENT_ID_COL_NAME]
	
	xCoord,yCoord=sampledDF["X"],sampledDF["Y"]
	
	
	
	X=sampledDF.values[:,3:-1]#don't include clientID (index 0) the coordinates (index 1 and 2) and the target variable (last column, index -1)
	y=sampledDF.values[:,-1] #only the target variable data, column -1 (last column)	
	return unscaledSampledDF,clientID,xCoord,yCoord,X,y
def scaleData(X):

	for c in X.columns:	
		if c != spatial.CLIENT_ID_COL_NAME and c != 'X' and c != 'Y': #avoid  location data and client id being scaled
			#iterate over each column
			std = X[c].std()
			m = X[c].mean()
			
			if std== 0:
				std= 0.000001 #avoid divisions by 0
				
			#scale the column values by the standard deviation
			X[c]= (X[c]-m)/std
	return X
def pls_variable_selection(clientID,xCoord,yCoord,X, y, trials,nFolds=5,):

	#chosen max number coponents based on number of features
	numberSamples, numberFeatures = X.shape
	max_comp=numberFeatures-2

	if trials <0:
		trials =1
		print("number of trials provided to small: setting it to 1 trial")
	if trials > (max_comp):
		trials=max_comp-1
		print("number of trials provided to big: setting it to "+str(trials)+" trial")
		
	# Define MSE matrix to be populated where each row represents a trial and and columns are for features
	mse = np.zeros((trials,numberFeatures))
	r2 = np.zeros((trials,numberFeatures))
		

	#print("shape of X: "+str(X.shape))
	#random search to tune hyperparameter (number of components (up to max_comp))
	cList = []
	number_of_components_list = random.sample(range(1, max_comp), trials)
	
	totalNumIterations =0
	for i in range(trials):
		numComp= number_of_components_list[i]
		for j in range(numberFeatures-numComp):
			totalNumIterations=totalNumIterations+1
			
	iteration=0
	
	#print("fitting to X (pred vars)")
	#print(X)
	#print("target var y")
	#print(y)
	# Loop over the number of PLS components via random search
	for i in range(trials):
	
		numComp= number_of_components_list[i]
		
		# Regression with specified number of components, using full number of features
		pls1 = PLSRegression(n_components=numComp)
		pls1.fit(X, y)
		
		coeffs = np.transpose(pls1.coef_,(1,0))
		# Indices of sort features according to ascending absolute value of PLS coefficients
		#(so sorted_ind[0] is the index of smallest coefficient, so coeffs[sorted_ind] would be sorted coeffs small to big)
		sorted_ind = np.argsort(np.abs(coeffs[:,0]))

		# Sort features accordingly 
		Xc = X[:,sorted_ind]
		#print("shape of Xc: "+str(X.shape))
		# Discard one feature at a time of the sorted features (features with smallest coefficient (least influence) discarded first), where 
		#first run no features discarded, second run one feature discarded
		# regress, and calculate the MSE cross-validation
		for j in range(numberFeatures-numComp):
			
			pls2 = PLSRegression(n_components=numComp)
			pls2.fit(Xc[:, j:], y) #i don't think this is necessary, since cross-validation will fit the data and override it
			
			#(pls2,Xc,xCoord,yCoord,y,nFolds):
			#y_cv = makePredictions(pls2,Xc[:, j:],t,y,temporalCVFlag,nFolds,blockClusterSize)
			y_cv = makePredictions(pls2,Xc[:, j:],clientID,xCoord,yCoord,y,nFolds)
									
			mse[i,j] = mean_squared_error(y, y_cv)
			r2[i,j] = r2_score(y,y_cv)
			
			iteration = iteration+1
			comp = 100*iteration/(totalNumIterations)
			stdout.write("\r%d%% completed" % comp)
			stdout.flush()
	stdout.write("\n")

	# # Calculate and the (i,j) position  of minimum in MSE matrix
	#INDICES of non zero MSE ENTRIES
	mseNonZeroIx = np.nonzero(mse)
	#the array of non-zero mse
	mseNonZero = mse[mseNonZeroIx]
	
	#array of flags indicating if the element is equal to minimum MSE 
	minMSEFlags = mse==np.min(mseNonZero)
	
	#find index pairs of minimum MSE element
	min_mse_rowIndices,min_mse_colIndices = np.where(minMSEFlags)
	#a few mse may be equals, so many indices may be returned. Just pick first one, since I think its safe to assume this won't happen (a more correct solution would be to pick identiacal MSE wiehre R2 is highest, but we just want basic info, not intersted in really maximizing performance)
	
	#sometimes may have more than 1  run with best MSE
	#if so, find the run among them with highest R2
	min_mse_rowIx=min_mse_rowIndices[0]
	min_mse_colIx = min_mse_colIndices[0]
	bestR2=r2[min_mse_rowIx,min_mse_colIx]
	for i in min_mse_rowIndices:
		for j in min_mse_colIndices:
			if bestR2 < r2[i,j]:
				min_mse_rowIx=i
				min_mse_colIx = j
	
	bestMSE = mse[min_mse_rowIx,min_mse_colIx]
	bestR2 = r2[min_mse_rowIx,min_mse_colIx] #may not be best R2, but is r2 of the best mse run
	bestNumberFeatures = numberFeatures-min_mse_colIx
	
	bestNumberComponents =number_of_components_list[min_mse_rowIx]
	
	stdout.write("\n")	

	# Calculate PLS with optimal components and export values
	pls = PLSRegression(n_components=bestNumberComponents)
	pls.fit(X, y)
	coeffs2=np.transpose(pls.coef_,(1,0))
	sorted_ind = np.argsort(np.abs(coeffs2[:,0]))
	
	
	#numFeatures,mse,r2,sorted_ind,trials
	return (bestNumberFeatures,bestMSE,bestR2, np.flip(sorted_ind),coeffs2,trials) #flip the sorted indices, so first index is best feature's ix

  	

#pls2: partial leasat squares model 
#Xc: predictor variable matrix (features are columns)
#xCoord: eastin/x-coordinate column of same length as Xc's columns
#yCoord: northing/y-coordinate column of same length as Xc's columns
#y: target variable column
def makePredictions(pls2,Xc,clientID,xCoord,yCoord,y,nFolds):
	#requires resampling
	#requires random
	#requires math
	#doing corvss vaildation where folds are defined by temporal splits?
	

	#is numpyt array?
	if isinstance(Xc, np.ndarray):
		#convert to pandas
		Xc = pd.DataFrame(Xc)
	
	#is numpyt array?
	if isinstance(y, np.ndarray):
		#convert to pandas
		y = pd.DataFrame(y,columns=["targetVar"])
		
	#insert the location columns / coordinates and client id to dataframe
	Xc.insert(0, "Y", yCoord, True)
	Xc.insert(0, "X", xCoord, True)
	Xc.insert(0, spatial.CLIENT_ID_COL_NAME, clientID, True)
		
	
	#create row id column and append to Xc as first column
	#this is used to keep track of the order of the predicitons made
	#to align to expect values
	ids = []
	for i in range(len(xCoord)):
		ids.append(i)
	
	Xc.insert(0, "rowIDTmp", ids, True)
	
	#make sure append the target variable to the matrix so the shuffling keeps track of 
	#target var too
	Xc.insert(len(Xc.columns), "targetVar", y, True)
	
	#folds = resampling.temporalClusteredSampling(Xc,blockCVClusterSize,nFolds) # 24 hours (1 day) clusters
	folds = spatial.nestedSpatialClusteredSampling(Xc,nFolds)
	
	#folds = temporalClusteredSampling(Xc,24,nFolds) # 24 hours (1 day) clusters
	
	predResultsMap={}
	predResultsMap["rowIDTmp"]=[]
	predResultsMap["pred"]=[]
	for i in range(len(folds)):
		testDF= folds[i]
		
		trainDF = None
		#create train datafarame
		for j in range(len(folds)):
		
			if j==i:
				continue
			if trainDF is None:
				trainDF=folds[j]
			else:
				trainDF = pd.concat([trainDF,folds[j]],ignore_index=True)
		
		trainDF.reset_index(drop=True, inplace=True)
		testDF.reset_index(drop=True, inplace=True)
		
		trainX=trainDF.values[:,4:-1]#don't include the coordinates (index 2 and 3), nor the row id (index0), nor the client id (index 1),  the  nor the target variable (last column, index -1)
		trainY = trainDF.values[:,-1]#only the traget variable column last column)
		testX=testDF.values[:,4:-1]#don't include the coordinates (index 2 and 3), nor the row id (index0), nor the client id (index 1),  the  nor the target variable (last column, index -1)
		
		
		pls2.fit(trainX,trainY)			
		predY = pls2.predict(testX)
		
		#store results
		for rowIx in range(len(testDF["rowIDTmp"])):
			rowID = testDF["rowIDTmp"][rowIx]
			pred = predY[rowIx]
			predResultsMap["rowIDTmp"].append(rowID)
			predResultsMap["pred"].append(pred)		
	
	predDF = pd.DataFrame(data=predResultsMap)
			
	#sorte the predictions by row id	
	predDF = predDF.sort_values(by=['rowIDTmp'])
	return predDF["pred"]
	
		
class ZeroR:

	
	def __init__(self):
		self.meanTargetVar=0
		pass
		
	def fit(self, X,Y):
		for val in Y:
			self.meanTargetVar=self.meanTargetVar+val
		self.meanTargetVar = self.meanTargetVar/len(Y)
	def predict(self,X):
		res = []
		for i in range(len(X)):
			res.append(self.meanTargetVar)
		return res
	
	
#only run the below code if ran as script
if __name__ == '__main__':
		
	ap = argparse.ArgumentParser()
	
	
	ap.add_argument("-d", "--dataset", type=str, required=True,
		help="file path of the dataset")
		
	args = vars(ap.parse_args())
	
	
	inPath = str(args["dataset"])

	outResPath = "output/feature-selection/feature-selection-summary.csv"
	outCoeffPath = "output/feature-selection/plsr-coefficients.csv"
	
	trials=100 #100 different hyperparameter choices are used to search for best hyperparameter choice	
	nFolds=10 #number of folds	
	nSamplesPerClient=500# 500 random samples taken per client
	
	runFeatureSelection(inPath,outResPath,outCoeffPath,trials,nFolds,nSamplesPerClient)