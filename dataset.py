import pandas as pd
import geopandas
import numpy as np
from datetime import datetime, timedelta
import os
import spatial

TIMESTAMP_COLUMN_NAME="timestamp"
SAMPLE_ID_EXTRA_COLUMN_NAME="sampleid"
CLIENT_ID_COL_NAME = "clientID"
EASTING_COL_NAME="X"
NORTHING_COL_NAME="Y"

NON_FEATURE_COLUMN_NAMES={SAMPLE_ID_EXTRA_COLUMN_NAME:None,CLIENT_ID_COL_NAME:None,EASTING_COL_NAME:None,NORTHING_COL_NAME:None}

#wrapper class to store the min and max of taraget variable, and 
#holds flag to determine if denormalization/unscaling is desired
class DataUnscaler:

	
	def __init__(self, minY,maxY,normalizeDataFlag):
		self.minY=minY
		self.maxY=maxY
		self.normalizeDataFlag=normalizeDataFlag
		
	def unscale(self,test_y):
		#should we apply denormalization to unscale data?
		if self.normalizeDataFlag:
			return denormalize(self.minY,self.maxY,test_y)
		else:
			return test_y
	
	
#compute number of timesteps in a tensor given dataset resolution (in minutes) and the
#tensor length in minutes
#def computeTensorTimeSteps(temporalResMins,tensorLenMins):
#	return (tensorLenMins/temporalResMins) +1
#note that this will be inintially called after DF normalized for CNN or LSTM
#and then in the folds we will take subset of the result of this function via the sampleid of the row for each index in
#fold pairs
#tensorNumTimeSteps: number of timesteps that a tensor will have
#temporalRes: resolution of dataset (in minutes)
#anySizeTensorFlag: when false, does not return tenors that have missing samples due to missing timestamps. true means return all any-sized tensors
#includeCoordInFeaturesFlag: flag indicating whether to include (True) the features in the resulting dataframes of predictor variables or not (False)
def createTensors(df,tileSize,spatialResolution,anySizeTensorFlag,includeCoordInFeaturesFlag):
	
	
	df =df.copy(deep=True)
	res=[]	
	coords=[]
	y=[] #predictor variable
	tileSize=int(tileSize)
	
	numberSampelsPerTile = tileSize * tileSize
	geodf = geopandas.GeoDataFrame(df,geometry=geopandas.points_from_xy(df[EASTING_COL_NAME],df[NORTHING_COL_NAME]))
	
	if not includeCoordInFeaturesFlag:
		#remove coordinates from  non-geo dataframe
		df.drop([EASTING_COL_NAME,NORTHING_COL_NAME],axis=1,inplace=True)
	
	finishedFlag=False
	targetVarCol = df.columns[-1]
	i=0
	
	for i in range(len(df.index)):

		#wStart = df.index[i] #starting of the window is first timestamp of dataset
		#wEnd = wStart+ timedelta(hours=tensorLenthHours)#ending of the window
	
		#get all samples within time window [tStart,tEnd) in sensor data				
		#get an index of all samples withint lag window
		#windowFilter = (df.index>= wStart) & (df.index< wEnd)
		
		
		#get all the samples that fall inside the tile around sample i
		tileSampleIxs = spatial.findTileSampelsIndicesAroundPoint(geodf,i,tileSize,spatialResolution)
		
		#get the samples in the tile
		subset = df.loc[tileSampleIxs]
		
		#any size tile returned
		if anySizeTensorFlag:
			res.append(subset)
			
			#keep track of center coordinates of each tile
			coords.append((geodf["geometry"][i].x,geodf["geometry"][i].y))
			y.append(df[targetVarCol][i])
			if len(subset.index) > numberSampelsPerTile:
				myio.logWrite("Expected tile size of "+str(numberSampelsPerTile)+", but tile with "+str(len(subset.index))+" samples returned. Look closely at the given spatial resolution and tile size, and the actual resolution of input dataset.",myio.LOG_LEVEL_WARNING)		
				
				
		else:
			#only include tensors of expected size. Skip tensors that had missing timestamps
			if len(subset.index) == numberSampelsPerTile:
				res.append(subset)
				
				#keep track of center coordinates of each tile
				coords.append((geodf["geometry"][i].x,geodf["geometry"][i].y))
				y.append(df[targetVarCol][i])
				
			elif len(subset.index) > numberSampelsPerTile:
				myio.logWrite("Expected tile size of "+str(numberSampelsPerTile)+", but tile with "+str(len(subset.index))+" samples returned. Look closely at the given spatial resolution and tile size, and the actual resolution of input dataset.",myio.LOG_LEVEL_WARNING)						
				
		
	return res,y,np.array(coords)
	
#given an array of indices, only extracts the tensors of those indices from df
def createTensorSubset(df,tileSize,spatialRes,indices):

	#print("creating tiles for dataframe: "+str(df))
	#print("tile size: "+str(tileSize))
	#print("spatial resolution: "+str(spatialRes))
	tiles,y,coords =createTensors(df,tileSize,spatialRes,anySizeTensorFlag=True,includeCoordInFeaturesFlag=True)
	
	#print("num tiles = "+str(len(tiles)))
	#E.g.: 30 min (0.5 h) resolution  dataset with tensors size 2 h = 2 / 0.5 + 1 = 5. +1 since the anchor teimstamp also included
	#so like 11:00,11:30,12:00,12:30,13:00 would be 2 h window with 30 min resolution
	tileSize = int(tileSize) #pixels are discrete, so tiles cannot be floats
	numSamplesPerTile=tileSize*tileSize
	res = []
	resCoords=[]
	resY=[]
	for i in indices:
		#skip out of bounds indices
		if i < 0 or i >= len(tiles):
			continue
		tile = tiles[i]
		
		#only consider tiles of desired index, and only those of appropriate size
		if len(tile.index) == numSamplesPerTile:
			res.append(tile)
			resCoords.append(coords[i])
			resY.append(y[i])
	
	return res,np.array(resY),np.array(resCoords)

	
#dataset is read into memory, filtering out non-selected features, and adding a sampling id column
#inputDatasetPath: file path to dataset with timestamps and a target variable to prepare into ML model friendly input format
#selectedFeaturesPath: file path to CSV that has entries for which column is selected (only 1 row, 1 indicating selected, 0 indicatign not)
def readDataset(inputDatasetPath,selectedFeaturesPath):
	
	if not os.path.exists(inputDatasetPath):
		raise Exception("Input dataset file "+inputDatasetPath+" does not exist.") 
		
	if not os.path.exists(selectedFeaturesPath):
		raise Exception("Selected features dataset file "+selectedFeaturesPath+" does not exist.") 
		
	#read the datsaets
	inDF = pd.read_csv(inputDatasetPath, sep=",")
	selectedFeatDF=pd.read_csv(selectedFeaturesPath, sep=",")
	
	
	#quality checks: make sure the column names align exactly
	if inDF.columns[0] != CLIENT_ID_COL_NAME:
		raise Exception("Input dataset expected '"+CLIENT_ID_COL_NAME+"' as first column: "+str(inputDatasetPath))
	if inDF.columns[1] != EASTING_COL_NAME:
		raise Exception("Input dataset expected '"+EASTING_COL_NAME+"' (easting) as 2nd column: "+str(inputDatasetPath))
	if inDF.columns[2] != NORTHING_COL_NAME:
		raise Exception("Input dataset expected '"+NORTHING_COL_NAME+"' (easting) as 3rd column: "+str(inputDatasetPath))
	
	
	#make sure columns align with both datsets (ignoring clikent id, X, and Y column (hence +3) and target variable (last colmn)
	for i in range (len(selectedFeatDF.columns)):
		if inDF.columns[i+3] != selectedFeatDF.columns[i]:#-1 since selecte feature dataframe doesn't have a timestamp
			raise Exception("Input dataset ("+inputDatasetPath+") and selected feature dataset ("+selectedFeaturesPath+") don't share the same feature columns ")
	
	#return a dataframe with only the selected sensors
	resMap={}
	
	#create sample id column
	sampleIds = []
	for i in range(len(inDF.index)):
		sampleIds.append(i)
	
	selectedFeatures=[]
	
	#sample id as 1st column, then timestamp as 2nd
	resMap[SAMPLE_ID_EXTRA_COLUMN_NAME]=sampleIds	
	resMap[CLIENT_ID_COL_NAME]=inDF[CLIENT_ID_COL_NAME]
	resMap[EASTING_COL_NAME]=inDF[EASTING_COL_NAME]
	resMap[NORTHING_COL_NAME]=inDF[NORTHING_COL_NAME]
	
	#only include selected features
	for col in selectedFeatDF.columns:
		#feature selected?
		if selectedFeatDF[col][0] == 1:
			resMap[col]=inDF[col]
			selectedFeatures.append(col)
	
	#add target variable to end 
	targetVarCol = inDF.columns[-1]
	resMap[targetVarCol]=inDF[targetVarCol]
	
	
	resDF=pd.DataFrame(data=resMap)
	
	#make sure every value exists (no missing values)
	for i in range(len(resDF.columns)):
		c = resDF.columns[i]		
		if np.any(np.isnan(resDF[c])):
			raise Exception("Missing value in column "+c+" of the selected feature data.")
	return resDF,selectedFeatures
	
#	return inDF

	
#min-max scaling
#returns deep copy of dataset with features and target variable normalized via min-max scaling
#where a target dataset is normalized (targetDF) using the min and max values of features in a source dataset (sourceDF)
#if sourceDF isn't specified, we use the min and max of the target dataset for scaling
#both targetDF and sourceDF should have same column names
#blackListCols: column names in this list aren't normalized
def normalize(targetDF,sourceDF=None):
	
	if not sourceDF is None:
		#integrity check. We expect both sets to have same columns (same number)
		if len(targetDF.columns) != len(sourceDF.columns):
			raise Exception("Cannot normlize datsets, the 2 datsets given don't have the same number of columns")
			
		#make sure all features have same name
		
		for i in range(len(targetDF.columns)-1): #we do -1 to ignore the last column, since its the target variable and may have a dffirent name
			if targetDF.columns[i] != sourceDF.columns[i]:
				raise Exception("Cannot normalize datsets, some of the column names do not match ("+targetDF.columns[i]+" vs. "+sourceDF.columns[i]+")")
			
		
	#avoid changign original dataframe by making a copy
	df =targetDF.copy(deep=True)
	
	#normalize data between 0 and 1 via min max scaling
	
	for i in range(len(df.columns)):	
		c = df.columns[i]
		
		#ignore blacklist column names
		if isNonFeatureColumnName(c):
			continue
		
			
		#iterate over each column
		#scale the column values between 0 and 1 via min max scaling
		if sourceDF is None: 
			minVal = df[c].min()
			maxVal = df[c].max()
		else:#scale columns by max and min of source dataset
			#use column name of sourceDF cause last target variable column might have a different name
			c2 = sourceDF.columns[i]
			minVal = sourceDF[c2].min()
			maxVal = sourceDF[c2].max()
			
		quotient =maxVal-minVal
		if quotient == 0:
			myio.logWrite("A feature ("+c+") has constant values, but is included anyway in the ML process. Consider removing it from the input dataset.",myio.LOG_LEVEL_WARNING)
			continue
		df[c]=(df[c]-minVal)/(quotient)


	return df
def denormalize(minVal,maxVal,normalizedVal):
	quotient =maxVal-minVal
	res =(normalizedVal*quotient)+minVal
	return res


def extractTrainTestValues(df,numNonFeatCols=2):

	df =df.copy(deep=True)
	
	
	#print("extracting feature matrix from columns: ")
	#print(str(df.columns[numNonFeatCols:-1]))
	
	#remove the non-feature columns (we assume their first)
	X=df.values[:,numNonFeatCols:-1]
	y=df.values[:,-1]
	
	X=np.asarray(X).astype(np.float32)
	y=np.asarray(y).astype(np.float32)
	return X,y		


#we take the sub indices pointing to sample/row indices in dfSubset, get the actual index in normDF
#via the sample id column, then extract as many fully-sized tensors as possible from normDF
#as feature, label (X,y) output
#shape of results is (samples,timesteps, features)
#temporalRes: temporeal resolution of dataset in minuts
def createTensorSets(normDF,dfSubset,dfSubsetIndices,tileSize,spatialRes):
	
	indices =[]
	#samplesIds=dfSubset[SAMPLE_ID_EXTRA_COLUMN_NAME]
	
	tmpDF =dfSubset.iloc[dfSubsetIndices]
	samplesIds=tmpDF[SAMPLE_ID_EXTRA_COLUMN_NAME]
	
	for ptrIx in samplesIds: #use the subset indices as pointer		
		ptrIx = int(ptrIx)
		
		indices.append(ptrIx)
	
	
	#now that we have locations of sample subset in the full dataset
	#with consecutive timestamps sampels, extract tensors
	#tensors,timestamps =createTensorSubset(normDF,tensorNumTimeSteps,temporalRes,indices)
	tiles,_y,coords =createTensorSubset(normDF,tileSize,spatialRes,indices)
	
	#print("Number of tiles: "+str(len(tiles)))
	
	X=[] #feature matrix
	y=[] #label list  #TODO: 
	for i in range(len(tiles)):
	#for t in tiles:
		tile = tiles[i]
		#coord = coords[i]
		
		tile =tile.copy(deep=True)
		#sort by coordinates so when reshaping it has appripriate 2D tile ordering
		tile.sort_values([EASTING_COL_NAME,NORTHING_COL_NAME],ascending=True,inplace=True)
		
		tX,_ =extractTrainTestValues(tile,4) #4 non feature columns (sample id and client id, X, Y), since coordinates column removed in the tensor extract process
		X.append(tX)
				
		#tiles have a single target variable value (center of tile)
		y.append(_y[i])
			
		
	#gotta now convert datset into 3d numpy arrays with 1 dimension being sample id and 2nd being the feature, and 3rd being the time
	return np.asarray(X),np.asarray(y),coords
	
#removes any row from the dataframe normDF that has missing samples in a  'tileSize' x 'tileSize' block around the row 
def removeUnfilledTensorTiles(normDF,tileSize,spatialResolution):
	
	#tileSize x tileSize tiles
	numberSamplesPerTile= tileSize * tileSize
	
	tiles,y,_ = createTensors(normDF,tileSize,spatialResolution,True,includeCoordInFeaturesFlag=False)
	rowIndicesToRemove = []
	#find rows that don't have complete tileSize x tileSize tiles of samples  around the target variable
	for  i in range(len(tiles)):
		tile = tiles[i]
		if len(tile.index) != numberSamplesPerTile:
			rowIndicesToRemove.append(i)
	
	if len(rowIndicesToRemove)>0:
		#remove the rows
		normDF.drop(rowIndicesToRemove,inplace=True)
		
		#so that the gaps from removed column removed and index is re-aligned to consecutive numbers
		normDF.reset_index(drop=True, inplace=True)

#returns flag indicating whether column name is a feature or not (e.g., 'X', the location column ins't a feature. )				
#without loss of generality of naming convention, the target varaible columns returned True by this function
def isNonFeatureColumnName(colName):
	
	if colName == None:
		return False
	return colName in NON_FEATURE_COLUMN_NAMES
	
def sampleDataFrame(df,nSamples):
	
	df =df.copy(deep=True)
	
	#don't sample?
	if nSamples<0:
		return df
		
	if df is None:
		return None
	
	if len(df.index)==0:
		return df
	
	#trying to sample more than exist in dataset?
	if nSamples > len(df.index):
		nSamples=len(df.index)
	
	
	
	#sample without replacement
	df = df.sample(n=nSamples, replace=False) 
	df.sort_index(inplace=True)
	df.reset_index(drop=True, inplace=True)	
	return df