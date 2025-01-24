import common

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import box
import random

ROW_ID_COL_NAME = "rowID"
CLIENT_ID_COL_NAME = "clientID"
EASTING_COL_NAME="X"
NORTHING_COL_NAME="Y"
#spatially clusters dataset and returns a list of dataframes, one for each cluster, containing samples for each cluster
def spatialSplit(df,numClusters):
	#list of resulting DF containing samples of their associated cluster
	res =[]
	
	clusterIxLists = _spatialSplit(df,numClusters)
	#for each cluster extract a subset of the samples as dataframes
	#and retuern dataframes for each cluster
	for cid in range(numClusters):
		clusteredDF= df.loc[clusterIxLists[cid]]
		res.append(clusteredDF)
		
	
	return res

#spatially clusters dataset and returns a of list indices of samples in each cluster, one for each cluster, containing samples for each cluster
def _spatialSplit(df,numClusters):	
	#split a dataframe that has X,Y coordinates into nCluster clusters by applying kmeans clustering
	
	coords = np.array([df[EASTING_COL_NAME],df[NORTHING_COL_NAME]])
	coords = np.array(np.transpose(coords, (1,0))) #make the first dimension be samples, and 2nd be the column of coordinates
	
	#cluster the dataset by coordinates
	clusteredDS = KMeans(n_clusters=numClusters, n_init="auto").fit(coords)
	
	clusterIxLists =[]
	for cid in range(numClusters):
		clusterIxLists.append([]) #empty list of indices for each cluster
	
	#iterate over each sample 
	for i in range(len(df.index)):
		
		#place the ith sample's index in appropriate cluster index list
		clusterIx =clusteredDS.labels_[i]
		
		clusterIxList = clusterIxLists[clusterIx]
		
		clusterIxList.append(i)		
	

	return clusterIxLists



#applies block/spatial cross-validation to split the dataset into spatially disjoint folds in parralel,
#where  sub-folds/inner-folds are created for each client and then merged together
	
#the dataset is first split at a macro-level (client level) into clusters (e.g., each agriculture field would be a macro-level group)
#then  each group is clustered into nFolds, and the folds are merged beteen each client to make sure stratified sampling is
#used so each client has data in each fold
#returns folds with sample indices in each one	
def _nestedSpatialClusteredSampling(df,nFolds):

	if not CLIENT_ID_COL_NAME in df:
		raise Exception("expected client id column named '"+CLIENT_ID_COL_NAME+"' when clustering samples of each client. Missing column.")
		
	df = df.copy(deep=True)
	df.reset_index(drop=True, inplace=True)	
	
	
	#add row id to keep track of sample's index in original dataset
	ids = []
	for i in range(len(df.index)):
		ids.append(i)
	
	df.insert(0, ROW_ID_COL_NAME, ids, True) #add as first column
	
	
	
	
	clientIdList=findListOfClientIDs(df)
	
	clientFoldIxs =[] #lists of indices for each fold of client cid
	
	#for each client we cluster their dataset into nFolds 
	for	cid in clientIdList: #double check this syntax is fine. what i mean is for	cid in clientIxMap.keys():
		
		
		
		#subset of dataframe only belonging to client cid (.loc uses a list of bools to extract index and the np.array == cid expression flags all samples belinging to client cid)
		clientDF = df.loc[np.array(df[CLIENT_ID_COL_NAME]) == cid]
		clientDF.reset_index(drop=True, inplace=True)
		
		
		#spatially cluster the client's dataset into nFolds (the indices in clusters are local)
		localIxClusters = _spatialSplit(clientDF,nFolds)	
		
		globalIxClusters=[]
		#convert the local indices to global indices
		for localIxCluster in localIxClusters:
			globalCluster = []
			
			#iterate every local index in fold/cluster
			for localIx in localIxCluster: 
				
				#convert local index to global index
				globalIx = clientDF[ROW_ID_COL_NAME][localIx]
				globalCluster.append(globalIx)
			
			globalIxClusters.append(globalCluster)
			
		
		#track the list of indices of each clueters for each client
		clientFoldIxs.append(globalIxClusters)
	
	
	#shuffle order of clusters  for each client
	for clusters in clientFoldIxs:		
		random.shuffle(clusters)				
			
	#merge the clusters such that fold i of each client are merged together into a global fold i
	folds = []
	
	#empty list of samples indices for each fold
	for i  in range(nFolds):
		folds.append([])
		
	#iterate over each client's inner folds/clusters
	for cix in range(len(clientFoldIxs)):
		
		clusters = clientFoldIxs[cix]
		#iterate over each cliednt's cluster
		for foldIx  in range(len(clusters)):
			
			cluster = clusters[foldIx]
			
			#put all samples indices of cluster into the global fold foldIx
			common.listAppend(folds[foldIx],cluster)			
				
	#shuffle sample indices in each fold
	for fold in folds:
		random.shuffle(fold)
		
	#convert the samples index list to dataframes
	
	return folds
#spatially nested split returns folds as dataframes
def nestedSpatialClusteredSampling(df,nFolds):
	
	dfCpy = df.copy(deep=True)#make a copy so we affect anything. 
	
	resDFList = []
	
	folds = _nestedSpatialClusteredSampling(dfCpy,nFolds)
	
	for fold in folds:
		resDF = df.iloc[fold]
		resDFList.append(resDF)
		
	return resDFList

#searches the CLIENT_ID_COL_NAME colunm in dataframe for client ids
#and returns a list of unique client ids
def findListOfClientIDs(df):

	#no client id colun in dataframe?
	if not CLIENT_ID_COL_NAME in df:
		return []
		
	#used to track new client ids
	clientNumSamplesMap={}

	#tracks index of client
	clientIdList=[]
	
	#iterate every sample in the dataset
	for i in range(len(df.index)):
		
		#client id of sample i
		cid = df[CLIENT_ID_COL_NAME][i]
		
		#new client id?
		if not cid in clientNumSamplesMap:
			
			#new client found (value is a placeholder, the exists of the key is the focus)
			clientNumSamplesMap[cid] = None 
			
			#log new client id
			clientIdList.append(cid)
	return clientIdList		
		
	
#df that has a column clientID to indicate what client samples are from
#subset of df is taken by sampling a subset of each client
def clientSampler(df, numSamples):
	
	clientIdList=findListOfClientIDs(df)
	
	#used to track minimum number of samples found in a client
	minSamplesPerClient = len(df.index )+1 
	
	#iterate over every client 
	for	cid in clientIdList: 
		
		
		
		#flags indicating which sample belongs to client cid
		isClientSampleFlagList = np.array(df[CLIENT_ID_COL_NAME]) == cid
		
		clientSampleCount = 0
		
		for flag in isClientSampleFlagList:
			if flag:
				clientSampleCount = clientSampleCount +1
		
		#found the client with least number of samples?
		if clientSampleCount < minSamplesPerClient:
			minSamplesPerClient=clientSampleCount
		
		
	#can only sample up to max of minimum client's sample count 
	if numSamples>minSamplesPerClient:
		numSamples = minSamplesPerClient
	
	#sample without replacement each client so they have exactly numSamples instances
	df = df.groupby("clientID").sample(n=numSamples, replace=False) 
	df.sort_index(inplace=True)
	df.reset_index(drop=True, inplace=True)	
	return df

def findTileSampelsIndicesAroundPoint(geodf,sampleIx,tileSize,spatialRes):
	coords = geodf["geometry"][sampleIx]
	centerX=coords.x
	centerY=coords.y
	return _findTileSampelsIndicesAroundPoint(geodf["geometry"],centerX,centerY,tileSize,spatialRes)
	

def _findTileSampelsIndicesAroundPoint(geoPts,centerX,centerY,tileSize,spatialRes):
	tileHalfLength=spatialRes*tileSize/2.0	
	boundingbox = box(centerX-tileHalfLength,centerY-tileHalfLength,centerX+tileHalfLength,centerY+tileHalfLength,False) #fase means it startes from xmin-ymin and goes clockwise in search	
	indicesInTile= geoPts.sindex.query(boundingbox)	
	
	
	return indicesInTile 
