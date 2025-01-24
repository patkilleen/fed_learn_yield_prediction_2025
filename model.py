import traceback
import numpy as np

DEBUGGING_FLAG=False#set to false when run on linux

try:
	from sklearn.model_selection import ParameterSampler
	from sklearn.ensemble import RandomForestRegressor
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import BatchNormalization
	from tensorflow.keras.layers import LSTM	
	from tensorflow.keras.layers import Activation
	from tensorflow.keras.layers import Dropout
	from tensorflow.keras.layers import Dense
	from tensorflow.keras.layers import Flatten
	from tensorflow.keras.layers import Input
	from tensorflow.keras.models import Model
	from tensorflow.keras.layers import concatenate
	from tensorflow.keras.callbacks import EarlyStopping
	from tensorflow.keras.layers import Bidirectional
	from tensorflow.keras.layers import AveragePooling1D
	from tensorflow.keras.layers import Layer
	import tensorflow as tf
	from keras.initializers import RandomNormal
	
	from scipy.stats import loguniform	
	import keras_tuner as kt
	from tensorflow.keras.optimizers import Adam	
	from keras.layers import Flatten	
	from keras.layers.convolutional import Conv2D
	from keras.layers.convolutional import MaxPooling2D
	from keras.layers.convolutional import AveragePooling2D	
	from sklearn.linear_model import LinearRegression
	#from tensorflow.keras.optimizers.legacy import Adam
	#import tensorflow_decision_forests as tfdf
	#from keras import backend as K
except ImportError as e:
	#only consider it error when not debugging and package import failed
	if not DEBUGGING_FLAG:
		print("Failed to import packages: "+str(e)+"\n"+str(traceback.format_exc()))
		exit()

ALG_ZEROR="ZeroR"
ALG_RANDOM_FOREST="RF"
ALG_DEEP_NEURAL_NETWORK="DNN"
ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK="sCNN"
ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK="rCNN"
ALG_LINEAR_REGRESSION="LR"
deepLearningModelMap={ALG_ZEROR:False,\
						ALG_RANDOM_FOREST:False,\
						ALG_DEEP_NEURAL_NETWORK:True,\
						ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK:True,\
						ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK:True,\
						ALG_LINEAR_REGRESSION:False}
modelsWithTensorsMap={ALG_ZEROR:False,\
						ALG_RANDOM_FOREST:False,\
						ALG_DEEP_NEURAL_NETWORK:False,\
						ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK:True,\
						ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK:True,\
						ALG_LINEAR_REGRESSION:False}
						
#algorith,year,hyperparameter name
FLStaticHyperParameterMap={ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK:{},\
							ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK:{},\
							ALG_DEEP_NEURAL_NETWORK:{},\
							ALG_RANDOM_FOREST:{},\
							ALG_LINEAR_REGRESSION:{},\
							ALG_ZEROR:{}}							


MONO_TEMPORAL_DOY201_KEY="mono-temporal-DoY201"
MONO_TEMPORAL_DOY216_KEY="mono-temporal-DoY216"
MULTI_TEMPORAL_KEY="multi-temporal"

FLStaticHyperParameterMap[ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK][MONO_TEMPORAL_DOY201_KEY]={"FC_size":38,"activation":"sigmoid","batchSize":239,"early_stop_patience":3,"epochs":600,"filterSize":2,"learning_rate":0.00019274592475316,"nFeatureMaps":112}	
FLStaticHyperParameterMap[ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK][MONO_TEMPORAL_DOY216_KEY]={"FC_size":202,"activation":"tanh","batchSize":995,"early_stop_patience":15,"epochs":650,"filterSize":2,"learning_rate":0.000223005305995785,"nFeatureMaps":104}	
FLStaticHyperParameterMap[ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK][MULTI_TEMPORAL_KEY]={"FC_size":228,"activation":"sigmoid","batchSize":436,"early_stop_patience":9,"epochs":600,"filterSize":2,"learning_rate":0.0000720564216429465,"nFeatureMaps":130}	

FLStaticHyperParameterMap[ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK][MONO_TEMPORAL_DOY201_KEY]={"FC_size":170,"activation":"relu","batchSize":512,"early_stop_patience":12,"epochs":650,"filterSize":2,"learning_rate":0.00466727177696785,"nFeatureMaps":154}	
FLStaticHyperParameterMap[ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK][MONO_TEMPORAL_DOY216_KEY]={"FC_size":17,"activation":"sigmoid","batchSize":393,"early_stop_patience":18,"epochs":600,"filterSize":2,"learning_rate":0.00477609664864307,"nFeatureMaps":221}	
FLStaticHyperParameterMap[ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK][MULTI_TEMPORAL_KEY]={"FC_size":244,"activation":"relu","batchSize":510,"early_stop_patience":12,"epochs":500,"filterSize":2,"learning_rate":0.0041332917097823,"nFeatureMaps":215}	



FLStaticHyperParameterMap[ALG_DEEP_NEURAL_NETWORK][MONO_TEMPORAL_DOY201_KEY]={"FC0_size":153,"FC1_size":220,"FC2_size":169,"FC3_size":120,"FC4_size":211,"FC5_size":193,"activation":"relu","batchSize":356,"early_stop_patience":4,"epochs":550,"learning_rate":0.021689911453311,"number_of_layers":1}	
FLStaticHyperParameterMap[ALG_DEEP_NEURAL_NETWORK][MONO_TEMPORAL_DOY216_KEY]={"FC0_size":16,"FC1_size":74,"FC2_size":80,"FC3_size":232,"FC4_size":248,"FC5_size":93,"activation":"tanh","batchSize":1171,"early_stop_patience":16,"epochs":500,"learning_rate":0.0000317571395418609,"number_of_layers":3}	
FLStaticHyperParameterMap[ALG_DEEP_NEURAL_NETWORK][MULTI_TEMPORAL_KEY]={"FC0_size":40,"FC1_size":205,"FC2_size":135,"FC3_size":164,"FC4_size":35,"FC5_size":45,"activation":"relu","batchSize":495,"early_stop_patience":12,"epochs":650,"learning_rate":0.00102820887828938,"number_of_layers":2}	

FLStaticHyperParameterMap[ALG_RANDOM_FOREST][MONO_TEMPORAL_DOY201_KEY]={"n_estimators":325,"min_samples_leaf":1,"max_samples":0.9,"max_features":1,"max_depth":19,"bootstrap":True}	
FLStaticHyperParameterMap[ALG_RANDOM_FOREST][MONO_TEMPORAL_DOY216_KEY]={"n_estimators":175,"min_samples_leaf":7,"max_features":1,"max_depth":None,"bootstrap":False}	
FLStaticHyperParameterMap[ALG_RANDOM_FOREST][MULTI_TEMPORAL_KEY]={"n_estimators":375,"min_samples_leaf":2,"max_samples":0.65,"max_features":5,"max_depth":20,"bootstrap":True}	


FLStaticHyperParameterMap[ALG_LINEAR_REGRESSION][MONO_TEMPORAL_DOY201_KEY]={}	
FLStaticHyperParameterMap[ALG_LINEAR_REGRESSION][MONO_TEMPORAL_DOY216_KEY]={}	
FLStaticHyperParameterMap[ALG_LINEAR_REGRESSION][MULTI_TEMPORAL_KEY]={}	

FLStaticHyperParameterMap[ALG_ZEROR][MONO_TEMPORAL_DOY201_KEY]={}	
FLStaticHyperParameterMap[ALG_ZEROR][MONO_TEMPORAL_DOY216_KEY]={}	
FLStaticHyperParameterMap[ALG_ZEROR][MULTI_TEMPORAL_KEY]={}										
class Model:

	#inputShape: the dimensions of an imput sample. So for basic ML, first and only element states number of features.
	def __init__(self,algName,hyperParams,inputShape,model=None):
		self.algName = algName
		self.hyperParams = hyperParams
		self.fittedFlag=False #not trained yet
		self.inputShape = inputShape
		if not (model is None):
			if isinstance(model,Model):
				raise Exception("Expected a raw model but a model wrapper was provided to mymodel.Mode")
				
		if self.algName == ALG_ZEROR:
			self.predictionBuffer =[] #empty list of predictions to avoid re-creating a list each tiem
			self.model = None
		elif self.algName == ALG_RANDOM_FOREST:
			if not hyperParams["bootstrap"]:
				#sklearn RF can't hanve the 'max_samples' defien when bootstrap is false
				#hpSet["max_samples"]=None	
				if "max_samples" in hyperParams:
					hyperParams.pop("max_samples")
					
			if model is None:
				self.model =RandomForestRegressor(**hyperParams)
			else:
				self.model = model
		elif self.algName == ALG_DEEP_NEURAL_NETWORK:			
			
			nLayers =hyperParams["number_of_layers"]
					
			layerSizes = []
			for i in range(nLayers):
				layerSizes.append(hyperParams["FC"+str(i)+"_size"])
			
			if model is None:
				self.model = buildDNN(layerSizes,hyperParams["activation"],hyperParams["learning_rate"],inputShape)
			else:
				self.model = model				
		elif self.algName == ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK:
			if model is None:			
				self.model = buildSpatialCNN(hyperParams["nFeatureMaps"],hyperParams["filterSize"],hyperParams["FC_size"],hyperParams["activation"],hyperParams["learning_rate"],inputShape)
			else:
				self.model = model
		elif self.algName == ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK:
			if model is None:			
				self.model = buildSpectralCNN(hyperParams["nFeatureMaps"],hyperParams["filterSize"],hyperParams["FC_size"],hyperParams["activation"],hyperParams["learning_rate"],inputShape)
			else:
				self.model = model		
		elif self.algName == ALG_LINEAR_REGRESSION:
			if model is None:			
				self.model =LinearRegression()
			else:
				self.model = model
		else:
			raise Exception("Cannot create model. unknown algorithm "+str(algName))
			
		if isDeepLearningModel(self.algName):				
			self.epochs=hyperParams["epochs"]
			self.patience=hyperParams["early_stop_patience"]
			self.batchSize=hyperParams["batchSize"]
			
	#trains the model
	#X is the feature matrix
	#y is target variable list
	def fit(self,train_X,train_y,test_X,test_y):
		self.fittedFlag=True #training complet
		
		if self.algName == ALG_ZEROR:
			self.meanVal = train_y.mean()
		else:
			if isDeepLearningModel(self.algName):				
				stop_early = EarlyStopping(monitor='val_loss', patience=self.patience)
				self.model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=self.epochs,batch_size=self.batchSize,verbose = 0) #deep learning models use the epoch parameter
			else:
				self.model.fit(train_X,train_y) #basic ML models don't need test data for model fitting
		pass	
		
	#make predictions using trained model (fit must be called first)
	#X is matrix of features
	def predict(self,X):
		
		#can only predict after training the model
		if not self.fittedFlag:			
			raise Exception("Cannot make "+str(self.algName)+" model predictions before training the model (did you forget to call 'fit'?) ")
		
		
		if self.algName == ALG_ZEROR:
			self.predictionBuffer.clear()
			#iterate over each row
			for i in range(len(X)):				
				self.predictionBuffer.append(self.meanVal) #zeroR always predicts mean of target variable in training data
			return np.array(self.predictionBuffer)
		else:
			preds = self.model.predict(X)					
			
			if isDeepLearningModel(self.algName):
				#neural networks output an array of dimenison one for each sample, so 
				#reshape to 1D array (n samples) instead of 2D array  (1 x n samples)
				preds = preds.reshape(len(preds))
			return preds									
				
	def toString(self):
		resStr = "Algorithm: "+self.algName+", fitted: "+str(self.fittedFlag)+". Model: "+str(self.model)+". Hyperparameters: "
		for k in hyperParams:
			resStr = k+": "+str(hyperParams[k])+","				 				
		return resStr
	
#generates 'trials' sets of randomly chosen hyperparameter values using random search, 
def generateHyperparameterSets(algName,spatialResolution,inputTileSize,trials,inputShape,nTrainSamples,nTestSampels):
	
	if trials <=0:
		raise Exception("Expected positive number of random search trials, but received '"+str(trials)+"'")
		
	hyperParamSets=[]
	paramDict={}
	if algName == ALG_ZEROR:		
		for i in range(trials):
			hyperParamSets.append(None) #no hyperparameters for ZeroR		
	elif algName == ALG_RANDOM_FOREST:
		nFeatures=inputShape[0]
		paramDictValueRanges = getRandomForestHypParamSearchSpace(nFeatures)
		hyperParamSets = ParameterSampler(param_distributions =paramDictValueRanges,n_iter = trials)
							
	
	elif algName == ALG_DEEP_NEURAL_NETWORK:#DNN is a MLP with 2-6 hidden layers
		
		paramDict["number_of_layers"]=[]#number layers
		minNumLayers=1
		maxNumLayers=6
		maxLayerSize=256
		for i in  range(minNumLayers,maxNumLayers,1) :#1 to 6 layers
			paramDict["number_of_layers"].append(i)
		for l in range(maxNumLayers):
			layerLName= "FC"+str(l)+"_size"
			paramDict[layerLName]=[]#size of fully connected layer l
			for i in range(8,maxLayerSize,1) : #min layer size 8, steps by 1
				paramDict[layerLName].append(i)
		
		
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
	
	elif algName == ALG_SPATIAL_CONVOLUTIONAL_NEURAL_NETWORK:
		
		paramDict["nFeatureMaps"]=[]
		for i in  range(8,256,1):
			paramDict["nFeatureMaps"].append(i)
		
		paramDict["filterSize"]=[]
		for i in  range(2,inputTileSize,1):
			paramDict["filterSize"].append(i)
		
		
		
		paramDict["FC_size"]=[]
		for i in  range(8,256,1):
			paramDict["FC_size"].append(i)
					
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
	
	elif algName == ALG_SPECTRAL_CONVOLUTIONAL_NEURAL_NETWORK:
		
		paramDict["nFeatureMaps"]=[]
		for i in  range(8,256,1):
			paramDict["nFeatureMaps"].append(i)
		
		paramDict["filterSize"]=[]
		for i in  range(2,inputTileSize,1):
			paramDict["filterSize"].append(i)
		
		
		
		paramDict["FC_size"]=[]
		for i in  range(8,256,1):
			paramDict["FC_size"].append(i)
					
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
	
	elif algName ==ALG_LINEAR_REGRESSION:
		pass #no hyperparameters for linear regression
	if isDeepLearningModel(algName):
		
		paramDict["early_stop_patience"]=[]#patience
		for i in  range(3,20,1):
			paramDict["early_stop_patience"].append(i)
				
		paramDict["epochs"]=[]
		for i in  range(400,700,50):
			paramDict["epochs"].append(i)				
		
		
		maxBatchSize = int(round(nTrainSamples*0.9)) #batch size is 90% of size of training dataset
		minBatchSize = int(round(nTrainSamples*0.15))#batch size is 15% of size of training dataset
		
		
		paramDict["batchSize"]=[]#patience
		for i in  range(minBatchSize,maxBatchSize,1):
			paramDict["batchSize"].append(i)
				
		paramDict["activation"]=['relu','tanh','sigmoid']
				
		paramDict["learning_rate"]=loguniform(1e-5, 1e-1)
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
	return hyperParamSets



def getRandomForestHypParamSearchSpace(nFeatures):

#tfdf.keras.RandomForestModel(
	#hyper params
	#max_depth , default 16, max depth of tree
	#num_trees: the number of decision trees in forest, default 300
	#num_candidate_attributes: how many features used at a node split, default is sqrt(number of input attributes) 
	#bootstrap_size_ratio: default to 1 (100%), so ration of number of samples used to train a tree
	#min_examples: minimum number examples in a node. default 5
	
	paramDict = {}
	paramDict["n_estimators"]=[]#number of trees
	for i in  range(50,500,25) :
		paramDict["n_estimators"].append(i)
	
	paramDict["max_depth"]=[]
	for i in range(8,24,1): #number of trees
		paramDict["max_depth"].append(i)
		
	paramDict["max_depth"].append(None) # no max depth is possible too
	
	paramDict["min_samples_leaf"]=[]
	for i in range(1,15,1): #minimum number of samples to be a leaf
		paramDict["min_samples_leaf"].append(i)
	
	if nFeatures == 1:
		paramDict["max_features"]=[1]
	else:
		paramDict["max_features"]=[]
		for i in range(1,nFeatures,1): #max number of features condiered when splitting
			paramDict["max_features"].append(i)
			
	paramDict["bootstrap"]=[True,False] #whether bootstrap samples is used or not (false means 100% of data set used, when false, then the max_samples ratio of samples used)
	paramDict["max_samples"]=[]
	for i in range(65,95,5): #65, 70,75,...,95%
		paramDict["max_samples"].append(i/100.0) #make sure in form: 0.65, 0.7....
	#paramDict["random_state"]=[42] #we will let the model be different every time, cause it'll allow the multiple executions per trial to have different randomness
	
	
	return paramDict
	
#model: a sequential model that already had the number of layers added to it
def buildDNN(layerSizes,activation,learnRate,inputShape):
	model = Sequential()
	model.add(Input(shape=(inputShape[0])))
		
	nLayers=len(layerSizes)
	
	for i in range(nLayers):
		model.add(Dense(units=layerSizes[i], activation=activation))
	model.add(Dense(1, activation='linear'))	
	model.compile(optimizer=Adam(learning_rate=learnRate),
			#loss='mean_squared_error',metrics=['mae',coeff_determination])
			loss='mean_squared_error',metrics=['mae'])
	return model

def buildSpatialCNN(nFeatureMaps,filterSize,fcLayerSize,activation,learnRate,inputShape):

	model = Sequential()
	model.add(Conv2D(nFeatureMaps, (filterSize,filterSize), activation='relu', input_shape=(inputShape[2],inputShape[1],inputShape[0])))#(tile-size,tile-size, number features)
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(fcLayerSize, activation=activation))
	model.add(Dense(1))#linear activation (default)
	model.compile(optimizer=Adam(learning_rate=learnRate),			
			loss='mean_squared_error',metrics=['mae'])
	return model


def buildSpectralCNN(nFeatureMaps,filterSize,fcLayerSize,activation,learnRate,inputShape):

	model = Sequential()
	model.add(Conv2D(nFeatureMaps, kernel_size=(1, 1),strides=(1,1), activation='relu', input_shape=(inputShape[2],inputShape[1],inputShape[0])))#(tile-size,tile-size, number features)
	model.add(AveragePooling2D(pool_size=(filterSize,filterSize)))
	model.add(Flatten())
	model.add(Dense(fcLayerSize, activation=activation))
	model.add(Dense(1))#linear activation (default)
	model.compile(optimizer=Adam(learning_rate=learnRate),			
			loss='mean_squared_error',metrics=['mae'])
	return model

def isDeepLearningModel(algName):
	return deepLearningModelMap[algName]

def isModelWithInputTensors(algName):
	return modelsWithTensorsMap[algName]

	
#returns the map of hyperparameter choice for a given algorimth for a hyperparameterSetKey for federated learning experiments
def getFLStaticModelHyperparameters(algName,hyperparameterSetKey):
	
	#error checking
	if not algName in FLStaticHyperParameterMap:
		raise Exception("Model "+algName+" hyperparameters not supported for federated learning experiments.")
	
	hyperParameterSet =FLStaticHyperParameterMap[algName]
	if not hyperparameterSetKey in hyperParameterSet:
		raise Exception("No static hyperparameter set for model "+algName+" for hyperparameterSetKey "+str(hyperparameterSetKey)+" for federated learning experiments.")
		
	return hyperParameterSet[hyperparameterSetKey]