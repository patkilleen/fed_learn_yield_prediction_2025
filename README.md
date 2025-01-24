## Crop Yield Prediction Using Federated Learning

This is the official documentation of the code repository of the paper: Patrick Killeen, Iluju Kiringa, and Tet Yeap "UAV Imagery-Based Yield Prediction Using Federated Learning and Model Sharing for Smart Farming Applications".

## Requirements

This is my experiment environment for the yield prediction experiments.
-	64-bit Linux (Ubuntu 20.04) desktop with a NVIDIA GeForce RTX 3060 graphics card
-	Python 3.8.19
-	TensorFlow version 2.12.0 
-	CUDA version 11.8
-	cuDNN version 8.6
-	sklearn version 1.3.2
-	numpy version 1.24.3
-	pandas version 2.0.3
-   geopandas version 0.13.2

This is my environment for creating yield precision maps using the R programming language.
- Windows 10
- R (64 bit) version 4.1.3 
- shape version 1.4.6
- ggplot2 version 3.3.5
- raster version 3.5-15
- sf version 1.0-8
- sp version 1.4-6

## Example of Running an MLP Yield Prediction Experiment

The below Linux shell commands will run a full example of the MLP experiments including centralized learning, local learning, federated learning, average ensemble learning, and stacked ensemble learning.
```
mkdir output
mkdir output/fed-learn
python FLExperimenter.py --inFile input/configs/main/iid/MLP-multi-temporal.csv --outDirectory output/fed-learn
```

## Documentation

The API and documentation of the project can be found in documentation/yield-predict-documentation.pdf.

### File Descriptions
We describe the files of interest of the project in this section.

- documentation/yield-predict-documentation.pdf: the documentation that explains the API of the scripts

The scripts required to run the yield prediction:
- common.py: common functionality shared by all models
- dataset.py: all the logic for reading and pre-processing the datasets
- experimenter.py: main, the core of running all the single-dataset cross-validation experiments used for hyperparameter tuning
- FLExperimenter.py:  main, the core of running all the federated learning experiments
- model.py: all the learning algorithm logic is in here
- myio.py: all the file input and output logic is in here
- spatial.py: dataset spatial partitioning logic is in here

Other scripts are below:
- plsr.py: the feature selection logic that uses partial least squares regression
- generatePrecisionMap.r: the R CRAN script used to create yield precision maps
Directories:
- input/ #holds all input files
	- configs/ #holds the configuration files 
		- main/  # configuration files used for the federated learning experiments		
			- iid/  # configuration files for the independent and identically distributed (IID) client dataset experiments
			- non-iid/  # configuration files for the non-IID (both non-IID1 and non-IID2) client dataset experiments
		- hyper-param-sel/ #configuration files of the single-dataset cross-validation experiments used to select the hyperparameters for the federated learning experiments
	- datasets/ #holds the pre-processed yield+imagery dataset files	
		-mono-temporal-DoY201.csv #yield and imagery data containing only imagery features from day-of-year 201
		-mono-temporal-DoY216.csv #yield and imagery data containing only imagery features from day-of-year 216
		-multi-temporal.csv #yield and imagery data containing imagery features from both day-of-year 201 and 216, and includes a temporal delta feature of reflectance difference between day-of-year 201 and 216 reflectance
	- selected-features/ #stores the files that indicate what feature is selected for each type of experiment  using same files structure as input/datasets
		-mono-temporal-DoY201.csv #selected features for the day-of-year 201 imagery dataset (input/datasets/mono-temporal-DoY201.csv)
		-mono-temporal-DoY216.csv #selected features for the day-of-year 216 imagery dataset (input/datasets/mono-temporal-DoY216.csv)
		-multi-temporal.csv #selected features for the dataset (input/datasets/multi-temporal.csv) that than contains imagery features from both day-of-year 201 and 216 
		-raw-bands-mono-temporal-DoY201.csv #only the red, green, blue, near-infrared (NIR), and red-edge mean reflectance day-of-year 201 features are selected from the (input/datasets/mono-temporal-DoY201.csv) dataset (simulated 2.5 m spatial resolution raw imagery)
		-raw-bands-mono-temporal-DoY216.csv #only the red, green, blue, near-infrared (NIR), and red-edge mean reflectance day-of-year 216 features are selected from the (input/datasets/mono-temporal-DoY216.csv) dataset (simulated 2.5 m spatial resolution raw imagery)
		-raw-bands-multi-temporal.csv #only the red, green, blue, near-infrared (NIR), and red-edge mean reflectance day-of-year 201 and 216 features are selected from the (input/datasets/multi-temporal.csv) dataset (simulated 2.5 m spatial resolution raw imagery)
- output: the directory that will be created when the project is run using the example scripts runExperiments.sh and runFeatureSelection.sh
- raw-data/ #contains raw data files
	- field-boundaries/# contains all the field boundaries in  shapefile format (useful for map generation)
	- area-x.o-satellite-image-raster.tif #low spatial resolution satellite image of area x.O. Useful for creating a background for precisin map generation (see generatePrecisionMap.r)
