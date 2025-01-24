#create directory for output files
mkdir output

mkdir output/hyper-param-select
python experimenter.py --inFile input/configs/hyper-param-sel/config_MLP-mono-temporal-DoY201.csv --outDirectory output/hyper-param-select
