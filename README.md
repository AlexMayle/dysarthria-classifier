# Diagnosing Dysarthria with LSTMS

## Installation
After cloning the repo, cd into and run
```
pipenv shell
pipenv install
```

## Overview
The main function is `fold_validation.py`. It takes many arguments, but the defaults are the ones used in the paper. `mode`, `--input_path`, and `--output_path` should be provided though. 
`python fold_validation.py MODE --input_path input/path --output_path output/path`
Where MODE corresponds to a model, following the table below. 
```
2 := LSTM-1
3 := LSTM-2
4 := Bi-LSTM-1
```

To reproduce the results, run each model on the corresponding dataset for an expirment. Then, use the appropriate analysis python script to produce the graphs. See the section for a specific expirment to train, predict, and analyze the results of each expirment.

## Creating MFCC datasets
The MFCC data is available in the repo as it was used in the expirments. The appropriate MFCC pickle object is included in the examples commands for each expirment below. If you'd like to process the raw WAV files yourself, you may download them here. See `createDatasets.py` for converting the WAV files to MFCC data.

## Experiments
The process consists of two steps. Executing scripts to cross validate, which encompasses in training and inference. The output is predictions. The next step is to analyze those predictions by running the analysis scripts. These calculate metrics and plots graphs. 

### A note on output paths
You need to cross validate each model before attempting to run analysis scripts, with respect to each expirerement. This is because the `output_path` is generally arbitrary, but the only requirement is that three folders exist, called `lstm-1`, `lstm-2`, and `bi-lstm-1`. The analysis scripts expect this. For example, you should have the following output folder structure before running analysis scripts. This is fufilled by the default examples given below. 
```
output_folder
|- lstm-1
|- lstm-2
|_ bi-lstm-1
```
## Expirment 2.0 (Novel Speakers)
```
# Train and cross-validate each model
python fold_validation.py 2 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/lstm-1
python fold_validation.py 3 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/lstm-2
python fold_validation.py 4 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/bi-lstm-1
# Analyze predictions
python analyze_predictions --data_path mfcc_data/mfcc_full_set-2 --pred_path output_pred_folder/ --save_path path/to/output/graphs
```

## Expirment 2.1 (Cepstrum Counts)
Here we are going to repeat the commands with different datasets for each number of cepstrums. We'll also repeat the analyze commands for each number of cepstrums.
```
# Train and cross-validate the normal, 13 cepstrums
python fold_validation.py 2 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/lstm-1
python fold_validation.py 3 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/lstm-2
python fold_validation.py 4 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/bi-lstm-1
# Repeat for 19 cepstrums
python fold_validation.py 2 --input_path mfcc_data/19_cepstrums.pkl --output_path output_pred_folder-19/lstm-1
python fold_validation.py 3 --input_path mfcc_data/19_cepstrums.pkl --output_path output_pred_folder-19/lstm-2
python fold_validation.py 4 --input_path mfcc_data/19_cepstrums.pkl --output_path output_pred_folder-19/bi-lstm-1
# Repeat for 25 cepstrums
python fold_validation.py 2 --input_path mfcc_data/25_cepstrums.pkl --output_path output_pred_folder-25/lstm-1
python fold_validation.py 3 --input_path mfcc_data/25_cepstrums.pkl --output_path output_pred_folder-25/lstm-2
python fold_validation.py 4 --input_path mfcc_data/25_cepstrums.pkl --output_path output_pred_folder-25/bi-lstm-1
# Analyze predictions for each cepstrum amount
python analyze_syl_type_preds --data_path mfcc_data/mfcc_full_set-2 --pred_path output_pred_folder/ --save_path path/to/output/graphs
python analyze_syl_type_preds --data_path mfcc_data/19_cepstrums --pred_path output_pred_folder-19/ --save_path path/to/output/graphs
python analyze_syl_type_preds --data_path mfcc_data/25_cepstrums --pred_path output_pred_folder-25/ --save_path path/to/output/graphs
```

# Expirment 2.2 (Syllable Types)
We'll train each model with each of the three datasets containing different distributions of syllables.
```
# Train and cross-validate the normal dataset with all syllables
python fold_validation.py 2 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/lstm-1/
python fold_validation.py 3 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/lstm-2
python fold_validation.py 4 --input_path mfcc_data/mfcc_full_set-2.pkl --output_path output_pred_folder/bi-lstm-1
# Repeat for data with no "cv" 
python fold_validation.py 2 --input_path mfcc_data/without_z.pkl --output_path output_pred_folder-without-z/lstm-1/
python fold_validation.py 3 --input_path mfcc_data/without_z.pkl --output_path output_pred_folder-without-z/lstm-2/
python fold_validation.py 4 --input_path mfcc_data/without_z.pkl --output_path output_pred_folder-without-z/bi-lstm-1/
# Repeat for data with no "c/a/"
python fold_validation.py 2 --input_path mfcc_data/without_zz.pkl --output_path output_pred_folder-without-zz/lstm-1/
python fold_validation.py 3 --input_path mfcc_data/without_zz.pkl --output_path output_pred_folder-without-zz/lstm-2/
python fold_validation.py 4 --input_path mfcc_data/without_zz.pkl --output_path output_pred_folder-without-zz/bi-lstm-1/
# Analyze predictions for each cepstrum amount
python analyze_syl_type_preds.py --z_path output_pred_folder-without-z/ --zz_path output_pred_folder-without-zz/ --norm_path output_pred_folder --save_path path/to/output/graphs
```
