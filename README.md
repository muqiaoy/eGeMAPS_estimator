# eGeMAPS_estimator

## Prerequisites
```
pip install -r requirements.txt
```

## Datasets
1. Please follow https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master to download the DNS Interspeech 2020 dataset.

2. Edit paths in `noisyspeech_synthesizer.cfg` and run `noisyspeech_synthesizer_multiprocessing.py` to generate your train (and validation) data.

    Most likely, you will not want to change the other parameters in .cfg for the train data, and then you will get 12,000 synthesized audios. You may change the `fileindex_end` in the .cfg to have a small set of validation data. 

    You can also manually change `num_train_files` in `conf/` to adjust the number of train audios in use.

3. Edit paths in `conf/` to make it consistent to your folders that contains the data.



## Usage
1. Train the eGeMAPS estimator.

    Generating the eGeMAPS feature for the first time could be slow.
```
python train_est.py --train_config conf/estimator.yaml 
```

2. Finetinue the enhancement model (only support Demucs so far).
```
python train.py --train_config conf/Demucs.yaml
```
