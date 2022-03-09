import pysepm
import os,sys
from glob import glob
import re
import librosa
import pickle
from tqdm import tqdm
import multiprocessing
import numpy as np
import pandas as pd

np.seterr(divide = 'ignore') 



clean_dir = "/home/muqiaoy/workhorse3/Datasets/DNS-Challenge/datasets/test_set/synthetic/no_reverb/clean/"
noisy_dir = "/home/muqiaoy/workhorse3/Datasets/DNS-Challenge/datasets/test_set/synthetic/no_reverb/noisy/"
enhanced_dir = "./enhanced_audios/estimator_vae10/"
demucs_dir = "./enhanced_audios/baseline/"


clean_ids   = []
noisy_ids   = []
enhanced_ids = []
demucs_ids = []

for wav_id in sorted(os.listdir(noisy_dir)):
    
    noisy_ids.append(wav_id)
    enhanced_ids.append(wav_id)
    demucs_ids.append(wav_id)
    
    clean_id = "_".join(["clean"] + wav_id.split("_")[-2:])
    clean_ids.append(clean_id)



def get_evaluation(clean_speech, noisy_speech, wav_id, sr = 16000):
    
    Y0 = pysepm.fwSNRseg(clean_speech, noisy_speech, sr)
    Y1 = pysepm.SNRseg(clean_speech, noisy_speech, sr)
    Y2 = pysepm.llr(clean_speech, noisy_speech, sr)
    Y3 = pysepm.wss(clean_speech, noisy_speech, sr)
    Y4 = pysepm.cepstrum_distance(clean_speech, noisy_speech, sr)
    Y5 = pysepm.stoi(clean_speech, noisy_speech, sr)
    Y6, Y7, Y8 = pysepm.csii(clean_speech, noisy_speech, sr)
    _, Y9 = pysepm.pesq(clean_speech, noisy_speech, sr)
    Y10, Y11, Y12 = pysepm.composite(clean_speech, noisy_speech, sr)
    Y13 = pysepm.ncm(clean_speech, noisy_speech, sr)
    Y14 = pysepm.srr_seg(clean_speech, noisy_speech, sr)
    
    return [wav_id, 
            Y0,  Y1,  Y2,  Y3,  Y4,
            Y5,  Y6,  Y7,  Y8,  Y9,
            Y10, Y11, Y12, Y13, Y14]



def execute_multiprocess(clean_dir, clean_ids, noisy_dir, noisy_ids):
    
    PROCESSES = 4
    
    assert(len(clean_ids) == len(noisy_ids))
    
    assert(set(["_".join(["clean"] + n_id.split("_")[-2:]) for n_id in noisy_ids]) == set(clean_ids))
    
    # Could be optimized to read in get_evaluation
    print("Loading Clean Samples")
    all_clean_speech = [librosa.load(clean_dir + c_id, sr=16000)[0] for c_id in tqdm(clean_ids)]
    
    print("Loading Noisy Samples")
    all_noisy_speech = [librosa.load(noisy_dir + n_id, sr=16000)[0] for n_id in tqdm(noisy_ids)]
    
    with multiprocessing.Pool(PROCESSES) as pool:
        
        in_args = zip(all_clean_speech, all_noisy_speech, noisy_ids)
        jobs    = [pool.apply_async(get_evaluation, in_arg) for in_arg in in_args]
        
        result = [None] * len(jobs)
        for i, job in enumerate(tqdm(jobs)):
            result[i] = job.get()
    
    return np.array(result)


def save_csv(filename, clean_dir, clean_ids, noisy_dir, noisy_ids):
    result = execute_multiprocess(clean_dir, clean_ids, noisy_dir, noisy_ids)
    result_rows = result[:, 0]
    result_cols = [
        "fwSNRseg",
        "SNRseg",
        "llr",
        "wss",
        "cepstrum_distance",
        "stoi",
        "csii_1",
        "csii_2",
        "csii_3",
        "pesq",
        "composite_1",
        'composite_2',
        "composite_3",
        'ncm',
        "srr_seg"]

    df = pd.DataFrame(result[:, 1:], index=result_rows, columns=result_cols)
    df.sort_index(inplace=True)
    df.to_csv(filename)
    print(df.astype(float).mean())


# save_csv("results/noisy.csv", clean_dir, clean_ids, noisy_dir, noisy_ids)
save_csv("results/enhanced_vae10.csv", clean_dir, clean_ids, enhanced_dir, enhanced_ids)
# save_csv("results/baseline.csv", clean_dir, clean_ids, demucs_dir, demucs_ids)

