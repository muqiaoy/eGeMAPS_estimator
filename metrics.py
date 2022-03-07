import pysepm
import os,sys
from glob import glob
import re
import librosa
import pickle


clean_files = glob("/home/muqiaoy/workhorse3/Datasets/DNS-Challenge/datasets/test_set/synthetic/no_reverb/clean/*.wav")
enhanced_files = glob("./enhanced_audios/estimator_e2e_L2_lr1e-4_lstm/*.wav")
# noisy_files = glob("/home/muqiaoy/workhorse3/Datasets/DNS-Challenge/datasets/test_set/synthetic/no_reverb/noisy/*.wav")
noisy_files = glob("./enhanced_audios/baseline/*.wav")




fid_pattern = ".fileid_(.*)\.."

# get file id given a path to .wav file
def get_fid(path):
    matched = re.findall(fid_pattern, path)
    if "_" in matched[0]:
        matched[0] = matched[0].split("_")[0]
    assert len(matched) == 1, print(path)
    return matched[0]

clean_files = sorted(clean_files, key=get_fid)
enhanced_files = sorted(enhanced_files, key=get_fid)
noisy_files = sorted(noisy_files, key=get_fid)


# For llr, wss, cd, bsd, the lower the better 
metrics = ["fwSNRseg", "SNRseg", "llr", "wss", "cepstrum_distance", "stoi", "csii", "pesq", "composite", "ncm", "srmr"]
enhanced_results = {}
noisy_results = {}
for metric in metrics:
    enhanced_results[metric] = []
    noisy_results[metric] = []



cnt = 0
for clean, enhanced, noisy in zip(clean_files, enhanced_files, noisy_files):
    print(cnt)
    assert get_fid(clean) == get_fid(enhanced) == get_fid(noisy)

    clean_speech, fs = librosa.load(clean, sr=16000)
    enhanced_speech, fs2 = librosa.load(enhanced, sr=16000)
    noisy_speech, fs3 = librosa.load(noisy, sr=16000)
    assert fs == fs2 == fs3
    for metric in metrics:
        # print(metric)
        func = getattr(pysepm, metric)
        if metric != "srmr":
            enhanced_result = func(clean_speech, enhanced_speech, fs)
            # print(enhanced_result)
            enhanced_results[metric].append(enhanced_result)
            noisy_result = func(clean_speech, noisy_speech, fs)
            # print(noisy_result)
            noisy_results[metric].append(noisy_result)
        else:
            enhanced_result = func(enhanced_speech, fs)
            # print(enhanced_result)
            enhanced_results[metric].append(enhanced_result)
            noisy_result = func(noisy_speech, fs)
            # print(noisy_result)
            noisy_results[metric].append(noisy_result)
    cnt += 1

for key in enhanced_results.keys():
    print(key)
    if key not in ["csii", "pesq", "composite"]: 
        print(sum(enhanced_results[key]) / len(enhanced_results[key]))
        print(sum(noisy_results[key]) / len(noisy_results[key]))
    # elif key == 'pesq':
    #     print(sum(enhanced_results[key]) / len(enhanced_results[key]))
    #     print(sum(noisy_results[key]) / len(noisy_results[key]))

f = open("enhanced_results.pkl", "wb")
pickle.dump(enhanced_results, f)
f = open("noisy_results.pkl", "wb")
pickle.dump(noisy_results, f)




