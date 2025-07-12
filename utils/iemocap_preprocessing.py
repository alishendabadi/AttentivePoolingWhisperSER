
import numpy as np
import pandas as pd
import librosa, os, torch
from torch.utils.data import Dataset
from collections import namedtuple
import re

iemofields = ('audios', 'labels', 'sessions', 'file_names', 'file_paths', 'genders')
IemoInfo = namedtuple('IemoInfo', iemofields, defaults=(None,) * len(iemofields))

def read_iemocap(iemocap_excel_path, iemo_audios_folder_path, model_sr=16000, save_file_names=True, save_file_path=True):


    df = pd.read_excel(iemocap_excel_path, index_col=0)
    df_cleaned = df[df['emotion'].isin(['ang', 'neu', 'hap', 'sad', 'exc'])]

    # whisper_tiny_processor = WhisperProcessor.from_pretrained('whisper_small')
    emo2num = {
        'hap' : 0,
        'neu' : 1,
        'sad' : 2,
        'ang' : 3,
        'exc' : 0
    }

    num2emo = {
        0 : 'Happy',
        1 : 'Neutral',
        2 : 'Sad',
        3 : 'Angery'
    }

    audios = []
    emotions = []
    labels = []
    file_names = []
    file_paths = []
    genders = []
    sessions = []


    for root, folders, files in os.walk(iemo_audios_folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if not filename.startswith('.'):
                title = filename.replace('.wav', '')
                if title in df_cleaned['name'].tolist():
                    file_path = os.path.join(root, filename)
                    audio_data, _ = librosa.load(file_path, sr=model_sr)

                    audios.append(audio_data)
                    emotion = df_cleaned[df_cleaned['name'] == title]['emotion'].item()
                    label = emo2num[emotion]

                    emotions.append(emotion)
                    labels.append(label)
                    if save_file_names == True: file_names.append(filename)
                    else: file_names.append(None)
                    if save_file_path == True: file_paths.append(file_path)
                    else: file_names.append(None)
                    sessions.append(int(title[4]))
                    genders.append(title[-4])

    return IemoInfo(
       audios= audios,
       labels = labels,
       file_names = file_names,
       file_paths = file_paths,
       sessions = sessions,
       genders = genders
    )


shemofields = ('audios', 'labels', 'file_names', 'file_paths', 'genders')
ShemoInfo = namedtuple('ShemoInfo', shemofields, defaults=(None,) * len(shemofields))

def read_shemo(shemo_audios_folder_path, model_sr=16000, save_file_names=True, save_file_path=True):

    num2emo = {
        0 : 'Happy',
        1 : 'Neutral',
        2 : 'Sad',
        3 : 'Angery',
        4 : 'Surprized'
    }

    emo2num = {
       'H': 0,
       'N': 1,
       'S': 2,
       'A': 3,
       'W': 4
    }

    audios = []
    emotions = []
    labels = []
    file_names = []
    file_paths = []
    genders = []

    pattern = r'^...(.)'

    for root, folders, files in os.walk(shemo_audios_folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            audio_data, _ = librosa.load(file_path, sr=model_sr)

            match = re.search(pattern, filename)
            if match:
               char = match.group(1)
               if char in ['H', 'N', 'S', 'W', 'A']:
                  audios.append(audio_data)
                  emotion = char
                  label = emo2num[emotion]
                  emotions.append(emotion)
                  labels.append(label)
                  if save_file_names: file_names.append(filename)
                  else: file_names.append(None)
                  if save_file_path: file_paths.append(file_path)
                  else: file_paths.append(None)
                  genders.append(filename[0])

    return IemoInfo(
       audios= audios,
       labels = labels,
       file_names = file_names,
       file_paths = file_paths,
       genders = genders
    )



class MyDataset(Dataset):
  def __init__(self, audios, labels, processor, model_sr=16000):
    self.audios = audios
    self.labels = labels
    self.processor = processor
    self.model_sr = model_sr

  def __len__(self):
    return len(self.audios)

  def __getitem__(self, idx):
    audio_data = self.audios[idx]
    label = torch.tensor(self.labels[idx])
    processed = self.processor(audio_data, sampling_rate=self.model_sr, return_tensors="pt")

    return {
        'input_features': torch.squeeze(processed['input_features'], dim=0), #squeezing is done to unbatch data and remove the vatch dimention (which is 1 here)
        'labels': label,
    }