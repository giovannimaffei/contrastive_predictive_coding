import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
import h5py
import os


class LibriSpeech100_dataset(Dataset):
    
    def __init__(self, data_path, seq_len, batch_size):
        
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data = h5py.File(data_path, 'r')
        self.file_list = list(self.data.keys())
        self.file_list = self.file_list[:(len(self.file_list)//batch_size)*batch_size]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        filename = self.file_list[idx]
        audio_file = self.data[filename][()]
        rnd_idx = np.random.randint(0, len(audio_file)-self.seq_len)
        audio_sample = audio_file[rnd_idx:rnd_idx+self.seq_len]
        audio_sample = audio_sample.reshape(1,-1)
        
        return audio_sample
    
    
def LibriSpeech100_preprocessing(file_list_path, data_path, output_path, seq_len):

    file_list = open(file_list_path).read().split('\n')
    dset_formatted = h5py.File(output_path, 'w')
    
    for filename in file_list[:100]:
#         print(filename)
        reader_id, chapter_id, _ = filename.split('-') 
        file_path = os.path.join(data_path,reader_id,chapter_id,filename)+'.flac'
#         print(file_path)
        audio_file, sr = sf.read(file_path)
        if audio_file.shape[0]>seq_len:
            dset_formatted.create_dataset(filename, data=audio_file)

    dset_formatted.close()



class LibriSpeech100_dataset_spk_class(Dataset):
    
    def __init__(self, data_path, seq_len, batch_size):
        
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data = h5py.File(data_path, 'r')
        self.file_list = list(self.data.keys())
        self.file_list = self.file_list[:(len(self.file_list)//batch_size)*batch_size]
        
        self.unique_spk_ids = np.unique([f.split('-')[0] for f in list(self.data.keys())])
        
        self.spk_labels = {}
        for i,s_id in enumerate(self.unique_spk_ids):
            self.spk_labels[s_id] = i
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        filename = self.file_list[idx]
        audio_file = self.data[filename][()]
        rnd_idx = np.random.randint(0, len(audio_file)-self.seq_len)
        audio_sample = audio_file[rnd_idx:rnd_idx+self.seq_len]
        audio_sample = audio_sample.reshape(1,-1)
        
        spk_id = filename.split('-')[0]
        speaker_label = self.spk_labels[spk_id]
        
        return audio_sample, speaker_label


class Simple_dataset(Dataset):
    
    def __init__(self, X):
        
        self.data = X
        
    def __len__(self):
        
        return len(self.data)
        
    def __getitem__(self,idx):
        
        return self.data[idx]

