from cpcspeech.data import Simple_dataset, LibriSpeech100_dataset
from cpcspeech.train import Training
from torch.utils.data import DataLoader
from cpcspeech.models import CPC
import torch
import torch.optim as optim
import numpy as np
import datetime

class CPC_speech:

	def __init__(self, gpu=False):

		self.gpu=gpu
		device_type = 'cpu'
		if self.gpu: device_type = 'cuda'
		self.device = torch.device(device_type)
		# self.batch_size = batch_size
		
		self.n_timesteps = 12
		self.seq_len = 20480
		self.emb_size = 256
		self.cpc_model = CPC(self.n_timesteps, self.seq_len).to(self.device)
	

	def load_pretrained_model(self,model_path):

		self.cpc_model.load_state_dict(torch.load(model_path, map_location=self.device))

	
	def freeze(self):

		for param in self.cpc_model.parameters():
			param.requires_grad = False

	
	def load_train_data(self, train_data_path, valid_data_path, test_data_path=None, batch_size=8):

		libri_train = LibriSpeech100_dataset(train_data_path,self.seq_len,batch_size)
		self.train_dataloader = DataLoader(libri_train, batch_size=batch_size, shuffle=True, num_workers=0)

		libri_valid = LibriSpeech100_dataset(valid_data_path,self.seq_len,batch_size)
		self.valid_dataloader = DataLoader(libri_valid, batch_size=batch_size, shuffle=True, num_workers=0)

		if test_data_path != None:
			libri_test = LibriSpeech100_dataset(test_data_path,self.seq_len,batch_size)
			self.test_dataloader = DataLoader(libri_test, batch_size=batch_size, shuffle=True, num_workers=0)

	
	def train(self, n_epochs, lr=0.0002, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True, plot=False):

		train_params = [p for p in self.cpc_model.parameters() if p.requires_grad == True]
		self.optimizer = optim.Adam(train_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

		self.training = Training(self.cpc_model, self.optimizer, self.train_dataloader, self.valid_dataloader, self.device)
		self.training.train(n_epochs, plot=plot)


	def save_model(self, save_model_folder):

		time_now = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
		torch.save(self.cpc_model.state_dict(), save_model_folder+'cpc_speech_model_{0}'.format(time_now))
		torch.save(self.optimizer.state_dict(), save_model_folder+'cpc_speech_opti_{0}'.format(time_now))


	def transform(self, X):

		dataset = Simple_dataset(X)
		dataloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
		audio_embedding_s = []

		for batch_idx, batch in enumerate(dataloader):

			batch = batch.float().to(self.device)
			batch_size = batch.size()[0]

			hidden = self.cpc_model.initialize_hidden(batch_size, gpu=self.gpu)
			cpc_embedding, hidden = self.cpc_model.predict(batch,hidden)
			cpc_embedding = cpc_embedding.contiguous().view((-1,self.emb_size))
			audio_embedding_s.append(cpc_embedding.cpu().detach().numpy())

			audio_embedding_s = np.vstack(audio_embedding_s)
			return audio_embedding_s
