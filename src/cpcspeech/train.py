import torch
import matplotlib.pyplot as plt
import numpy as np


class Training():


    def __init__(self, model, optimizer, train_dataloader, valid_dataloader, device):

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self.gpu = False
        if device == 'gpu':
            self.gpu=True


    def train_epoch(self):

        self.model.train()
        total_train_loss = 0
        total_train_acc = []

        for idx_batch, batch in enumerate(self.train_dataloader):
            
            batch_size = batch.size()[0]

            self.optimizer.zero_grad()

            hidden = self.model.initialize_hidden(batch_size,gpu=self.gpu)
            loss, acc, hidden = self.model.forward(batch.float().to(self.device),hidden, self.device)

            loss.backward()
            self.optimizer.step()
            
            total_train_loss += len(batch) * loss
            total_train_acc.append(acc)

        total_train_loss /= len(self.train_dataloader.dataset)

        return total_train_loss, total_train_acc



    def validation_epoch(self):

        self.model.eval()
        total_val_loss = 0
        total_val_acc = []

        with torch.no_grad():
            for idx_batch, batch in enumerate(self.valid_dataloader):
        #             print(batch.size())
                batch_size = batch.size()[0]
                
                hidden = self.model.initialize_hidden(batch_size, gpu=self.gpu)
                val_loss, val_acc, hidden = self.model.forward(batch.float().to(self.device), hidden, self.device)
                total_val_loss += len(batch) * val_loss
                total_val_acc.append(val_acc)

        total_val_loss /= len(self.valid_dataloader.dataset)
        
        return total_val_loss, total_val_acc


    def train(self, n_epochs, early_stop_epochs=20, plot=False):


        best_valid_loss = 1000
        early_stop_counter = 0

        train_loss_s = []
        valid_loss_s = []


        for epoch in range(n_epochs):

            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc = self.validation_epoch()
            
            train_loss_s.append(train_loss)
            valid_loss_s.append(valid_loss)
            

            # if best_valid_loss == None:
            #     best_valid_loss = valid_loss

            if valid_loss<best_valid_loss:
                
                best_valid_loss = valid_loss
                early_stop_counter = 0
            
            else:
                
                early_stop_counter += 1
            
            
            print('-- epoch {0} --'.format(epoch))
            print('train loss: {0}'.format(train_loss))
            print('valid loss: {0}'.format(valid_loss))
            print('best valid loss: {0}'.format(best_valid_loss))
            
            plt.subplot(2,1,1)
            plt.plot(train_loss_s,'r-o')
            plt.plot(valid_loss_s,'b-o')
            
            plt.subplot(2,1,2)
            plt.plot(np.mean(train_acc,0),'r-o')
            plt.ylabel('mean accuracy')
            plt.xlabel('time steps')

            plt.plot(np.mean(valid_acc,0),'b-o')
            plt.ylabel('mean accuracy')
            plt.xlabel('time steps')
            
            plt.show()
            
            
            if early_stop_counter >= early_stop_epochs:
                
                break



class Spk_Training():


    def __init__(self, cpc_model, spk_model, loss, optimizer, train_dataloader, valid_dataloader, device):

        self.cpc_model = cpc_model
        self.spk_model = spk_model
        self.optimizer = optimizer
        self.loss = loss
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self.gpu = False
        if device == 'gpu':
            self.gpu=True



    def train_epoch(self):

        self.cpc_model.eval()
        self.spk_model.train()
        
        total_acc = 0
        total_loss = 0

        for b_idx, batch in enumerate(self.train_dataloader):
         
            spk_data = batch[0]
            batch_size = spk_data.size()[0]
            spk_data = spk_data.float().to(self.device)

            hidden = self.cpc_model.initialize_hidden(batch_size,gpu=self.gpu)
            cpc_embedding, hidden = self.cpc_model.predict(spk_data,hidden)
            cpc_embedding = cpc_embedding.contiguous().view((-1,256)) # emb_size = hidden.size()[1]?

            spk_target = batch[1]
            spk_target = spk_target.to(self.device)

            # shuffle encoded data again


            # train mlp model
            self.optimizer.zero_grad()
            out = self.spk_model.forward(cpc_embedding)
            # spk_loss = F.nll_loss(out,spk_target)
            spk_loss = self.loss(out,spk_target)
            spk_loss.backward()
            self.optimizer.step()


            pred = out.max(1, keepdim=True)[1]
            correct = pred.eq(spk_target.view_as(pred))
            acc = 1.*correct.sum().item()/len(spk_data)
            
            total_acc += acc
            total_loss += spk_loss.item()
            
    #         print(spk_loss.item())
            
        total_acc /= len(self.train_dataloader.dataset)
        total_loss /= len(self.train_dataloader.dataset)

        
        # return spk_loss, acc
        return total_loss, total_acc


    def validation_epoch(self):

        self.cpc_model.eval()
        self.spk_model.eval()
        
        total_acc = 0
        total_loss = 0
        
        with torch.no_grad():
            for b_idx, batch in enumerate(self.valid_dataloader):

                spk_data = batch[0]
                batch_size = spk_data.size()[0]
                spk_data = spk_data.float().to(self.device)

                hidden = self.cpc_model.initialize_hidden(batch_size, gpu=self.gpu)
                cpc_embedding, hidden = self.cpc_model.predict(spk_data,hidden)
                cpc_embedding = cpc_embedding.contiguous().view((-1,256)) # emb_size = hidden.size()[1]?

                spk_target = batch[1]
                spk_target = spk_target.to(self.device)

                out = self.spk_model.forward(cpc_embedding)
                
                pred = out.max(1, keepdim=True)[1]
                correct = pred.eq(spk_target.view_as(pred))
                acc = correct.sum().item()
                
                # spk_loss = F.nll_loss(out,spk_target).item()
                spk_loss = self.loss(out,spk_target).item()
                
                total_loss+=spk_loss
                total_acc += acc


        total_loss /= len(self.valid_dataloader.dataset)
        total_acc  /= 1.*len(self.valid_dataloader.dataset) 

        
        # return spk_loss, acc
        return total_loss, total_acc


    def train(self, n_epochs, early_stop_epochs=20, plot=False):

        best_valid_loss = 1000
        early_stop_counter = 0

        train_loss_s = []
        valid_loss_s = []

        train_acc_s = []
        valid_acc_s = []


        for epoch in range(n_epochs):

            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc = self.validation_epoch()
            
            train_loss_s.append(train_loss)
            valid_loss_s.append(valid_loss)

            train_acc_s.append(train_acc)
            valid_acc_s.append(valid_acc)
            

            # if best_valid_loss == None:
            #     best_valid_loss = valid_loss

            if valid_loss<best_valid_loss:
                
                best_valid_loss = valid_loss
                early_stop_counter = 0
            
            else:
                
                early_stop_counter += 1
            

            print('-- epoch {0} --'.format(epoch))
            print('train loss: {0}'.format(train_loss))
            print('valid loss: {0}'.format(valid_loss))
            print('best valid loss: {0}'.format(best_valid_loss))

            print('train accuracy: {0}'.format(train_acc))
            print('valid accuracy: {0}'.format(valid_acc))
            
            plt.subplot(2,1,1)
            plt.plot(train_loss_s,'r-o')
            plt.plot(valid_loss_s,'b-o')
            
            plt.subplot(2,1,2)
            plt.plot(train_acc_s,'r-o')
            plt.plot(valid_acc_s,'b-o')
            plt.ylabel('mean accuracy')
            plt.xlabel('time steps')
            
            plt.show()
            
            
            if early_stop_counter >= early_stop_epochs:
                
                break

