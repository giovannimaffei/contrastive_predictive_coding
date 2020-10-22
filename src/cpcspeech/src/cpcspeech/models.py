import torch


class CPC(torch.nn.Module):

    def __init__(self, n_timesteps, seq_len):
        
        super(CPC, self).__init__()
        
        self.n_timesteps = n_timesteps
        self.seq_len = seq_len
        self.comp_rate = 160

        self.g_enc = torch.nn.Sequential(
            
                    torch.nn.Conv1d(1, 512, stride=5, kernel_size=10, padding=3, bias=False),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),

                    torch.nn.Conv1d(512, 512, stride=4, kernel_size=8, padding=2, bias=False),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),

                    torch.nn.Conv1d(512, 512, stride=2, kernel_size=4, padding=1, bias=False),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),

                    torch.nn.Conv1d(512, 512, stride=2, kernel_size=4, padding=1, bias=False),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),

                    torch.nn.Conv1d(512, 512, stride=2, kernel_size=4, padding=1, bias=False),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True)

                     )

        # autoregressive unit
        self.g_ar = torch.nn.GRU(512, 256, batch_first=True)

        # list of W for each k timestep
        # note the input=256 from the GRU, output=512 to match the size of z
        self.W = torch.nn.ModuleList([torch.nn.Linear(256,512) for t in range(self.n_timesteps)])

        # softmax function to compute the prediction
        self.softmax = torch.nn.Softmax()

        # log softmax function to compute final loss
        self.log_softmax = torch.nn.LogSoftmax()


        def weights_initialization(module):

            if type(module) == torch.nn.Conv1d:
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

            elif type(module) == torch.nn.BatchNorm1d:
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)

            elif type(module) == torch.nn.Linear:
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

            elif type(module) == torch.nn.GRU:
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='relu')


        self.apply(weights_initialization)



    # hidden state initialization with zeros for first gru loop
    def initialize_hidden(self, batch_size, gpu=False): # n_samples should be passed as a parameter, allows more freedom to train with different batch sizes
        
        if gpu:
            hidden = torch.zeros(1, batch_size, self.g_ar.hidden_size).cuda()
        else:
            hidden = torch.zeros(1, batch_size, self.g_ar.hidden_size)
            
        return hidden
    
    
    def forward(self, X, hidden, device):
        
        # n_samples should be passed as a parameter, allows more freedom to train with different batch sizes
        
        batch_size = X.size()[0]
        
        z = self.g_enc(X)
        z = z.transpose(1,2)

        t = torch.randint(1, self.seq_len // self.comp_rate - self.n_timesteps, size=(1,))
#         print('random time sample: ',t)
        
        c_t = z[:,:t,:]
        c_t, hidden = self.g_ar(c_t)
        c_t = c_t[:,t-1,:].view(batch_size, 256)

        loss = 0
        acc = []

        for k in range(self.n_timesteps):

            w_k = self.W[k]
            y_k = w_k(c_t)

            z_k = z[:,t+k,:].view(batch_size,512)      

            f_k = torch.mm(z_k, y_k.transpose(0,1))
            
            # compute loss
            loss_k = self.log_softmax(f_k)
            loss_k = torch.diagonal(loss_k)
            loss_k = torch.sum(loss_k)

            loss += loss_k
            
            # compute correct output
            pred_k = self.softmax(f_k)
            pred_k = torch.argmax(pred_k, dim=0)
            
            gt_k = torch.arange(0, batch_size, device=device)
#             print(gt_k.device)
            
            corr_k = torch.eq(pred_k,gt_k)
            corr_k = torch.sum(corr_k)
            acc_k = corr_k.item()/batch_size
            
            acc.append(acc_k)
            

        loss /= -1*batch_size*self.n_timesteps

        return loss, acc, hidden
    
    
    def predict(self, X, hidden):
        
        z = self.g_enc(X)
        z = z.transpose(1,2)
        c_t, hidden = self.g_ar(z)
        
        return c_t[:,-1,:], hidden



class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, n_labels):
        
        super(MLP, self).__init__()
        
        self.mlp = torch.nn.Sequential(
            
            torch.nn.Linear(input_dim,hidden_dim,bias=False),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,n_labels,bias=False)
        )
        
        def weights_initialization(module):

            if type(module) == torch.nn.BatchNorm1d:
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)

            elif type(module) == torch.nn.Linear:
                torch.nn.init.kaiming_normal_(module.weight, 
                                              mode='fan_out', 
                                              nonlinearity='relu')
        
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        
        self.apply(weights_initialization)
        
        
        
    def forward(self,X):
        
        y = self.mlp(X)
        y = self.log_softmax(y)
        
        return y
