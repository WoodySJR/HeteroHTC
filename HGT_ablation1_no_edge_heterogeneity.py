import torch, pandas as pd, numpy as np, math
from torch import nn
from pytorch_pretrained_bert import BertModel
import warnings
from torch.nn import BatchNorm1d
warnings.filterwarnings("ignore")

device = "cuda:0"

# define the network
class HGT(nn.Module):
    
    def __init__(self, hidden_dim, num_heads, num_labels, num_1, num_2, num_3, 
                 att_mask, label_tokens, num_hgt_layers):
        super(HGT, self).__init__()
        
        self.hidden_dim = hidden_dim # dimension of hidden state in attention
        self.num_heads = num_heads # number of attention heads
        self.hidden_dim_per_head = int(hidden_dim/num_heads)
        self.num_labels = num_labels # number of labels
        self.num_1 = num_1
        self.num_2 = num_2
        self.num_3 = num_3
        self.att_mask = att_mask.to(device) # 0/1
        self.att_mask[self.att_mask==0] = -1*math.inf
        self.att_mask[self.att_mask==1] = 0
        self.label_tokens = label_tokens.to(device)
        self.num_hgt_layers = num_hgt_layers
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # Bert for text and label_desc encoding
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        #self.label_encoder = BertModel.from_pretrained("bert-base-chinese")
        
        # transform from Bert output to attention input
        self.text_transform = nn.Linear(768, self.hidden_dim)
        self.label_transform = nn.Linear(768, self.hidden_dim)
        
        # HGT params
        ## Key
        self.WK1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.WK2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.WK3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        ## Query
        self.WQ1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.WQ2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.WQ3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        ## Value
        self.WV1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.WV2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.WV3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        ## W_phi_attention
        self.PA_p2c = nn.Linear(self.hidden_dim_per_head, self.hidden_dim_per_head, bias=False)
        self.PA_c2p = self.PA_p2c
        self.PA_sib = self.PA_p2c
        ## W_phi_message
        self.PM_p2c = nn.Linear(self.hidden_dim_per_head, self.hidden_dim_per_head, bias=False)
        self.PM_c2p = self.PM_p2c
        self.PM_sib = self.PM_p2c
        ## Aggregation
        self.WA1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.WA2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.WA3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # output layer
        # self.output_1 = nn.Linear(self.hidden_dim*self.num_labels, 5000)
        # self.output_2 = nn.Linear(5000, 2000)
        # self.output_3 = nn.Linear(2000, num_labels)
        self.output = nn.Linear(self.hidden_dim*self.num_labels, self.num_labels)
        
    def forward(self, inputs):
        
        # text and label_desc encoding
        text_encoding, f = self.text_encoder(input_ids = inputs.long(), attention_mask = (inputs>0).long().to(device))
        text_encoding = self.tanh(self.text_transform(text_encoding[-1])) # shape: (batch_size, length, hidden)
        
        _, label_encoding = self.text_encoder(self.label_tokens, torch.zeros_like(self.label_tokens).to(device),
                                             (self.label_tokens>0).float().to(device))
        label_encoding = self.label_transform(label_encoding) # shape: (#labels, hidden)
        
        # structure encoder using HGT
        for l in range(self.num_hgt_layers): 
            
            ## Heterogeneous K
            K1 = self.WK1(label_encoding[0:self.num_1,:])
            K2 = self.WK2(label_encoding[self.num_1:(self.num_1+self.num_2),:])
            K3 = self.WK3(label_encoding[(self.num_1+self.num_2):self.num_labels,:])
            ## Heterogeneous Q
            Q1 = self.WQ1(label_encoding[0:self.num_1,:])
            Q2 = self.WQ2(label_encoding[self.num_1:(self.num_1+self.num_2),:])
            Q3 = self.WQ3(label_encoding[(self.num_1+self.num_2):self.num_labels,:])
            ## Heterogeneous V
            V1 = self.WV1(label_encoding[0:self.num_1,:])
            V2 = self.WV2(label_encoding[self.num_1:(self.num_1+self.num_2),:])
            V3 = self.WV3(label_encoding[(self.num_1+self.num_2):self.num_labels,:])

            ## attention of level 2
            for h in range(self.num_heads):

                Q_temp = Q2[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                K_temp = K1[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                A_temp = (self.PA_p2c(Q_temp)@K_temp.T)/math.sqrt(self.hidden_dim_per_head)
                M_temp = self.PM_p2c(V1[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head])

                K_temp = K2[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                A_temp = torch.cat((A_temp,(self.PA_sib(Q_temp)@K_temp.T)/math.sqrt(self.hidden_dim_per_head)),dim=1)
                M_temp = torch.cat((M_temp,self.PM_sib(V2[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head])), dim=0)

                K_temp = K3[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                A_temp = torch.cat((A_temp,(self.PA_c2p(Q_temp)@K_temp.T)/math.sqrt(self.hidden_dim_per_head)),dim=1)
                M_temp = torch.cat((M_temp,self.PM_c2p(V3[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head])), dim=0)

                M = torch.softmax(A_temp+self.att_mask[self.num_1:(self.num_1+self.num_2),:], dim=1)@M_temp
                if h==0:
                    M2 = M
                else:
                    M2 = torch.cat((M2,M), dim=1)
            M2 = self.WA2(M2)
            
            ## attention of level 1
            for h in range(self.num_heads): 
                
                Q_temp = Q1[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                K_temp = K1[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                A_temp = (self.PA_sib(Q_temp)@K_temp.T)/math.sqrt(self.hidden_dim_per_head)
                M_temp = self.PM_sib(V1[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head])
                
                K_temp = K2[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                A_temp = torch.cat((A_temp,(self.PA_c2p(Q_temp)@K_temp.T)/math.sqrt(self.hidden_dim_per_head)),dim=1)
                M_temp = torch.cat((M_temp,self.PM_c2p(V2[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head])), dim=0)
                
                A_temp = torch.cat((A_temp,torch.zeros(self.num_1, self.num_3).to(device)), dim=1)
                M_temp = torch.cat((M_temp,torch.zeros(self.num_3, self.hidden_dim_per_head).to(device)), dim=0)
                
                M = torch.softmax(A_temp+self.att_mask[0:self.num_1,:], dim=1)@M_temp
                if h==0:
                    M1 = M
                else:
                    M1 = torch.cat((M1,M), dim=1)
            M1 = self.WA1(M1)
            
            ## attention of level 3
            for h in range(self.num_heads):
                
                Q_temp = Q3[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                A_temp = torch.zeros(self.num_3, self.num_1).to(device)
                M_temp = torch.zeros(self.num_1, self.hidden_dim_per_head).to(device)
                
                K_temp = K2[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                A_temp = torch.cat((A_temp,(self.PA_p2c(Q_temp)@K_temp.T)/math.sqrt(self.hidden_dim_per_head)),dim=1)
                M_temp = torch.cat((M_temp,self.PM_p2c(V2[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head])), dim=0)
                
                K_temp = K3[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head]
                A_temp = torch.cat((A_temp,(self.PA_sib(Q_temp)@K_temp.T)/math.sqrt(self.hidden_dim_per_head)),dim=1)
                M_temp = torch.cat((M_temp,self.PM_sib(V3[:,h*self.hidden_dim_per_head:(h+1)*self.hidden_dim_per_head])), dim=0)

                M = torch.softmax(A_temp+self.att_mask[(self.num_1+self.num_2):self.num_labels,:], dim=1)@M_temp
                if h==0:
                    M3 = M
                else:
                    M3 = torch.cat((M3,M), dim=1)
            M3 = self.WA3(M3)
            
            label_encoding = self.tanh(torch.cat((M1,M2,M3), dim=0)) + label_encoding
            
        att_weights = torch.bmm(label_encoding.unsqueeze(0).repeat(inputs.shape[0],1,1), 
                                text_encoding.permute(0,2,1)) # shape: (batch_size, #labels, #words)
        pad_mask = torch.zeros_like(inputs).to(device).float()
        pad_mask[inputs==0] = -1*math.inf
        pad_mask[inputs==101] = -1*math.inf
        pad_mask[inputs==102] = -1*math.inf
        att_weights = torch.softmax(att_weights+pad_mask.unsqueeze(1).repeat(1,self.num_labels,1), dim=2)
        features = torch.bmm(att_weights, text_encoding).reshape(inputs.shape[0], -1)
        # features = self.tanh(self.output_1(features))
        # features = self.tanh(self.output_2(features))
        # output = self.sigmoid(self.output_3(features))
        output = self.sigmoid(self.output(features))
        
        return output