import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    
    def __init__(self, vocab_size, embed_size,filter_sizes = [1,2,3],\
                    num_filters=50,embedding_matrix=None):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)


    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)  
        return logit
