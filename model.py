import numpy as np
import torch
from torch import nn
from torch import autograd, optim
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel

class ProtoNet(nn.Module):
    def __init__(self, sentence_encoder, dot=False, max_length = 128,args = None):
        '''
        sentence_encoder: Sentence encoder
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.conv_num = args.conv_num
        self.conv_size = args.conv_size
        self.sentence_encoder = sentence_encoder
        # self.sentence_encoder = nn.DataParallel(sentence_encoder)
        self.cost = nn.CrossEntropyLoss()
        self.dot = dot
        self.max_length = max_length
        self.hidden_size = args.hidden_size
        self.fc = torch.nn.Linear((self.hidden_size - self.conv_size + 1) * self.conv_num, self.hidden_size)
        self.maxpool = torch.nn.MaxPool1d(2)
        self.avgpool = torch.nn.AvgPool1d(2)
        self.drop_pool = torch.nn.Dropout(args.dropout)

    def __dist__(self, x, y, dim, att=None):
        if self.dot:
            return (x * y).sum(dim)
        elif att == None:
            return -(torch.pow(x - y, 2)).sum(dim)
        else :
            return -(torch.pow(x - y, 2) * att).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def get_entity(self,x,eh,eh_end):
        head_entity = []
        for i in range(len(eh)):
            h = []
            if (eh[i] >= eh_end[i]):
                eh_end[i] = eh[i] + 1
            for pos in range(eh[i], eh_end[i]):
                try:
                    h.append(x[i, pos].unsqueeze(0))
                except IndexError:
                    h.append(x[i, -1].unsqueeze(0))
            h = torch.cat(h, dim=0)
            head_entity.append(torch.mean(h, dim=0).unsqueeze(0))
        head_entity = torch.cat(head_entity, dim=0)
        return head_entity

    def feature_abstract(self,head_entity, tail_entity, x,hidden_size):
        head_entity = head_entity.view(-1, 1, self.conv_num, 1, self.conv_size)# (trainN * K, 1, num, size)
        head_entity = head_entity.view(-1, 1, 1, self.conv_size)
        tail_entity = tail_entity.view(-1, 1, self.conv_num, 1, self.conv_size)# (trainN * K, 1, num, size)
        tail_entity = tail_entity.view(-1, 1, 1, self.conv_size)
        x = x.view(1, -1, 1, hidden_size)
        h_x = F.relu(F.conv2d(x, head_entity, groups=x.size(1)) )   # (1, trainN * K * num, 1, D - size + 1)
        t_x = F.relu(F.conv2d(x, tail_entity, groups=x.size(1)) )   # (1, trainN * K * num, 1, D - size + 1)
        ht_x =torch.cat([h_x,t_x] , dim = 3)
        ht_x=ht_x.squeeze(2)
        ht_x_avg = self.avgpool(ht_x).unsqueeze(2)
        ht_x = self.drop_pool(ht_x_avg)
        ht_x = ht_x.view(x.size(1), -1)  #(trainN * K, num * ( D - size + 1 ) )
        ht_x = self.fc(ht_x)
        # ht_x = F.sigmoid(ht_x)# select
        return ht_x


    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        num ; num of convolutions
        size : size of convolutions
        '''

        all_support, support_emb = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        all_query, query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        hidden_size = support_emb.size(-1)


        support_head_entity = self.get_entity(all_support, support['eh'], support['eh_end'])# (N, D)
        support_tail_entity = self.get_entity(all_support, support['et'], support['et_end'])# (N, D)
        query_head_entity = self.get_entity(all_query, query['eh'], query['eh_end'])# (N, D)
        query_tail_entity = self.get_entity(all_query, query['et'], query['et_end'])# (N, D)

        support = self.feature_abstract(support_head_entity,support_tail_entity,support_emb,hidden_size)  #(trainN * K, D)
        support = support.view(-1, N, K, hidden_size)
        query = self.feature_abstract(query_head_entity,query_tail_entity,query_emb,hidden_size)  #(total_Q * K, D)
        query = query.view(-1, total_Q, hidden_size)

        # Prototypical Networks
        support = torch.mean(support, 2)  # Calculate prototype for each class
        logits = self.__batch_dist__(support, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1],
                           2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class BERTEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        all, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
        return all,x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        # head tail
        word = indexed_tokens
        word = torch.tensor(word).long()
        word = word.view(-1,self.max_length)
        eh = np.argmax((word == 1).cpu(),1)
        eh_end = np.argmax((word == 3).cpu(),1)
        et = np.argmax((word == 2).cpu(),1)
        et_end = np.argmax((word == 4).cpu(), 1)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, eh,eh_end,et,et_end
