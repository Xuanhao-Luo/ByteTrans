
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from collections import Counter

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.cuda.empty_cache() 

def load_and_split_data(file_path, separator):
    with open(file_path, 'rb') as file:
        content = file.read()
    packets = content.split(separator)  
    return packets


def clean_packets(packets):
    return [packet for packet in packets if packet]  


def prepare_sequences(packets, sequence_length, default_step=32, pad_value=257, end_of_packet_marker=256):
    samples = []
    for packet in packets:
        
        packet_data = np.frombuffer(packet, dtype=np.uint8).astype(np.int16)
        
        packet_data = np.append(packet_data, end_of_packet_marker)
        
        if len(packet_data) < sequence_length + 1:
            padding_needed = sequence_length + 1 - len(packet_data)
            packet_data = np.append(packet_data, np.full(padding_needed, pad_value, dtype=np.int16))
        
        step = min(default_step, len(packet_data))
        
        samples.extend([packet_data[i:i+sequence_length+1] for i in range(0, len(packet_data) - sequence_length, step)])
    return samples


class PacketDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.LongTensor(sample[:-1])
        target_seq = torch.LongTensor(sample[1:])
        return input_seq, target_seq


separator = b'\x00\xFF\x00\xFF' 
packets = load_and_split_data('byte_transformer/data_train/18192.bin', separator)
packets = clean_packets(packets)
SEQUENCE_LENGTH = 256  
step = 256
samples = prepare_sequences(packets, SEQUENCE_LENGTH, step)
dataset = PacketDataset(samples)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)


vocab_size = 258


for input_seq, target_seq in dataloader:
    print(input_seq, target_seq)  
    break  


def generate_square_subsequent_mask(sz):

    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    
class TextGen(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super(TextGen, self).__init__()
        self.pos_encoder = PositionalEncoding(max_len=SEQUENCE_LENGTH, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.kv_cache = None  
        
    def forward(self, x, use_cache=False):
        emb = self.emb(x)
        
        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        x = self.pos_encoder(emb)        
        x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)
        return out


import torch.nn.utils as utils
from torch.optim.lr_scheduler import LambdaLR
epochs = 50
learning_rate = 0.001
warmup_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextGen(
    vocab_size=vocab_size, 
    embed_dim=128,
    num_layers=6, 
    num_heads=8,
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda)


def train(model, epochs, dataloader, criterion, optimizer, scheduler, save_path):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        i = 0
        for input_seq, target_seq in dataloader:
            # start_time = time.time()  
            i +=1
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            outputs = model(input_seq)
            
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, vocab_size)
            loss = criterion(outputs, target_seq)
            
            optimizer.zero_grad()
            loss.backward()
            
            
            utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.detach().cpu().numpy()
            
            # end_time = time.time()  
            # batch_time = end_time - start_time  
                        
            # print(f"Batch {i}, Time: {batch_time:.3f} sec")  
        

        scheduler.step()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} loss: {epoch_loss:.3f}")
        
        if epoch == 0:
            print("Total batches: ", i)
        torch.cuda.empty_cache() 
        

    torch.save(model.state_dict(), save_path)


save_path = 'byte_transformer/model_pth/18192_5M.pth'
# train(model, epochs, dataloader, criterion, optimizer, scheduler, save_path)


def load_model(model_path, vocab_size, embed_dim, num_layers, num_heads):
    model = TextGen(vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# load model
model_path = 'byte_transformer/model_pth/18192_5M.pth'
model = load_model(model_path, vocab_size, 128, 6, 8)

def read_byte_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    byte_sequences = [list(map(int, line.strip().split())) for line in lines]
    return byte_sequences

def return_byte_vector(byte_sequence):
    
    input_tensor = torch.LongTensor(byte_sequence).unsqueeze(0)
    return input_tensor


def predict_and_record_ranks(model, byte_sequence):
    model.eval()
    sequence = [byte_sequence[0]]  
    ranks = [0]  

    
    for actual_byte in byte_sequence[1:]:
        input_tensor = return_byte_vector(sequence[-SEQUENCE_LENGTH:])  
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            predictions = model(input_tensor)
        
        
        probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

        
        rank = (sorted_indices.squeeze() == actual_byte).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)  
        sequence.append(actual_byte) 

    return ranks

def write_ranks_to_file(byte_sequence, ranks, output_file_path):
    with open(output_file_path, 'a') as file:
        
        file.write(f"{byte_sequence[0]} ")
        
        rank_strings = ' '.join(map(str, ranks[1:]))  
        file.write(rank_strings + '\n')

def main(input_file_path, output_file_path, model_path):
    # model = load_model(model_path, vocab_size=256, embed_dim=128, num_layers=6, num_heads=8)
    byte_sequences = read_byte_file(input_file_path)
    all_ranks = []

    for byte_sequence in byte_sequences:
        ranks = predict_and_record_ranks(model, byte_sequence)
        all_ranks.append(ranks)
        write_ranks_to_file(byte_sequence, ranks, output_file_path)


# main('byte_transformer/20080.txt', 'byte_transformer/output_5M.txt', model_path)
# main('byte_transformer/test/txt/20496.txt', 'byte_transformer/test/rank_5M/rank20496.txt', model_path)
main('byte_transformer/test/txt/20240.txt', 'byte_transformer/test/rank_5M/rank20240.txt', model_path)
main('byte_transformer/test/txt/19984.txt', 'byte_transformer/test/rank_5M/rank19984.txt', model_path)
main('byte_transformer/test/txt/19728.txt', 'byte_transformer/test/rank_5M/rank19728.txt', model_path)
main('byte_transformer/test/txt/19472.txt', 'byte_transformer/test/rank_5M/rank19472.txt', model_path)
main('byte_transformer/test/txt/19216.txt', 'byte_transformer/test/rank_5M/rank19216.txt', model_path)
main('byte_transformer/test/txt/18960.txt', 'byte_transformer/test/rank_5M/rank18960.txt', model_path)
main('byte_transformer/test/txt/18704.txt', 'byte_transformer/test/rank_5M/rank18704.txt', model_path)
main('byte_transformer/test/txt/18448.txt', 'byte_transformer/test/rank_5M/rank18448.txt', model_path)
main('byte_transformer/test/txt/18192.txt', 'byte_transformer/test/rank_5M/rank18192.txt', model_path)
