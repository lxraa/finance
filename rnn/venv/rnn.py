import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
BATCH_SIZE = 100
DATA_SIZE = 10

class Stock(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            # 日期，收盘价，最高价，最低价，开盘价，前收盘，涨跌额，涨跌幅，换手率，成交量，成交金额
            reader = csv.reader(f)
            #去掉日期列
            list = [d[1:] for d in reader]
        # 去掉第一行训练集为除掉最后一天的数据
        self.datas = np.array(list[1:-1], dtype=float)
        labels = [[item[6]] for item in list[2:]]
        self.labels = np.array(labels, dtype=float)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.datas[index])
        label = torch.FloatTensor(self.labels[index])
        return data.view(-1, DATA_SIZE), label.view(-1, 1)

    def getAllData(self):
        return torch

trains = Stock("./601857.csv")
dataloader = DataLoader(dataset=trains, batch_size=BATCH_SIZE, shuffle=False)
dataloader2 = DataLoader(dataset=trains, batch_size=trains.__len__(), shuffle=False)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, input):
        batch_size = input.shape[0]
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out
'''
    
        |-------------|
10->    |             |     ->1
        |-------------|
    以100个为一批训练
'''
net = Model(DATA_SIZE, 1, BATCH_SIZE)
#涨跌额是数值，采用平方差计算损失
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

if __name__ == "__main__":
    for epoch in range(15):
        for index, d in enumerate(dataloader):
            trains, labels = d[0], d[1]
            optimizer.zero_grad()
            outputs = net(d[0])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(loss)
        # print("loss: " + loss.value)