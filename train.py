from model import C3D
from datalist import VideoDataset
import torch
from torch.utils.data import DataLoader


def train_model():
    train_data = VideoDataset("train")
    test_data = VideoDataset("test")
    train_loader = DataLoader(train_data,batch_size=20,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=20,shuffle=True)
    device = torch.device('cuda:0')
    model = C3D(2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        for i,(x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            out = model(x) # torch.Size([128,10])
            # 获取损失
            loss = loss_func(out,y)
            # 使用优化器优化损失
            optimizer.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            optimizer.step() # 将参数更新值施加到net的parmeters上
            if i % 50 == 0:
                for a,b in test_loader:
                    a = a.to(device)
                    b = b.to(device)
                    a = torch.unsqueeze(a, 1)
                    out = model(a)
                    # print('test_out:\t',torch.max(out,1)[1])
                    # print('test_y:\t',test_y)
                    accuracy = torch.max(out,1)[1].cpu().numpy() == b.cpu().numpy()
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy.mean())
                    break
    torch.save(C3D,'c3d.pkl')