from datalist import VideoDataset
import torch
from model import C3D
from torch.utils.data import DataLoader


def test():
    model = C3D()
    model.load_state_dict(torch.load('parameter8.pth', map_location=torch.device('cpu')), False)
    model.eval()
    test_data = VideoDataset("test")
    test_loader = DataLoader(test_data,batch_size=1,shuffle=True)
    y = 0
    n = 0
    fn = 0
    
    with torch.no_grad():
        for a, b in test_loader:
            out = model(a)
            accuracy = torch.max(out,1)[1].numpy() == b.numpy()
            print(b, out)
            if b == 1 and torch.max(out,1)[1].numpy() == 0:
                fn+=1
                print(fn)
            if accuracy == 1:
                y+=1
            else:
                n+=1
            print(y,n,y/(y+n))
            print(fn)
            print("--------------------------------")


'''
            accuracy = torch.max(out,1)[1].numpy() == b.numpy()

            if b == 1 and torch.max(out,1)[1].numpy() == 0:
                fn+=1
                print(fn)
            if accuracy == 1:
                y+=1
            else:
                n+=1
            print(y,n,y/(y+n))

print('| test accuracy: %.2f' % accuracy.mean())
'''