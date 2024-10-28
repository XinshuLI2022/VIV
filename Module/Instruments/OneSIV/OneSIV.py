import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from utils import set_seed, cat

class OneSIV(nn.Module):
    def __init__(self, z_dim, x_dim, t_dim, dropout):
        super(OneSIV, self).__init__()

        t_input_dim, y_input_dim = z_dim+x_dim, t_dim+x_dim

        self.t_net = nn.Sequential(nn.Linear(t_input_dim, 1280),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(1280, 320),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(320, 32),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(32, 1))

        self.y_net = nn.Sequential(nn.Linear(y_input_dim, 1280),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(1280, 320),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(320, 32),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(32, 1))
        
    def forward(self, z, x, t):   
        pred_t = self.t_net(cat([z,x]))
        yt_input = torch.cat((pred_t,x), 1)
        pred_yt = self.y_net(yt_input)
        
        return pred_t, pred_yt

def run(exp, data, train_dict, log, device, resultDir, others):
    set_seed(train_dict['seed'])
    print(f"Run {exp}/{train_dict['reps']}")

    data.cuda()
    train_loader = DataLoader(data.train, batch_size=train_dict['batch_size'])

    OneSIV_dict = {
            'z_dim':data.train.z.shape[1], 
            'x_dim':data.train.x.shape[1], 
            't_dim':train_dict['t_dim'], 
            'dropout':train_dict['dropout'],
        }

    net = OneSIV(**OneSIV_dict)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=train_dict['learning_rate'], betas=(train_dict['beta1'], train_dict['beta2']),eps=train_dict['eps'])
    t_loss = torch.nn.MSELoss()
    y_loss = torch.nn.MSELoss()

    def estimation(data):
        return net.y_net(cat([data.t-data.t, data.x])), net.y_net(cat([data.t, data.x])), net.y_net(cat([data.e, data.x]))

    for epoch in range(train_dict['epochs']):
        net.train()

        for idx, inputs in enumerate(train_loader):
            z = inputs['z'].to(device)
            x = inputs['x'].to(device)
            t = inputs['t'].to(device)
            y = inputs['y'].to(device)

            pred_t, pred_y = net(z,x,t)
            loss = train_dict['w1'] * y_loss(pred_y, y) + train_dict['w2'] * t_loss(pred_t, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        if (train_dict['verbose'] >= 1 and epoch % train_dict['show_per_epoch'] == 0 ) or epoch == train_dict['epochs']-1:
            pred_test_y0, pred_test_y,pred_test_yt = estimation(data.test)
            print(f'Epoch {epoch} for y : {y_loss(pred_test_y, data.test.g)}, {y_loss(pred_test_y, data.test.v)}, {y_loss(pred_test_y, data.test.y)}. ')
            print(f'Epoch {epoch} for y0 : {y_loss(pred_test_y0, data.test.m[:,0])}, {y_loss(pred_test_y0, data.test.m[:,1])}, {y_loss(pred_test_y0, data.test.m[:,2])}. ')
            print(f'Epoch {epoch} for yt : {y_loss(pred_test_yt, data.test.b)}, {y_loss(pred_test_yt, data.test.f)}, {y_loss(pred_test_yt, data.test.h)}. ')
    return estimation