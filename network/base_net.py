import torch
import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in) 
        q = self.fc2(h)
        return q, h

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.model = torch.models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.args.n_agents * self.args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.model(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class CRNN_9(nn.Module):
    def __init__(self, args):
        # design for fov=9
        super(CRNN_9, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1)
      # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
      # self.bn2 = nn.BatchNorm2d(32)
        self.mlp1 = nn.Linear(2+args.n_agents+args.n_actions, 10)
        self.rnn = nn.GRUCell(5*5*32+10, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs, hidden_state):
        pixel, vec = torch.split(
            inputs, [self.args.fov*self.args.fov*4, self.args.n_agents+self.args.n_actions+2], dim=1)
        pixel = pixel.reshape((-1, 4, self.args.fov, self.args.fov))
        pixel = f.relu(self.conv1(pixel))
        pixel = f.relu(self.conv2(pixel))
        pixel = pixel.reshape((-1, 5*5*32))
        vec = f.relu(self.mlp1(vec))  # (batch,10)
        x = torch.cat([pixel, vec], dim=1)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc1(h)
        return q, h

class CRNN_5(nn.Module):
    def __init__(self, args):
        # design for fov=5
        super(CRNN_5, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3)
      # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
      # self.bn2 = nn.BatchNorm2d(32)
        self.mlp1 = nn.Linear(2+args.n_agents+args.n_actions, 10)
        self.rnn = nn.GRUCell(1*1*32+10, args.rnn_hidden_dim) # fov9: (5*5*32+10) fov5: (batch,3*3*32+10)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs, hidden_state):
        pixel, vec = torch.split(
            inputs, [self.args.fov*self.args.fov*4, self.args.n_agents+self.args.n_actions+2], dim=1)
        pixel = pixel.reshape((-1, 4, self.args.fov, self.args.fov))
        pixel = f.relu(self.conv1(pixel))
        pixel = f.relu(self.conv2(pixel))
        pixel = pixel.reshape((-1, 1*1*32))  # fov9: (batch,800) fov5: (batch,288)
        vec = f.relu(self.mlp1(vec))  # (batch,10)
        x = torch.cat([pixel, vec], dim=1)  # fov9: (batch,810) fov5: (batch,298)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc1(h)
        return q, h

class CRNN_Inception(nn.Module):
    def __init__(self, args):
        # design for fov=5
        super(CRNN_Inception, self).__init__()
        self.args = args
        # first path
        # self.conv1 = nn.Conv2d(4, 16, kernel_size=1)
        # second path
        # self.conv2 = nn.Conv2d(4, 24, kernel_size=1)
        self.conv2_1 = nn.Conv2d(4, 32, kernel_size=(1,5))
        self.conv2_2 = nn.Conv2d(4, 32, kernel_size=(5,1))
        # third path
        # self.conv3 = nn.Conv2d(4, 24, kernel_size=1)
        # self.conv3_1 = nn.Conv2d(4, 28, kernel_size=(1,3))
        # self.conv3_1_1 = nn.Conv2d(28, 32, kernel_size=(3,1))
        self.conv3_1_1_1 = nn.Conv2d(4, 32, kernel_size=(2,2))
        self.conv3_1_1_2 = nn.Conv2d(32, 32, kernel_size=(2,2))
        self.conv3_1_1_3 = nn.Conv2d(32, 64, kernel_size=(2,2))

        self.mlp1 = nn.Linear(6+args.n_agents+args.n_actions, 10)
        self.rnn = nn.GRUCell(2*2*64+2*5*32+10, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # attention code
        # self.attention_fc = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        pixel, vec = torch.split(
            inputs, [self.args.fov*self.args.fov*4, self.args.n_agents+self.args.n_actions+6], dim=1)
        pixel = pixel.reshape((-1, 4, self.args.fov, self.args.fov))
        # first path
        # pixel_1 = f.relu(self.conv1(pixel))
        # second path
        # pixel_2 = f.relu(self.conv2(pixel))
        pixel_2_1 = f.relu(self.conv2_1(pixel))
        pixel_2_2 = f.relu(self.conv2_2(pixel))
        # thrid path
        # pixel_3 = f.relu(self.conv3(pixel))
        # pixel_3 = f.relu(self.conv3_1(pixel_3))
        # pixel_3 = f.relu(self.conv3_1_1(pixel_3))
        pixel_3_1 = f.relu(self.conv3_1_1_1(pixel))
        pixel_3_2 = f.relu(self.conv3_1_1_2(pixel_3_1))
        pixel_3_3 = f.relu(self.conv3_1_1_3(pixel_3_2))

        # pixel_1 = pixel_1.reshape((-1, 5*5*16))  # fov9: (batch,800) fov5: (batch,32)

        pixel_2_1 = pixel_2_1.reshape((-1, 5*1*32))
        pixel_2_2 = pixel_2_2.reshape((-1, 5*1*32))
        
        pixel_3_3 = pixel_3_3.reshape((-1, 2*2*64))
        # pixel_3_2 = pixel_3_2.reshape((-1, 3*3*16))

        vec = f.relu(self.mlp1(vec))  # (batch,10)
        x = torch.cat([pixel_2_1, pixel_2_2, pixel_3_3, vec], dim=1)  # fov9: (batch,810) fov5: (batch,42)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        # attention code
        # weight = f.softmax(self.attention_fc(h), dim=1)
        # att = torch.matmul(h, torch.diag(weight[0]))*42

        q = self.fc1(h)
        return q, h

# class CRNN_Inception(nn.Module):
#     def __init__(self, args):
#         # design for fov=5
#         super(CRNN_Inception, self).__init__()
#         self.args = args
#         # first path
#         # self.conv1 = nn.Conv2d(4, 16, kernel_size=1)
#         # second path
#         # self.conv2 = nn.Conv2d(4, 24, kernel_size=1)
#         # self.conv2_1 = nn.Conv2d(4, 32, kernel_size=(1,self.args.chip_size))
#         # self.conv2_2 = nn.Conv2d(4, 32, kernel_size=(self.args.chip_size,1))
#         # third path
#         # self.conv3 = nn.Conv2d(4, 24, kernel_size=1)
#         # self.conv3_1 = nn.Conv2d(4, 28, kernel_size=(1,3))
#         # self.conv3_1_1 = nn.Conv2d(28, 32, kernel_size=(3,1))
#         self.conv3_1_1_1 = nn.Conv2d(4, 16, kernel_size=(3,3))
#         self.conv3_1_1_2 = nn.Conv2d(16, 32, kernel_size=(3,3))
#         self.conv3_1_1_3 = nn.Conv2d(32, 64, kernel_size=(3,3))
#         self.conv3_1_1_4 = nn.Conv2d(64, 128, kernel_size=(3,3))
#         # self.conv3_1_1_5 = nn.Conv2d(128, 256, kernel_size=(3,3))

#         self.mlp1 = nn.Linear(2+args.n_agents+args.n_actions, 10)
#         self.rnn = nn.GRUCell(522, args.rnn_hidden_dim)
#         self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

#         # attention code
#         # self.attention_fc = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

#     def forward(self, inputs, hidden_state):
#         pixel, vec = torch.split(
#             inputs, [self.args.chip_size*self.args.chip_size*4, self.args.n_agents+self.args.n_actions+2], dim=1)
#         pixel = pixel.reshape((-1, 4, self.args.chip_size, self.args.chip_size))
#         # first path
#         # pixel_1 = f.relu(self.conv1(pixel))
#         # second path
#         # pixel_2 = f.relu(self.conv2(pixel))
#         # pixel_2_1 = f.relu(self.conv2_1(pixel))
#         # pixel_2_2 = f.relu(self.conv2_2(pixel))
#         # thrid path
#         # pixel_3 = f.relu(self.conv3(pixel))
#         # pixel_3 = f.relu(self.conv3_1(pixel_3))
#         # pixel_3 = f.relu(self.conv3_1_1(pixel_3))
#         pixel_3_1 = f.relu(self.conv3_1_1_1(pixel))
#         pixel_3_2 = f.relu(self.conv3_1_1_2(pixel_3_1))
#         pixel_3_3 = f.relu(self.conv3_1_1_3(pixel_3_2))
#         pixel_3_4 = f.relu(self.conv3_1_1_4(pixel_3_3))
#         # pixel_3_5 = f.relu(self.conv3_1_1_5(pixel_3_4))

#         # pixel_1 = pixel_1.reshape((-1, 5*5*16))  # fov9: (batch,800) fov5: (batch,32)

#         # pixel_2_1 = pixel_2_1.reshape((-1, self.args.chip_size*1*32))
#         # pixel_2_2 = pixel_2_2.reshape((-1, self.args.chip_size*1*32))
        
#         pixel_3_4 = pixel_3_4.reshape((-1, 512))
#         # pixel_3_2 = pixel_3_2.reshape((-1, 3*3*16))

#         vec = f.relu(self.mlp1(vec))  # (batch,10)
#         # x = torch.cat([pixel_2_1, pixel_2_2, pixel_3_3, vec], dim=1)  # fov9: (batch,810) fov5: (batch,42)
#         x = torch.cat([pixel_3_4, vec], dim=1)
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)

#         # attention code
#         # weight = f.softmax(self.attention_fc(h), dim=1)
#         # att = torch.matmul(h, torch.diag(weight[0]))*42

#         q = self.fc1(h)
#         return q, h


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
