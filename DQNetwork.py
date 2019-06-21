from DQN import DQN
import torch
import copy

class DQNetwork:
    def __init__(self, actions, input_shape,
                 minibatch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.9):
                 
        self.actions = actions
        self.discount_factor = discount_factor  
        self.minibatch_size = minibatch_size 
        self.learning_rate = learning_rate 

        self.model = DQN().double()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, batch, target_network):
        """
        Takes as input a batch of transitions and updates the network
        """
        batch_input = torch.zeros(len(batch), 4, 84, 84).double()
        y_true = torch.zeros(len(batch), self.actions).double()

        for i, experience in enumerate(batch):
            current_state = experience.state 
            
            if type(current_state) != torch.Tensor:
                 current_state = torch.from_numpy(current_state)
            if len(current_state.shape) == 3:
                current_state = current_state[None, :, :, :]
            batch_input[i]     = current_state    
            new_state      = experience.new_state
            new_state_pred = target_network.predict(new_state)
            new_q_value    = torch.max(new_state_pred)

            new_y_true = torch.zeros(self.actions)
            if experience.final:
                new_y_true[experience.action] = experience.reward
            else:
                new_y_true[experience.action] = experience.reward + \
                                                self.discount_factor * new_q_value

            y_true[i] = new_y_true

        y_pred = self.model.forward(batch_input)
        loss = self.criterion(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        


    def predict(self, state):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state)
        if len(state.shape) == 3:
            state = state[None, :, :, :]

        prediction = self.model.forward(state)
        if len(prediction.shape) > 1:
            prediction = prediction.squeeze()
        return prediction

    def save(self, epoch, epsilon, append):
        filename = f'checkpoints/checkpoint_{str(epoch)}_{str(epsilon)}_{append}'
        checkpoint_dict = {
            'optimizer' : self.optimizer.state_dict(),
            'model'     : self.model.state_dict(),
            'epoch'     : epoch,
            'epsilon'   : epsilon
        }
        torch.save(checkpoint_dict, filename)


    def load(self, filename):
        filename = 'checkpoints/'+filename
        checkpoint_dict = torch.load(filename)
        epoch           = checkpoint_dict['epoch']
        epsilon         = checkpoint_dict['epsilon']
        self.model.load_state_dict(checkpoint_dict['model'])
        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        return epoch