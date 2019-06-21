from DQN import DQN

class DQNetwork:

    
    def __init__(self, actions, input_shape,
                 minibatch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.9):
                 
        self.actions = actions
        self.discount_factor = discount_factor  
        self.minibatch_size = minibatch_size 
        self.learning_rate = learning_rate 

        self.model = DQN()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, batch, target_network):
        """
        Takes as input a batch of transitions and updates the network
        """
        batch_input = torch.zeros(len(batch), **batch[0].source.shape)
        y_true = torch.zeros(len(batch), self.actions)

        for i, experience in enumerate(batch):
            x_train[i] = experience.state    



        loss = self.criterion(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        


    def predict(self, state):
        prediction = self.model.forward(state)
        if len(prediction.shape) > 1:
            prediction = prediction.squeeze()
        return prediction

    def save(self):
        pass

    def load(self):
        pass


