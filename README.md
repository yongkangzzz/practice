# Part2 - regression - API
Regressor(self,x, nb_epoch = 100, batch_size = 64, learning_rate = 0.001,layer1_nodes = 300,layer2_nodes = 800):  

        Initialization

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

            - batch_size{int} -- batch size in the model

            - learning_rate{int} -- learning rate in the model

            - layer1_nodes{int} --  number of nodes on the first hidden layer

            - layer1_nodes{int} -- number of nodes on the second hidden layer


fit(self, x, y):  

        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.


