# Part2 - regression - API

## Regressor class
**Regressor(__self__,x, nb_epoch = 100, batch_size = 64, learning_rate = 0.001,layer1_nodes = 300,layer2_nodes = 800):** 

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
        
**fit(self, x, y):**

        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.  
            
**predict(self, x):**

        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).  
            
**score(self, x, y):**  

        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            - MSE{float} -- normalized mean square error on x

            -RMSE{float} -- non-normalized root mean square error on x
 
## Hyper-parameter searching

**RegressorHyperParameterSearch(x_train, y_train, x_val, y_val):**  
    
    Performs a hyper-parameter for fine-tuning the regressor implemented in the Regressor class.

    Arguments:
            - x_train {pd.DataFrame} -- Training input array 
            - y_train {pd.DataFrame} -- Training output array
            - x_val {pd.DataFrame} -- Validation input array
            - y_val {pd.DataFrame} -- Training output array
        
    Returns:
        best_hp_list -- list including best epochs, batch_size, lr, layer1_nodes and layer2_nodes
        
## An example pipeline

        x_train,y_train,x_test,y_test,x_val,y_val
        regressor = Regressor(x_train)  #create a regressor
        regressor.fit(x_train, y_train) # fit the regressor
        pred = regressor.predict(x_test,y_test)
        MSE,RMSE = regressor.score(x_test,y_test)
        best_hp_list = RegressorHyperParameterSearch(x_train, y_train, x_val, y_val)
        save_regressor(regressor)
