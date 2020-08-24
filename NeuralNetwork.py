from tqdm.notebook import tqdm
import numpy as np


class MultiLayerPerceptron:
    def __init__(self, n_hidden_nodes,
                 hid_activ_fun='sigmoid',
                 out_activ_fun='linear',
                 optimizer='SGD',
                 use_bias=True,
                 learning_rate=0.01,
                 momentum=0.0, #only for SGD
                 regularization=0.0,
                 exp_decay1=0.9, #only for Adam
                 exp_decay2=0.999, #only for Adam
                 delta=10**-8, #only for Adam
                 batch=True,
                 batch_size=None,
                 epochs=100,
                 evaluate=False,
                 init_weigths='glorot',
                 eps=10**-3,
                 verbose=False,
                 seed=None):
        try:
            self.n_hidden_nodes = list(n_hidden_nodes)
        except:
            self.n_hidden_nodes = [n_hidden_nodes]
        
        activation_functions = {'sigmoid':lambda x: 1/(1+np.e**-x),
                                'tanh':np.tanh,
                               'linear':lambda x: x}
        der_activation_functions = {'sigmoid':lambda x: activation_functions['sigmoid'](x)*(1-activation_functions['sigmoid'](x)),
                                    'tanh':lambda x: 1-np.square(np.tanh(x)),
                                   'linear': np.vectorize(lambda x: 1.0)}
          
        if not isinstance(hid_activ_fun, str):
            hid_activ_fun = list(hid_activ_fun)
        else:
            hid_activ_fun = [hid_activ_fun]
    
    
        self.activ_fun = [activation_functions[f] for f in hid_activ_fun] + [activation_functions[out_activ_fun]]
        self.der_activ_fun = [der_activation_functions[f] for f in hid_activ_fun] + [der_activation_functions[out_activ_fun]]
    
        self.optimizer=optimizer
        self.use_bias = use_bias
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.regularization = regularization
        
        self.exp_decay1 = exp_decay1
        self.exp_decay2 = exp_decay2
        self.delta = delta
        
        self.batch = batch
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.evaluate = evaluate
        
        self.init_weigths = init_weigths
        
        self.eps = eps
        
        self.verbose = verbose
        self.seed = seed
        
        
    def initialize_network(self, train_data, train_labels):
        self.n_in_nodes = len(train_data[0])
        try:
            self.n_out_nodes = len(train_labels[0])
        except:
            self.n_out_nodes = 1
            
        layers = [self.n_in_nodes]+self.n_hidden_nodes+[self.n_out_nodes]
        
        #initialize the matrixes of weigths
        np.random.seed(seed=self.seed)
        if self.init_weigths == 'uniform':
            self.weigths = [np.random.rand(layers[i+1],layers[i])-0.5 for i in range(len(layers)-1)]
        elif self.init_weigths == 'normal':
            self.weigths = [np.random.randn(layers[i+1],layers[i]) for i in range(len(layers)-1)]
        elif self.init_weigths == 'fan_in':
            self.weigths = [(np.random.rand(layers[i+1],layers[i])-0.5)*2/np.sqrt(layers[i]) for i in range(len(layers)-1)]
        elif self.init_weigths == 'glorot':
            self.weigths = [(np.random.rand(layers[i+1],layers[i])-0.5)*2*np.sqrt(6)/np.sqrt(layers[i]+layers[i+1]) for i in range(len(layers)-1)]
            
        #initialize the bias
        self.bias = [np.zeros(layers[i]).reshape(-1,1) for i in range(1,len(layers))]
        
        #initialize the list of all the nodes including input and output ones: every element of the list contains an array of node of a layer
        self.nodes = list(range(len(self.weigths)+1))
        
        
    #update the nodes
    def forward_pass(self, input_record):
        self.nodes[0] = np.array(input_record).reshape(-1,1)
        for level in range(len(self.weigths)):
            net = np.dot(self.weigths[level], self.nodes[level]) + self.bias[level]
            self.nodes[level+1] = self.activ_fun[level](net)
        
        
    #compute the gradient for each record with backpropagation and add the result to the deltas for the weigths
    def backpropagation(self, output_value):
        for level in range(len(self.weigths)-1,-1,-1):
            #check if it's the output layer
            if level == len(self.weigths)-1:
                #compute the output error
                error = output_value - self.nodes[level+1]
            else:
                #compute the hidden error backpropagating the previous computed error
                error = np.dot(self.weigths[level+1].T, error)
                
            net = np.dot(self.weigths[level], self.nodes[level]) + self.bias[level]
            self.delta_weights[level] += np.dot(error * self.der_activ_fun[level-1](net), self.nodes[level].T)
            if self.use_bias:
                self.delta_bias[level] += error * self.der_activ_fun[level-1](net)
            
            
    #update weigths with momentum and regularization
    def update_weigths(self):
        #print(self.delta_weights,'\n\n')
        if self.optimizer == 'SGD':
            #compute momentum
            for level in range(len(self.weigths)):
                #compute regularization term
                regularization = 2*self.regularization*np.sign(self.weigths[level])
                #compute momentum term
                momentum_weigths = self.momentum*self.delta_weights_old[level]
                if self.use_bias:
                    momentum_bias = self.momentum*self.delta_bias_old[level]
                    
                self.delta_weights[level] = self.learning_rate * self.delta_weights[level] + momentum_weigths - regularization
                self.weigths[level] += self.delta_weights[level]
                self.delta_weights_old[level] = self.delta_weights[level].copy()
                if self.use_bias:
                    self.delta_bias[level] = self.learning_rate * self.delta_bias[level] + momentum_bias
                    self.bias[level] += self.delta_bias[level]
                    self.delta_bias_old[level] = self.delta_bias[level].copy()
            
        elif self.optimizer == 'Adam':
            for level in range(len(self.weigths)):
                self.first_momentum_w[level] = self.exp_decay1*self.first_momentum_w[level] + (1 - self.exp_decay1)*self.delta_weights[level]
                self.second_momentum_w[level] = self.exp_decay2*self.second_momentum_w[level] + (1 - self.exp_decay2)*self.delta_weights[level]**2
                bias_corr_first_momentum_w = self.first_momentum_w[level]/(1-self.exp_decay1**self.n_updates_weigths)
                bias_corr_second_momentum_w = self.second_momentum_w[level]/(1-self.exp_decay2**self.n_updates_weigths)
                regularization = 2*self.regularization*np.sign(self.weigths[level])
                self.weigths[level] += self.learning_rate*bias_corr_first_momentum_w/(np.sqrt(bias_corr_second_momentum_w)+self.delta)-regularization
                if self.use_bias:
                    self.first_momentum_b[level] = self.exp_decay1*self.first_momentum_b[level] + (1 - self.exp_decay1)*self.delta_bias[level]
                    self.second_momentum_b[level] = self.exp_decay2*self.second_momentum_b[level] + (1 - self.exp_decay2)*self.delta_bias[level]**2
                    bias_corr_first_momentum_b = self.first_momentum_b[level]/(1-self.exp_decay1**self.n_updates_weigths)
                    bias_corr_second_momentum_b = self.second_momentum_b[level]/(1-self.exp_decay2**self.n_updates_weigths)
                    self.bias[level] += self.learning_rate*bias_corr_first_momentum_b/(np.sqrt(bias_corr_second_momentum_b)+self.delta)
                    
    def compute_metrics(self, output_values, target_values, valid_set = False):
        if valid_set:
            n_records = self.n_validation
        else:
            n_records = self.n_train
        MSE = sum([sum([(target_values[i][j]-output_values[i][j])**2 for j in range(self.n_out_nodes)]) for i in range(n_records)])/n_records
        MEE = sum([np.sqrt(sum([(target_values[i][j]-output_values[i][j])**2 for j in range(self.n_out_nodes)])) for i in range(n_records)])/n_records
        
        #compute the sum of the square of the weigths for regularization
        weigths_sum = sum([sum([abs(layer_weigths[i,j]) for i in range(layer_weigths.shape[0]) for j in range(layer_weigths.shape[1])]) for layer_weigths in self.weigths])
        loss = MSE + self.regularization * weigths_sum
        
        #compute accuracy
        if valid_set:
            self.MSE_values_validation.append(MSE)
            self.MEE_values_validation.append(MEE)
            self.loss_values_validation.append(loss)
        else:
            self.MSE_values_train.append(MSE)
            self.MEE_values_train.append(MEE)
            self.loss_values_train.append(loss)
    
    def compute_gradient_norm(self, train_data, train_target):
        for (input_record,output_value) in zip(train_data, train_target):
            output_value = np.array(output_value).reshape(-1,1)
            self.forward_pass(input_record)
            self.backpropagation(output_value)
            
        gradient_norm = 0
        
        for w in self.delta_weights:
            gradient_norm += np.linalg.norm(w)**2
            
        gradient_norm = np.sqrt(gradient_norm)
        
        #set again the matrixes to zeros in order to start to compute new gradient
        self.delta_weights = [np.zeros(W.shape) for W in self.weigths]
        
        return gradient_norm
    
    
    def fit(self, train_data, train_target, validation_data=None, validation_target=None):
        self.n_train = len(train_data)
        if validation_data is not None:
            self.n_validation = len(validation_data)
        #initialize the network
        self.initialize_network(train_data, train_target)
        
        if self.optimizer == 'SGD':
            #initialize the matrixes containing the deltas of previous epoch for the momentum
            self.delta_weights_old = [np.zeros(W.shape) for W in self.weigths]
            if self.use_bias:
                self.delta_bias_old = [np.zeros(b.shape) for b in self.bias]
            
        elif self.optimizer == 'Adam':
            self.first_momentum_w = [np.zeros(W.shape) for W in self.weigths]
            self.second_momentum_w = [np.zeros(W.shape) for W in self.weigths]
            self.first_momentum_b = [np.zeros(b.shape) for b in self.bias]
            self.second_momentum_b = [np.zeros(b.shape) for b in self.bias]
        
        #initialize lists containing values for MSE, loss and accuracy for training and validation set
        self.MSE_values_train = list()
        self.MSE_values_validation = list()
        self.loss_values_train = list()
        self.loss_values_validation = list()
        self.MEE_values_train = list()
        self.MEE_values_validation = list()
        
        #initialize the matrix containing the deltas for updating the weigths
        self.delta_weights = [np.zeros(W.shape) for W in self.weigths]
        if self.use_bias:
            self.delta_bias = [np.zeros(b.shape) for b in self.bias]
        self.n_updates_weigths = 1
        
        #set batch size if batch and adjust regularization otherwise
        if self.batch:
            self.batch_size = len(train_data)
        else:
            self.regularization = self.regularization*self.batch_size/len(train_data)

        #compute first gradient
        self.gradient_values = list()
        first_gradient_norm = self.compute_gradient_norm(train_data, train_target)
        self.gradient_values.append(first_gradient_norm)
        
        #compute first metrics
        if self.evaluate:
            #compute the metrics for training set
            output_values = self.predict(train_data)
            self.compute_metrics(output_values, train_target, valid_set = False)
            
            #compute the metrics for validation set if there is
            if validation_data is not None:
                output_values = self.predict(validation_data)
                self.compute_metrics(output_values, validation_target, valid_set = True)
        
        
        if self.verbose:
            bar= tqdm(range(self.epochs), bar_format='{desc}')
        else:
            bar = range(self.epochs)
            
        for i in bar:
            if self.verbose:
                if self.evaluate:
                    bar.set_description('Epoch {0}/{1}  ||g||/||g0||: {2:.10f} Loss: {3:.10f}'.format(i+1, self.epochs, self.gradient_values[-1]/first_gradient_norm, self.loss_values_train[-1]))
                else:
                    bar.set_description('Epoch {0}/{1}  ||g||/||g0||: {2:.10f}'.format(i+1, self.epochs, self.gradient_values[-1]/first_gradient_norm))
            if self.gradient_values[-1]/first_gradient_norm < self.eps:
                break
            #shuffle the dataset if not batch
            if not self.batch:
                np.random.shuffle(train_data)
                np.random.shuffle(train_target)
                
            #compute forward pass and backpropagation for each record
            for k, (input_record,output_value) in enumerate(zip(train_data, train_target)):
                output_value = np.array(output_value).reshape(-1,1)
                self.forward_pass(input_record)
                self.backpropagation(output_value)
                
                #update weigths
                if k+1 % self.batch_size == 0 or k+1==len(train_data):
                    self.update_weigths()
                    self.n_updates_weigths += 1

                    #set again the matrixes to zeros in order to start to compute new gradient
                    self.delta_weights = [np.zeros(W.shape) for W in self.weigths]
                    if self.use_bias:
                        self.delta_bias = [np.zeros(b.shape) for b in self.bias]

                    #compute the gradient and save the value
                    self.gradient_values.append(self.compute_gradient_norm(train_data, train_target))
            
            if self.evaluate:
                #compute the metrics for training set
                output_values = self.predict(train_data)
                self.compute_metrics(output_values, train_target, valid_set = False)
                
                #compute the metrics for validation set if there is
                if validation_data is not None:
                    output_values = self.predict(validation_data)
                    self.compute_metrics(output_values, validation_target, valid_set = True)
            
            
    def predict(self, data):
        output_values = list()
        for record in data:
            self.forward_pass(record)

            output_values.append(self.nodes[-1].reshape(self.nodes[-1].shape[0]))
            
        return np.array(output_values)
    



#run model with monk dataset
if __name__ == '__main__':
    from pandas import read_csv
    df = read_csv('ML-CUP19-TR.csv', skiprows=7, header=None)
    n_records = len(df)
    
    X = np.array(df.iloc[:,1:-2])
    y = np.array(df.iloc[:,-2:])
    
    #fit standard momentum descent
    NN = MultiLayerPerceptron(n_hidden_nodes=50,
                              hid_activ_fun='sigmoid',
                              learning_rate=8*10**-5,
                              momentum=0.5,
                              regularization=10**-6,
                              init_weigths='glorot',
                              evaluate=True,
                              epochs=100,
                              eps=10**-3)

    NN.fit(X,y)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,5))
    plt.subplot(1,2,1)
    plt.title('Gradient standard momentum descent')
    plt.plot(NN.gradient_values/NN.gradient_values[0])
    plt.xlabel('Epochs',{'fontsize':13})
    
    
    """
    #fit standard momentum descent
    NN = MultiLayerPerceptron(n_hidden_nodes=50,
                              hid_activ_fun='sigmoid',
                              optimizer='Adam',
                              learning_rate=10**-3,
                              exp_decay1=0.9,
                              exp_decay2=0.995,
                              regularization=10**-6,
                              evaluate=False,
                              epochs=200)

    NN.fit(X,y)
    
    plt.subplot(1,2,2)
    plt.title('Gradient Adam')
    plt.plot(NN.gradient_values)
    plt.xlabel('Epochs',{'fontsize':13})
    """
    
    
    
    