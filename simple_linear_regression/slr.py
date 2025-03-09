import numpy as np

class LinearRegression:

    def __init__(self):
        self.parameters = {}

    def predicting(self,train_input):
        m = self.parameters['m']
        c = self.parameters['c']

        return np.multiply(m,train_input) + c

    def error_calc(self,prediction,train_output):
        return np.mean((prediction - train_output)**2)
    
    def gradient(self,prediction,train_output,train_input):
        derivatives = {}
        diff = prediction - train_output
        derivatives['dm'] = 2 * np.mean(np.multiply(diff,train_input))
        derivatives['dc'] = 2 * np.mean(diff)
        return derivatives
    
    def update(self,learning_rate,derivatives):
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc'] 

    def train(self,train_input,train_output,learning_rate,iters):
        self.parameters['m'] = np.random.uniform(0,1) * -1
        self.parameters['c'] = np.random.uniform(0,1) * -1

        for i in range(iters):
            prediction = self.predicting(train_input)
            cost = self.error_calc(prediction,train_output)
            derivatives = self.gradient(prediction,train_output,train_input)
            self.update(learning_rate,derivatives)