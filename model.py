import numpy as np

class hd_model:

    def __init__(self, input_amount_of_labels, output_data_dim):

        self.model_data = {}
        self.bipolar_data = {}

        self.hd_dim = output_data_dim
        self.amount_of_label = input_amount_of_labels
        
        for i in range(0 , self.amount_of_label):
            self.model_data[str(i)] = np.zeros(self.hd_dim)
            self.bipolar_data[str(i)] = np.zeros(self.hd_dim)

    
    def add_hdv(self, data, label):
        self.model_data[label] = np.add(self.model_data[label], data)


    def bipolar_model_data(self):
        for i in self.bipolar_data.keys():
            self.bipolar_data[i] = np.where( self.model_data[i] > 0, 1, -1)

