import torch
import torch.nn as nn
import numpy as np
import random
#torch.manual_seed(2323)

#TODO: 50 points
class RNNCell(torch.nn.Module):
    """
    A class for defining the cell in a RNN model. Here we will build a simple
    RNN. Other options include GRUs and LSTMs. 

    Attributes:
        Your documentation goes here
    """

    #TODO: 10 points
    def __init__(self, inputDim, hiddenDim):

        super().__init__()
        self.input_size = inputDim
        self.hidden_size = hiddenDim
        
        self.i2h = nn.Linear(inputDim, hiddenDim)
        self.h2o = nn.Linear(hiddenDim, hiddenDim)


    #TODO: 40 points
    def forward(self, x, hidden):
        """
        Defines the forward computation for RNN cell, here a SRNN. 

        Args: 
            x (torch.tensor): Input of shape (batchSize, input size)
            hidden (torch.tensor): Hidden representation of (batchSize,
                        hiddenSize)
        Returns:
            torch.Tensor: New hidden representation
        """
        # your code goes here
        self.output = torch.nn.ReLU()
        return self.output(self.i2h(x) + self.h2o(hidden))


#TODO: 50 points
class RNNModel(torch.nn.Module):
    """
    A class for a RNN model which uses a cell class (which in prinicple 
    could be swapped for other cells; e.g., LSTMs). 

    Attributes:
        Your documentation goes here

    """

    #TODO: 10 points
    def __init__(self, vocabSize:int, inputDim:int, 
                 hiddenDim:int, nLayers:int):

        super().__init__()
        self.nLayers = nLayers
        self.hiddenDim = hiddenDim
        self.vocabSize = vocabSize
        self.inputDim = inputDim

        self.Encoder = nn.Embedding(vocabSize, inputDim)
        self.Decoder = nn.Linear(hiddenDim, vocabSize, bias=False)

        self.hidden_layers_dim = []
        for layer_num in range(nLayers):
            if layer_num == 0:
                layer_output = nn.Linear(inputDim, hiddenDim)

            elif layer_num == range(nLayers):
                layer_output = nn.Linear(hiddenDim, self.Decoder)
            
            else:
                layer_output = nn.Linear(hiddenDim,hiddenDim)
            
            self.hidden_layers_dim.append(layer_output)
        
        #print(hidden_layers)


    


    #TODO: 40 points
    def forward(self, x:torch.Tensor, 
                hidden:torch.Tensor=None) -> (torch.tensor,torch.tensor):
        """
        Defines the forward operation of the model. Recall that
        RNNs update a hidden representation through time. The cell 
        takes care of the nature of this update. Note: The input 
        x is expected to be the output of applying the tokenizer to the 
        input words (ie it will be a tensor of token ids). 

        Args:
            x (torch.Tensor): Input ids of shape (batch size, seq length, input
                                                                        dimension)
            hidden (torch.Tensor | None): Initial hidden representation to
                        use. If None, one should be generated. Shape is 
                            (number of layers, batch size, hidden dimension)

        For example, 

            We might have as input ["the cat is near the window", "the dog is
            happily outside now"]. The input would
            be that string tokenized. Perhaps, as: 
                torch.tensor([[[0], [1], [2], [3], [0], [4]], 
                              [[0], [5], [2], [6], [7], [8]]])
            Notice that this is 3D and has a shape of (2, 6, 1). The 2 
            corresponds to the batch size and tell us that there are two input
            samples ('the cat near the window', and 'the dog is happily outside
            now'). The second dimension registers as 6, meaning we have
            sequences of length 6 (6 words). Finally, the 3rd dimension tells us
            the dimensionality of each element in the sequence. Here it is 1,
            meaning only 1 number is used to express the input. We might imagine 
            a larger number like 40, which would mean that each input element is
            a 40 dimensional object (perhaps a word embedding). 

        Returns:
            torch.Tensor: output logits for each input. shape is 
                    (batch size, sequence length, vocab size)
            torch.Tensor: final hidden representation
        """
        if hidden == None:
            hidden = nn.Linear()

        input_layer = self.forward(self.Encoder, hidden)
        input_relu = nn.ReLU(input_layer)
        
        


        

    def initHidden(self, batchSize:int):
        """
        Creates a hidden representation of all zeros to initalize the model. 

        Args:
            batchSize (int): Size of the batch

        Returns:
            torch.tensor: Tensor of zeros of shape (nLayers, batchSize,
                                                    hiddenDim)
                         That is, each batch has its own hidden representation
                         for each layer. 
        """

        return torch.zeros([self.nLayers, batchSize, self.hiddenDim], dtype=torch.float)
