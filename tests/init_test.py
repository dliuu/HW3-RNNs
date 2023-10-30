import os
import sys
import torch
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model as M

def testCellInit():

    cell = M.RNNCell(4, 10)

    numLinear = 0
    act = 0
    for module in cell.modules():
        if isinstance(module, torch.nn.Linear):
            numLinear += 1
        else:
            act += 1

    assert numLinear == 2, "You need to create two linear layers for the cell "\
                            "(one for hidden one for input)"
    assert act, "You need an activation function for the cell"

def testModelInit():

    model = M.RNNModel(10, 4, 20, 2)
    
    foundLinear = 0
    foundEmbedding = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.in_features == 20, "You aren't using the hiddenDim argument correctly in creating the decoder."
            assert module.out_features == 10, "You aren't using the vocabSize argument correctly in creating the decoder."
            foundLinear += 1
        elif isinstance(module, torch.nn.Embedding):
            shape = list(module.weight.shape)
            assert shape[0] == 10, "You aren't using the hiddenDim argument correctly in creating the embedding."
            assert shape[1] == 4, "You aren't using the vocabSize argument correctly in creating the embedding."
            foundEmbedding +=1

    assert foundLinear, 'You failed to create a linear layer for RNNModel'
    assert foundEmbedding, 'You failed to create an embedding layer for RNNModel'
