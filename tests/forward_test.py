import os
import sys
import torch
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M

def testCellForward():

    model = M.RNNCell(4, 3)

    x = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=float)

    hidden = torch.tensor([1, 2, 3], dtype=float)

    inputU = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                          dtype=float)
    hiddenW = torch.tensor([[11, 12, 13], [15, 16, 17], 
                            [19, 20,21]],
                          dtype=float)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if module.in_features == 4:
                module.weight = torch.nn.Parameter(inputU)
                module.bias = torch.nn.Parameter(torch.tensor(
                                    [1, 2, 3], dtype=float))
            else:
                module.weight = torch.nn.Parameter(hiddenW)
                module.bias = torch.nn.Parameter(torch.tensor(
                                    [1, 2, 3], dtype=float))

    try:
        newHidden = model(x, hidden)
    except NotImplementedError: 
        assert 0, "You have to implement the forward method for RNNCell."
    
    assert newHidden.equal(torch.tensor([[76, 102, 128], [86, 128, 170]],dtype=float)), "Forward for RNNCell is incorrect."

def testModelForward():

    model = M.RNNModel(10, 4, 3, 2)

    #Set up model
    #First the cells

    cell1 = M.RNNCell(4, 3)
    cell2 = M.RNNCell(3, 3)
    inputU = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                          dtype=torch.float)
    hiddenW = torch.tensor([[11, 12, 13], [15, 16, 17], 
                            [19, 20,21]],
                          dtype=torch.float)

    for module in cell1.modules():
        if isinstance(module, torch.nn.Linear):
            if module.in_features == 4:
                module.weight = torch.nn.Parameter(inputU)
                module.bias = torch.nn.Parameter(torch.tensor(
                                    [1, 2, 3], dtype=torch.float))
            else:
                module.weight = torch.nn.Parameter(hiddenW)
                module.bias = torch.nn.Parameter(torch.tensor(
                                    [1, 2, 3], dtype=torch.float))

    for module in cell2.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(hiddenW)
            module.bias = torch.nn.Parameter(torch.tensor(
                                [1, 2, 3], dtype=torch.float))

    model.cells = [cell1, cell2]

    #Now set up encoder/decoder
    embedding = torch.tensor([
        [0,1,2,3],
        [4,5,6,7],
        [8,9,0,1],
        [2,3,4,5],
        [6,7,8,9],
        [0,1,2,3],
        [4,5,6,7],
        [8,9,0,1],
        [2,3,4,5],
        [6,7,8,9]], dtype=torch.float)

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight = torch.nn.Parameter(embedding)
        elif isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(embedding[:,:-1])

    x = torch.tensor([[[0], [1], [2], [3], [0], [4]], 
                              [[0], [5], [2], [6], [7], [8]]], dtype=int)

    try:
        output, newHidden = model(x)
    except NotImplementedError: 
        assert 0, "You have to implement the forward method for RNNModel."
    output = torch.sum(torch.sum(output, dim=-1), dim=-1)

    val0 = int(float(output[0])/1_000_000_000_000)
    val1 = int(float(output[1])/1_000_000_000_000)

    assert val0 == 529 and val1 == 510, "Your model forward is incorrect for " \
                                        "RNNModel"
