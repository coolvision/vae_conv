'''
Generates a summary of a model's layers and dimensionality
'''

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd

class Summary(object):

    def __init__(self, model, input_size=(1,1,256,256)):
        '''
        Generates summaries of model layers and dimensions.
        '''
        self.model = model
        self.input_size = input_size

        self.summarize()
        print(self.summary)

    def get_variable_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        mods = list(self.model.modules())
        in_sizes = []
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            in_sizes.append(np.array(input_.size()))
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.in_sizes = in_sizes
        self.out_sizes = out_sizes
        return

    def get_layer_names(self):
        '''Collect Layer Names'''
        mods = list(self.model.named_modules())
        names = []
        layers = []
        for m in mods[1:]:
            names += [m[0]]
            layers += [str(m[1].__class__)]

        layer_types = [x.split('.')[-1][:-2] for x in layers]

        self.layer_names = names
        self.layer_types = layer_types
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []

        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            modsz = []
            for j in range(len(p)):
                modsz.append(np.array(p[j].size()))
            sizes.append(modsz)

        self.param_sizes = sizes
        return

    def get_parameter_nums(self):
        '''Get number of parameters in each layer'''
        param_nums = []
        for mod in self.param_sizes:
            all_params = 0
            for p in mod:
                all_params += np.prod(p)
            param_nums.append(all_params)
        self.param_nums = param_nums
        return

    def summary(self):
        '''
        Makes a summary listing with:

        Layer Name, Layer Type, Input Size, Output Size, Number of Parameters
        '''

        df = pd.DataFrame( np.zeros( (len(self.layer_names), 5) ) )
        df.columns = ['Name', 'Type', 'InSz', 'OutSz', 'Params']

        df['Name'] = self.layer_names
        df['Type'] = self.layer_types
        df['InSz'] = self.in_sizes
        df['OutSz'] = self.out_sizes
        df['Params'] = self.param_nums

        self.summary = df
        return

    def summarize(self):
        self.get_variable_sizes()
        self.get_layer_names()
        self.get_parameter_sizes()
        self.get_parameter_nums()
        self.summary()

        return
