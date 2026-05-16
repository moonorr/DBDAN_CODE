

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
import utils
import KAN




class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    def forward(self, x):
        return x.view_as(x)
    def backward(self, grad_output):
        return (grad_output*-self.lambd)

def pretrained_CFE():
    model  = KAN.KAN([310, 128, 64])

    return model

def grad_reverse(x,lambd=1.0):
    return GradReverse(lambd)(x)

class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64,32),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x

class DSC(nn.Module):
    def __init__(self):
        super(DSC, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(32, 3),

        )

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.module(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.restored = False
        self.discriminator = nn.Sequential(

            nn.Linear(32, 3+ 1)
        )

    def forward(self, x):
        x = F.dropout(F.relu(x), training=self.training)
        out = self.discriminator(x)
        return out


class MSMDAERNet(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4):
        super(MSMDAERNet, self).__init__()
        self.sharedNet = pretrained_CFE()
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')



    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0,data=0):



        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            DSFE_name = 'self.DSFE' + str(mark)


            data_tgt_DSFE = eval(DSFE_name)(data_tgt_CFE)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)


            return  data_src_DSFE,data_tgt_DSFE

        else:
            data_CFE = self.sharedNet(data)
            feature_DSFE = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                feature_DSFE.append(feature_DSFE_i)

            return feature_DSFE

class classifier(nn.Module):
    def __init__(self, number_of_source=15):
        super(classifier, self).__init__()

        for i in range(number_of_source):
            exec('self.DSC' + str(i) + '=DSC()')



    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0,feature_out=0):


        if self.training == True:
            DSC_name = 'self.DSC' + str(mark)
            data_tgt_out = eval(DSC_name)(data_tgt)
            data_src_out = eval(DSC_name)(data_src)

            return data_src_out, data_tgt_out

        else:

            pred = []

            for i in range(number_of_source):
                DSC_name = 'self.DSC' + str(i)
                pred.append(eval(DSC_name)(feature_out[i]))

            return pred


class Discriminator1(nn.Module):
    def __init__(self, number_of_source=15):
        super(Discriminator1, self).__init__()

        for i in range(number_of_source):
            exec('self.Discriminator' + str(i) + '=Discriminator()')


    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0,feature_out=0):
        DIS_name = 'self.Discriminator' + str(mark)
        data_tgt_out = eval(DIS_name)(data_tgt)
        data_src_out = eval(DIS_name)(data_src)

        return data_src_out, data_tgt_out



class Discriminator2(nn.Module):
    def __init__(self, number_of_source=15):
        super(Discriminator2, self).__init__()

        for i in range(number_of_source):
            exec('self.Discriminator' + str(i) + '=Discriminator()')


    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0,feature_out=0):
        DIS_name = 'self.Discriminator' + str(mark)
        data_tgt_out = eval(DIS_name)(data_tgt)
        data_src_out = eval(DIS_name)(data_src)

        return data_src_out, data_tgt_out





