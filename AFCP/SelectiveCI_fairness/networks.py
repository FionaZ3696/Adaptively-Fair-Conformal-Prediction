import torch 
import torch.nn as nn
import numpy as np


class ClassNNet(nn.Module):
    def __init__(self, num_features, num_classes, device, use_dropout=False):
        super(ClassNNet, self).__init__()

        self.device = device

        self.use_dropout = use_dropout

        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, num_classes)

        self.z_dim = 256 + 128

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, x, extract_features=False):
        x = self.layer_1(x)
        x = self.relu(x)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm1(x)

        z2 = self.layer_2(x)
        x = self.relu(z2)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm2(x)

        z3 = self.layer_3(x)
        x = self.relu(z3)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm3(x)
        x = self.layer_4(x)
        x = self.relu(x)

        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm4(x)

        x = self.layer_5(x)

        if extract_features:
          return x, torch.cat([z2,z3],1)
        else:
          return x

    def predict_prob(self, inputs):
        """
        Predict probabilities given any input data
        """
        self.eval()
        get_prob = nn.Softmax(dim = 1)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self(inputs)
            prob = get_prob(outputs).cpu().numpy()
        return prob


class BinaryClassification(nn.Module):
    def __init__(self, device, **kwargs):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.device = device
        self.layer_1 = nn.Linear(kwargs['input_shape'], 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

    def get_anomaly_scores(self, inputs):
        """
        Compute the anomaly scores for a given set of inputs as the rescontruction error
        """
        self.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self(inputs)
            prob_1 = torch.sigmoid(outputs)
            scores = prob_1.cpu().numpy()
            scores = scores.reshape([1,len(scores)])[0]

        return list(scores)



class AutoEncoder(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def get_anomaly_scores(self, inputs):
        """
        Compute the anomaly scores for a given set of inputs as the rescontruction error
        """
        self.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self(inputs)
            Loss = torch.nn.MSELoss(reduction='none')
            scores = Loss(outputs, inputs)

        scores = np.mean(scores.cpu().numpy(), axis = 1)
        return list(scores)

