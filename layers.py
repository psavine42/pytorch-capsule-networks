import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(tensor, dim=1):
    # [batch_size, num_caps, vec_len, 1]
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


def routing_softmax(tensors, dim=2):
    """
        The coupling coefficients between capsule i and all the capsules in the layer above
        sum to 1 and are determined by a “routing softmax”
        whose initial logits b_i|j are the
        log prior probabilities that capsule i should be coupled to capsule j.
        :input: tensor of size [num_batches, ]
        :return:
    """
    transposed_input = tensors.transpose(dim, len(tensors.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(tensors.size()) - 1)


class RoutingLayer(nn.Module):
    def __init__(self,
                 num_capsules=10,
                 num_iters=3,
                 num_routes=32 * 6 * 6,
                 in_channels=8,
                 out_channels=16):
        super(RoutingLayer, self).__init__()
        self.num_iters = num_iters
        print(num_capsules, num_routes, in_channels, out_channels)
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_routes, in_channels, out_channels))

    def forward(self, x):
        """

            :param x: tensor of size [num_capsules, weight_matrix_size,     ]
                      paper  of size [8,            32 * 6 * 6,         8]
            :return: outputs of size [num_batches, num_classes, out_channels]
            """
        # u_hat_j|i = W_ij @ u_i (linear layer w/o a bias)
        prediction_vectors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

        # for all capsule i in layer l and capsule j in layer (l + 1): bij ← 0.
        logits = Variable(torch.zeros(list(prediction_vectors.size()))).cuda()

        for i in range(self.num_iters):
            # for all capsule i in layer l: ci ← softmax(bi)v
            coupling_coefficent = routing_softmax(logits, dim=2)
            outputs = squash((coupling_coefficent * prediction_vectors).sum(dim=2, keepdim=True))

            if i != self.num_iters - 1:
                delta_logits = (prediction_vectors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits
        return outputs


class Capsules(nn.Module):
    """
        The second layer (PrimaryCapsules) is a convolutional capsule layer with 32 channels
        of convolutional 8D capsules
        each primary capsule contains 8 convolutional units  with a 9 × 9 kernel and a stride of 2).
        """
    def __init__(self,
                 num_capsules=8,
                 in_channels=256,
                 out_channels=32,
                 kernel_size=9,
                 stride=2):
        super(Capsules, self).__init__()
        self.out_chans = out_channels
        self.in_chans = in_channels
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels, out_channels,
                                                 kernel_size=kernel_size, stride=stride)
                                       for _ in range(num_capsules)])

    def forward(self, x):
        """
            1st layer converts pixel intensities to detectors
            that are used as inputs to the primary caps
            Each kernel (conv2d) returns 32 channels
            :param x: input size [num_batches, in_channels, in_size, in_size]
                    paper [b, 256, 20, 20]
            :return: [num_batches, num_classes, num_]
                    paper [b, 6, 6, 32, 8]
            """
        x = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
        x = torch.cat(x, -1)
        return squash(x, 2)


class CapsuleLayer(nn.Module):
    def __init__(self, 
                 num_classes=10,
                 in_channels=256):
        super(CapsuleLayer, self).__init__()
        self.num_classes = num_classes
        self.primary_caps = Capsules(in_channels=self.conv1_units)
        self.digit_caps = RoutingLayer()

    def forward(self, inputs):
        """ """
        x = self.primary_caps(inputs)
        outputs = self.digit_caps(x).squeeze().permute(1, 0, 2)
        return outputs


class FCDecoder(nn.Module):
    def __init__(self, input_size=160, first_layer=512, output_size=784):
        super(FCDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, first_layer)
        self.fc2 = nn.Linear(first_layer, first_layer * 2)
        self.fc3 = nn.Linear(first_layer * 2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return torch.sigmoid(self.fc3(x))