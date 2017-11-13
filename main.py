import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets.mnist import MNIST
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import torchnet as tnt
from layers import *
from model_conf import conf
import argparse
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import logger as sl

batch_size = 16
test_batch_size = 16
epochs = 10
lr = 0.00001
momentum = 0.5
global_step = 0
seed = 1
log_interval = 10
torch.manual_seed(seed)

def accuracy(classes, targets):
    _, max_indices = classes.max(dim=1)
    return len(torch.nonzero(max_indices.data.cpu() - targets).squeeze().tolist())

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.m_plus = 0.9
        self.m_minus = 0.1
        self._lambda = 0.5
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, input_images, target_classes, output_images, output_classes):
        """
            L_k = T_k max(0, m+ − ||vk||) ^ 2 + λ (1 − Tk) max(0, ||vk|| − m−) ^ 2

            :param inputs: input images
                            size: [batch, channels, width, height]
            :param outputs: reconstructed image from capsule tensors
                            size: [batch, channels, width, height]
            """

        m_loss1 = F.relu(self.m_plus - output_classes, inplace=True) ** 2
        m_loss2 = F.relu(output_classes - self.m_minus, inplace=True) ** 2

        margin_loss = target_classes * m_loss1 + self._lambda * (1. - target_classes) * m_loss2

        reconstruction_loss = self.criterion(output_images, input_images)

        return (margin_loss.sum() + 0.0005 * reconstruction_loss) / input_images.size(0)


class CapsuleNet(nn.Module):
    """
        """
    def __init__(self, num_classes=10, fc_first_layer=512, output_size=784,
                 num_layer1_convs=256, kernel=9, primary_out_channels=32,
                 primary_out_dim=8, data_shape=[2, 3, 5],
                 final_caps_dim=16, image_channels=1):
        super(CapsuleNet, self).__init__()
        # ataset
        self.num_classes = num_classes
        self.output_size = output_size
        # capsule neyt definition
        self.conv1_out_chans = num_layer1_convs
        self.primary_out_channels = primary_out_channels
        self.primary_out_dim = primary_out_dim
        self.final_caps_dim = final_caps_dim
        self.num_iters = 3
        # misc
        self.kernel_size = kernel

        self.conv1 = nn.Conv2d(image_channels,
                               out_channels=self.conv1_out_chans,
                               kernel_size=self.kernel_size, stride=1, padding=0)

        self.primary_caps = Capsules(num_capsules=self.primary_out_dim,
                                     in_channels=self.conv1_out_chans,
                                     out_channels=self.primary_out_channels,
                                     kernel_size=self.kernel_size)
        # dynamically initialize wieght matrix
        self.weight_matrix_size = \
            self.primary_caps(self.conv1(Variable(torch.randn(data_shape)))).size(1)

        self.digit_caps = RoutingLayer(num_capsules=self.num_classes,
                                       num_routes=self.weight_matrix_size,
                                       in_channels=self.primary_out_dim,
                                       out_channels=self.final_caps_dim,
                                       num_iters=self.num_iters)

        # The final Layer (DigitCaps) has one 16D capsule per digit class
        # and each of these capsules receives input from all the capsules in the layer below.
        self.decoder = FCDecoder(input_size=self.final_caps_dim * num_classes,
                                 first_layer=fc_first_layer,
                                 output_size=output_size)

    def top_capsules(self, x):
        classes = x.norm(dim=-1)
        classes = F.softmax(classes)

    def forward(self, inputs):
        """
            :inputs:image tensors size [num_batches, num_chan, h, w]
            """
        # [batches, img_chans, img_h, img_w] -> [batches, feature_maps, feature_h, feature_w]
        x = self.conv1(inputs)
        x = F.relu(x, inplace=True)

        # [batches, feature_maps, feature_h, feature_w] -> [batches, , out_vec_size]
        x = self.primary_caps(x)
        x = self.digit_caps(x).squeeze().permute(1, 0, 2)

        classes = x.norm(dim=-1)
        classes = F.softmax(classes)
        # maximally activated capsule
        _, max_indices = classes.max(dim=1)
        y = Variable(torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_indices.data)
        max_capsule = (x * y[:, :, None]).view(x.size(0), -1)

        outputs = self.decoder(max_capsule)
        return classes, outputs



def test(model, test_loader, label_fn, Loss_Module, num_classes):
    print(sl.global_step)
    inputs, target = test_loader.__iter__().__next__()
    inputs = Variable(inputs).cuda()
    classes, outputs = model(inputs)

    target = target if label_fn is None else label_fn(target)
    labels_ix = Variable(torch.eye(num_classes).index_select(dim=0, index=target)).cuda()

    avg_loss = Loss_Module(inputs, labels_ix, outputs, classes)
    avg_acc = accuracy(classes, target)

    sl.log_images(inputs.data, target, outputs.data.view(inputs.size()), classes)
    sl.log_loss('test', avg_loss, 1 - avg_acc/inputs.size(0), guard=False)


def train(args):
    model_params, train_loader, test_loader, label_fn = conf(args)
    print(model_params)
    model = CapsuleNet(**model_params).cuda()
    optimizer = Adam(model.parameters()) # , lr=lr
    model.train()
    print(model)

    Loss_Module = CapsuleLoss().cuda()
    num_classes = model_params['num_classes']

    for epoch in range(args.epochs):

        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data).cuda()
            optimizer.zero_grad()
            #
            classes, outputs = model(data)
            #
            target = target if label_fn is None else label_fn(target)
            labels_ix = Variable(torch.eye(num_classes).index_select(dim=0, index=target)).cuda()
            loss = Loss_Module(data, labels_ix, outputs, classes)
            loss.backward()
            optimizer.step()
            #
            acc = accuracy(classes, target)
            sl.log_loss('train', loss, 1 - acc / args.batch_size)
            if sl.global_step % 1000 == 0:
                test(model, test_loader, label_fn, Loss_Module, num_classes)
            sl.global_step += 1

        sl.log_model(model)
        torch.save(model, './models/model_e{}_d_{}.pkl'.format(epoch, args.dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--act', nargs='?', type=str, default='train', help='[]')
    parser.add_argument('--dataset', nargs='?', type=str, default='mnist', help='')
    parser.add_argument('--data_dir', nargs='?', type=str, default='', help='')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16, help='')
    parser.add_argument('--load', nargs='?', type=str, default='', help='load model and state')
    parser.add_argument('--log_step', nargs='?', type=int, default=100, help='load model and state')
    parser.add_argument('--epochs', nargs='?', type=int, default=10, help='')
    args = parser.parse_args()
    print(args)
    sl.log_step += args.log_step
    if args.act == 'test':
        pass
    else:
        train(args)

# python capsnet.py --dataset mnist --epochs 1
# python capsnet.py --dataset svhn --epochs 1 --data_dir svhn --batch_size 12
