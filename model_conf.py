from layers import * 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda


gray = lambda tensor : tensor.sum(0, keepdim=True) / 3

def svhn_targets(tensor):
    return (tensor - 1).long().squeeze()

def conf(args):
    # con1, primarycaps_ 16 out channels, out_diminsions
    if args.data_dir == '':
        args.data_dir = './data'
    label_fn = None
    train_args = {'download':True}
    test_args = {}
    misc_args = {'kernel':9, 'fc_first_layer':512}

    if args.dataset == 'mnist':
        data_param = {'num_classes': 10, 'data_shape': [1, 1, 28, 28],
                      'output_size': 28 * 28, 'image_channels': 1}
        net_params = {'num_layer1_convs':256,
                      'primary_out_channels':32,
                      'primary_out_dim':8,
                      'final_caps_dim':16}
        train_data = datasets.mnist.MNIST
        train_kys = {'train': True, 'transform': Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])}
        test_kyes = {'train': False, 'transform': Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])}

    elif args.dataset == 'svhn':
        data_param = {'num_classes': 10, 'data_shape': [1, 1, 32, 32],
                      'output_size': 32 * 32, 'image_channels': 1}
        net_params = {'num_layer1_convs':64,
                      'primary_out_channels':16,
                      'primary_out_dim':6,
                      'final_caps_dim':8}
        train_data = datasets.SVHN
        train_kys = {'transform': Compose([ToTensor(), Lambda(gray), Normalize((0.1307,), (0.3081,))])}
        test_kyes = {'transform': Compose([ToTensor(), Lambda(gray), Normalize((0.1307,), (0.3081,))])}
        label_fn = svhn_targets

    elif args.dataset == 'fashion':
        data_param = {'num_classes': 10, 'data_shape': [1, 1, 28, 28],
                      'output_size': 28 * 28, 'image_channels': 1}
        net_params = {'num_layer1_convs':256,
                      'primary_out_channels':32,
                      'primary_out_dim':8,
                      'final_caps_dim':16}
        train_data = datasets.mnist.FashionMNIST
        train_kys = {'train': True, 'transform': Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])}
        test_kyes = {'train': False, 'transform': Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])}

    elif args.dataset == 'cifar10':
        data_param = {'num_classes': 10, 'data_shape': [1, 1, 28, 28],
                      'output_size': 28 * 28, 'image_channels': 1}
        net_params = {'num_layer1_convs':256,
                      'primary_out_channels':32,
                      'primary_out_dim':8,
                      'final_caps_dim':16}
        train_data = datasets.cifar.CIFAR10
        train_kys = {'train': True, 'transform': ToTensor()}
        test_kyes = {'train': False, 'transform': ToTensor()}

    else:
        data_param, net_params, train_data, train_kys, test_kyes = {}, {}, {}, {}, {}

    train_ = {**train_args, **train_kys}
    test__ = {**test_args, **test_kyes}
    train_loader = DataLoader(train_data(args.data_dir, **train_),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(train_data(args.data_dir, **test__),
                             batch_size=args.batch_size, shuffle=True)

    return {**data_param, **net_params, **misc_args}, train_loader, test_loader, label_fn