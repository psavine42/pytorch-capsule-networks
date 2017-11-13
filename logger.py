from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.autograd import Variable

dolog = True
writer = SummaryWriter()
global_step = 0
log_step = 100

def to_log(tnsr):
    clone = tnsr.clone()
    if type(clone) == Variable:
        clone = clone.data
    sizes = list(clone.size())

    if len(sizes) > 1 and not all((s == 1 for s in sizes)):
        return clone.cpu().squeeze().numpy()
    else:
        return clone.cpu().numpy()


def log_if(name, data, f=None):
    if global_step % log_step == 0:
        if f is None:
            writer.add_histogram(name, data, global_step, bins='sturges')
        else:
            f(name, data, global_step)


def log_images(input_images, target_classes, output_images, output_classes):
    print(output_images.size())
    inputs = vutils.make_grid(input_images, normalize=False, scale_each=True)
    outputs = vutils.make_grid(output_images, normalize=False, scale_each=True)
    writer.add_image('test.inputs', inputs, global_step)
    writer.add_image('test.outputs', outputs, global_step)


def log_model(model):
    for name, param in model.named_parameters():
        writer.add_histogram(name, to_log(param), global_step, bins='sturges')


def log_loss(stage, loss, accy, guard=True):
    if global_step % log_step == 0 or guard is False:
        writer.add_scalar(stage + '.loss', loss.clone().cpu().data[0], global_step)
        writer.add_scalar(stage + '.accy', accy, global_step)

