import torch
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_checkpoint(save_dir, filename):
    checkpoint = torch.load(os.path.join(save_dir, filename), map_location='cpu')
    return checkpoint

def save_checkpoint(net, optimizer, step, filename):
    checkpoint = {
            'state_dict': net.state_dict(),
            'step': step,
            'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, filename)