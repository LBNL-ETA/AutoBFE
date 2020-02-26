import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from modeling.dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, type, writer, dataset, image, target, output, global_step):
        if (type == 'train'):
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Training Image', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Training Predicted label', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Training Groundtruth label', grid_image, global_step)

        else:
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Validation Image', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Validation Predicted label', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Validation Groundtruth label', grid_image, global_step)