import os
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import itertools

from lib.core.config import cfg


def test_transform():
    transform_list = [transforms.ToTensor()]
    return transforms.Compose(transform_list)

class CustomDataset(data.Dataset):
    def __init__(self, content_folder, style_folder):
        self.content_images = [os.path.join(content_folder, f) for f in os.listdir(content_folder) if
                               f.endswith(('.jpg', '.png', '.jpeg'))]
        self.style_images = [os.path.join(style_folder, f) for f in os.listdir(style_folder) if
                             f.endswith(('.jpg', '.png', '.jpeg'))]
        self.combinations = list(itertools.product(self.content_images, self.style_images))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        content_image_path, style_image_path = self.combinations[idx]
        content_image = Image.open(content_image_path)
        style_image = Image.open(style_image_path)

        return (test_transform()(content_image).to(self.device),
                test_transform()(style_image).to(self.device),
                content_image_path,
                style_image_path)


import torch.nn as nn
from lib.models.base_models import Encoder, DecoderAWSC1, DecoderAWSC2
from lib.core.mast import MAST


from lib.core.config import get_cfg

class NSTModel(nn.Module):
    def __init__(self, encoder_path, decoder_path, cfg_path):
        super(NSTModel, self).__init__()

        self.cfg = get_cfg(cfg_path)

        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(encoder_path))

        layers = self.cfg.TEST.PHOTOREALISTIC.LAYERS.split(',')
        if self.cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_TYPE == 'AWSC2':
            self.decoder = DecoderAWSC2(layers, self.cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_WEIGHT)
        if self.cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_TYPE == 'AWSC1':
            self.decoder = DecoderAWSC1(layers, self.cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_WEIGHT)

        self.decoder.load_state_dict(torch.load(decoder_path))

        self.mast = MAST(self.cfg)

    def forward(self, content, style, content_seg_path=None, style_seg_path=None, seg_type='dpst'):
        cf = self.encoder(content)
        sf = self.encoder(style)

        style_weight = self.cfg.TEST.PHOTOREALISTIC.STYLE_WEIGHT
        csf = {}
        for layer in self.cfg.TEST.PHOTOREALISTIC.LAYERS.split(','):
            temp = self.mast.transform(cf[layer], sf[layer], content_seg_path, style_seg_path, seg_type)
            temp = style_weight * temp + (1 - style_weight) * cf[layer]
            csf[layer] = temp

        out_tensor = self.decoder(csf)
        return out_tensor
