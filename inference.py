import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image, ImageChops
from tqdm import tqdm

from lib import torchutils
from lib.model_zoo.hrnet import HRNet_Base
from lib.model_zoo.texrnet import TexRNet


class TextOCRDataset:
    
    def __init__(self, data_path='/media/xinyu/SSD1T/data/TextOCR', n=10, slice_idx=0) -> None:
        self.data_path = data_path
        with open(os.path.join(data_path, 'annotations/TextOCR_0.1_train.json'), 'r') as f:
            data = json.load(f)
        self.captions = {}
        # with open(os.path.join(data_path, 'annotations/TextCaps_0.1_train.json'), 'r') as f:
        #     caps = json.load(f)
        #     caps = caps['data']
        #     for cap in caps:
        #         self.captions[cap['image_id']] = cap['caption_str']
                
        self.images = list(data['imgs'].values())
        self.anns = data['anns']
        self.img2anns = data['imgToAnns']
        
        total_images = len(self.images)
        images_per_slice = total_images // n
        start_idx = slice_idx * images_per_slice
        end_idx = start_idx + images_per_slice if slice_idx != n - 1 else total_images
        self.images = self.images[start_idx:end_idx]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, 'images', self.images[idx]['file_name'])
        gt_bboxes = []
        for gt_id in self.img2anns[self.images[idx]['id']]:
            bbox = self.anns[gt_id]['bbox']
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            gt_bboxes.append(bbox)
        
        return img_path, gt_bboxes

class TextRNet_HRNet_Wrapper(object):
    """
    This is the UltraSRWrapper with render-level batchification.
    """
    def __init__(self,
                 device,
                 pth=None,):
        """
        Create uspcale instance
        :param device: device on run the upscale pipeline (if GPU is accessible should be 'cuda')
        :param pth: path to model
        """
        self.model = self.make_model(pth)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

    @staticmethod
    def make_model(pth=None):
        backbone = HRNet_Base(
            oc_n=720, 
            align_corners=True, 
            ignore_label=999, 
            stage1_para={
                'BLOCK'       : 'BOTTLENECK',
                'FUSE_METHOD' : 'SUM',
                'NUM_BLOCKS'  : [4],
                'NUM_BRANCHES': 1,
                'NUM_CHANNELS': [64],
                'NUM_MODULES' : 1 },
            stage2_para={
                'BLOCK'       : 'BASIC',
                'FUSE_METHOD' : 'SUM',
                'NUM_BLOCKS'  : [4, 4],
                'NUM_BRANCHES': 2,
                'NUM_CHANNELS': [48, 96],
                'NUM_MODULES' : 1 },
            stage3_para={
                'BLOCK'       : 'BASIC',
                'FUSE_METHOD' : 'SUM',
                'NUM_BLOCKS'  : [4, 4, 4],
                'NUM_BRANCHES': 3,
                'NUM_CHANNELS': [48, 96, 192],
                'NUM_MODULES' : 4 },
            stage4_para={
                'BLOCK'       : 'BASIC',
                'FUSE_METHOD' : 'SUM',
                'NUM_BLOCKS'  : [4, 4, 4, 4],
                'NUM_BRANCHES': 4,
                'NUM_CHANNELS': [48, 96, 192, 384],
                'NUM_MODULES' : 3 },
            final_conv_kernel = 1,
        )
        
        model = TexRNet(
            bbn_name='hrnet',
            bbn=backbone,
            ic_n=720,
            rfn_c_n=[725, 64, 64],
            sem_n=2,
            conv_type='conv',
            bn_type='bn',
            relu_type='relu',
            align_corners=True,
            ignore_label=None,
            bias_att_type='cossim',
            ineval_output_argmax=False,
        )
        if pth is not None:
            paras = torch.load(pth, map_location=torch.device('cpu'))
            new_paras = model.state_dict()
            new_paras.update(paras)
            model.load_state_dict(new_paras)
        return model

    def process(self, pil_image):
        im = np.array(pil_image.convert("RGB"))
        im = im/255
        im = im - np.array([0.485, 0.456, 0.406])
        im = im / np.array([0.229, 0.224, 0.225])
        im = np.transpose(im, (2, 0, 1))[None]
        im = torch.FloatTensor(im).to(self.device)

        # This step will auto-adjust model if it is torch-DDP
        netm = getattr(self.model, 'module', self.model)
        _, _, oh, ow = im.shape
        ac = True

        prfnc_ms, pcount_ms = {}, {}

        for mstag, mssize in [
                ['0.75x', 385],
                ['1.00x', 513],
                ['1.25x', 641],
                ['1.50x', 769],
                ['1.75x', 897],
                ['2.00x', 1025],
                ['2.25x', 1153],
                ['2.50x', 1281], ]:
            # by area
            ratio = np.sqrt(mssize**2 / (oh*ow))
            th, tw = int(oh*ratio), int(ow*ratio)
            tw = tw//32*32+1
            th = th//32*32+1

            imi = {
                'nofp' : torchutils.interpolate_2d(
                    size=(th, tw), mode='bilinear', 
                    align_corners=ac)(im)}                    
            imi['flip'] = torch.flip(imi['nofp'], dims=[-1])

            for fliptag, imii in imi.items():
                with torch.no_grad():
                    pred = netm(imii)
                    psem = torchutils.interpolate_2d(
                        size=(oh, ow), 
                        mode='bilinear', align_corners=ac)(pred['predsem']) 
                    prfn = torchutils.interpolate_2d(
                        size=(oh, ow), 
                        mode='bilinear', align_corners=ac)(pred['predrfn']) 

                    if fliptag == 'flip':
                        psem = torch.flip(psem, dims=[-1])
                        prfn = torch.flip(prfn, dims=[-1])
                    elif fliptag == 'nofp':
                        pass
                    else:
                        raise ValueError
                
                try:
                    prfnc_ms[mstag]  += prfn
                    pcount_ms[mstag] += 1
                except:
                    prfnc_ms[mstag]  = prfn
                    pcount_ms[mstag] = 1

        pred = sum([pi for pi in prfnc_ms.values()])
        pred /= sum([ni for ni in pcount_ms.values()])
        pred = torch.argmax(psem, dim=1)
        pred = pred[0].cpu().detach().numpy()
        pred = (pred * 255).astype(np.uint8)
        return Image.fromarray(pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/media/xinyu/SSD1T/data/TextOCR')
    parser.add_argument("--output", type=str, default='/media/xinyu/SSD1T/data/TextOCR/masks/')
    parser.add_argument("--slice_idx", type=int, default=0)
    args = parser.parse_args()
    if not osp.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    enl = TextRNet_HRNet_Wrapper(torch.device("cuda:0"), 'pretrained/texrnet_hrnet.pth')

    dataset = TextOCRDataset(data_path=args.data_path, slice_idx=args.slice_idx)
    

    for fin, bboxes in tqdm(dataset):
        save_path = os.path.join(args.output, os.path.basename(fin))
        if osp.exists(save_path):
            continue
        x = Image.open(fin).convert('RGB')
        mask = enl.process(x)
        for bbox in bboxes:
            bbox = [int(coord) for coord in bbox]
            text = x.crop(bbox)
            text_mask = enl.process(text)
            mask_tmp = Image.new("L", x.size)
            mask_tmp.paste(text_mask, (bbox[0], bbox[1]))
            mask = ImageChops.add(mask, mask_tmp)
        mask.save(save_path)
