import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageMath
from config import cfg
import torch
# ===============================img tranforms============================

def add_margin(pil_img, new_width, new_height, left, top, color):
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def resize_target(pil_img, tw, th):
    w, h = pil_img.size
    ratio_rounded = (h * w ) / (int(tw) * int(th))
    mask = np.array(pil_img.resize((int(tw) ,int(th)), Image.BOX)) * ratio_rounded
    return Image.fromarray(mask, 'F')

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            xmin = w - bbx[:,3]
            xmax = w - bbx[:,1]
            bbx[:,1] = xmin
            bbx[:,3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img, mask
        return img, mask, bbx

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.mean = (115, 114, 110)#(tuple([int(i*255) for i in cfg.MEAN_STD[0]]))

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw  = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            #return img.resize((tw, th), Image.BILINEAR), resize(mask, tw, th)
            return add_margin(img, tw,th, 0, 0, color=tuple(self.mean)), add_margin(mask, tw,th, 0, 0, color=0)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))



class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.NEAREST), resize_target(mask, self.size[1],  self.size[0])

class ScaleDown(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, mask):
        return  resize_target(mask,self.size[1]/cfg.TRAIN.DOWNRATE, self.size[0]/cfg.TRAIN.DOWNRATE)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)           
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)

class RandomDownOverSampling(object):
    # Downsampling then upsampling the image to the original size. 
    # Simulate bad encoding, to make the network more resilient
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img, mask):
        if self.factor < 0 :# or random.random() > 0.5:
            return img, mask
        
        if img.size != mask.size:
            print(img.size)
            print(mask.size)           
        assert img.size == mask.size
        w, h = img.size
        factor = random.choice(range(self.factor)) + 1
        return img.resize((int(w / factor) ,int(h / factor)), Image.BILINEAR).resize((w, h), Image.NEAREST), mask

class RandomDownSampling(object):
    # Downsampling then upsampling the image to the original size. 
    # Simulate bad encoding, to make the network more resilient
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, img, mask):
        if self.factor < 0 or random.random() > 0.5:
            return img, mask
        
        if img.size != mask.size:
            print(img.size)
            print(mask.size)           
        assert img.size == mask.size
        w, h = img.size
        return img.resize((int(w / self.factor) ,int(h / self.factor)), Image.BILINEAR) ,resize_target(mask, int(w / self.factor) ,int(h / self.factor))
       
# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor==1:
            return img
        tmp = np.array(img.resize((w//self.factor, h//self.factor), Image.BICUBIC))*self.factor*self.factor
        img = Image.fromarray(tmp)
        return img