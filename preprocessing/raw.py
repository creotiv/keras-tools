import numpy as np
import cv2
import os
import sys
from rawkit.raw import Raw
from rawkit.options import interpolation, WhiteBalance
from scipy.misc import imread, imsave


def raw2png(input_dir, output_dir, alpha=0.5):
    not_extensions = ['png','jpg','jpeg','bmp','tif','tiff','gif', 'webp', 'svg']
    files = filter(lambda x: x.split('.')[-1].lower() not in not_extensions, os.listdir(input_dir))
    
    for f in files:
        print f
        if not os.path.isfile(os.path.join(input_dir, f)):
            continue
        with Raw(filename=os.path.join(input_dir, f)) as raw:
            raw.options.interpolation = interpolation.ahd
            raw.options.white_balance = WhiteBalance(camera=False, auto=True)
            raw.options.auto_brightness = True
            raw.options.green_matching = True
            raw.options.rgbg_interpolation = False
            raw.options.auto_brightness_threshold = 0.01
            raw.options.bps = 8
            raw.save('/tmp/buffer.tiff')
            img = imread('/tmp/buffer.tiff', mode="RGB")
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            new = img.copy().astype(np.uint8)
            new[:,:,0] = clahe.apply(img[:,:,0])
            new[:,:,1] = clahe.apply(img[:,:,1])
            new[:,:,2] = clahe.apply(img[:,:,2])
            
            new = img * alpha + (1-alpha) * new
            imsave(os.path.join(output_dir, '.'.join(f.split('.')[:-1])+'.png'), (new).astype(np.uint8))
            
if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
    
