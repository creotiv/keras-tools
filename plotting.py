import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_colors(n, name='brg'):
    '''Returns list of colors in HEX'''
    cmap = plt.cm.get_cmap(name, n)
    out = []
    for i in range(n):
        out.append((cmap(i)[0],cmap(i)[1],cmap(i)[2]))
    out = [matplotlib.colors.rgb2hex(c) for c in out]
    return out
    
def draw_mask(img,mask,color,alpha=0.4):
    color = np.array(matplotlib.colors.hex2color(color))
    color = color*255
    mask = imresize(mask, img.shape[:2])
    mask = np.clip(mask,0,255).astype(np.uint8)
    img = np.clip(img,0,255).astype(np.uint8)
    color = color[:3]
    img[:,:][mask>0] = ((1-alpha)*img[:,:][mask>0]).astype(np.uint8)
    img[:,:][mask>0] += (color*alpha).astype(np.uint8)
    return img
