import torch
from PIL import Image
import numpy as np

def save_img(array, filename):
    array = array.view(64, 64).cpu().data.numpy()
    img = Image.fromarray(np.rint(array*255).astype(np.uint8))
    img.save(filename)
