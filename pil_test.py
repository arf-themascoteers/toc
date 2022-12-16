from PIL import Image
import numpy as np

im = Image.open("data/original/1.jpg")
ar = np.asarray(im)
print(ar.shape)