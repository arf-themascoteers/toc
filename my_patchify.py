import numpy as np
from PIL import Image


def patchify(image, size, step):
    patches = []
    for i in range(0, image.shape[0], step[0]):
        for j in range(0, image.shape[1], step[1]):
            top_0 = i+size[0]
            if top_0 > image.shape[0]:
                continue
            top_1 = j+size[1]
            if top_1 > image.shape[1]:
                continue
            patch = image[i:top_0, j:top_1,:]
            patches.append(patch)
    return patches


if __name__ == "__main__":
    path = "data/original/1.jpg"



