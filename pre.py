import my_patchify
import os
from PIL import Image
import numpy as np
import pandas as pd

df = pd.read_csv("data/carbon.csv")

img_dir = "data/original"
gen_dir = "data/gen"

records = []

index = 1

for img in os.listdir(img_dir):
    filtered = df.loc[df['image'] == img]
    oc = filtered.iloc[0, 0]

    img_path = os.path.join(img_dir, img)
    img = Image.open(img_path)
    image = np.asarray(img)
    image = image[800:-800,800:-800,:]
    patches = my_patchify.patchify(image, (128,128), (128,128))
    for i in patches:
        name = f"{index}.jpg"
        index = index + 1
        save_path = os.path.join(gen_dir, name)
        im = Image.fromarray(i)
        im.save(save_path)
        record = [name, oc]
        records.append(record)

out_df = pd.DataFrame(data=records, columns=["oc","name"])
out_df.to_csv("data/csv.csv", index=False)
print(f"All done. Total image: {index-1}")


