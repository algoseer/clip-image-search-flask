import torch
import json
import clip
import numpy as np
from PIL import Image
import sys
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

fnames = []
d = {"hash" : []}
n=0
with torch.no_grad():

    #iterate over images since it won't input batch
    for img in sys.stdin:
        img = img.rstrip()
        fname = '/'.join(img.split('/')[-2:])
        print(fname)
        fnames.append(fname)

        image = preprocess(Image.open(img)).unsqueeze(0).to(device)
        imf = model.encode_image(image)

        dd = {"filename" : fname, "ftr": imf[0].numpy().tolist()}
        d["hash"].append(dd)
        d["all_files"]=fnames

        n+=1
        if n%100 ==0:
            print("Saving...")
            with open("data/img-hash.json","w") as fout:
                json.dump(d,fout)
        
