from flask import Flask, send_file, request
import clip
import json
import numpy as np
import logging
import torch

app = Flask(__name__)

model = None
img_feats = [] 
fnames = []

def load_clip_model():
    global model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

def load_image_db(path):
    global img_feats, fnames
    with open(path) as fin: 
        d = json.load(fin)
    img_feats = np.array([a["ftr"] for a in d["hash"]])
    fnames = [a["filename"] for a in d["hash"]]

@app.route("/search", methods=['GET','POST'])
def query_images():
    global model, device
    query = request.args.get('text')
    logging.info("Query:" + query)
    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text).numpy()[0]
        logits_per_text = torch.FloatTensor(np.dot(img_feats, text_features))
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        img_ind = probs.argsort()
        ret = [(probs[i], fnames[i])  for i in img_ind[-3:]]
        top_img = ret[-1][1]

        #Send the top image
        return send_file("static/images/"+top_img)


if __name__ == "__main__":
    load_clip_model()
    logging.info("Models loaded..")

    load_image_db("data/img-hash.json")
    logging.info("Database loaded..")

    app.run(host="localhost", port=8000, debug=True)
