from flask import Flask, send_file, request
import clip
import operator
import json
import os
import numpy as np
import logging
import torch

app = Flask(__name__)

model = None
rootdir = "static/images/"
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



'''
Serve static files given relative paths
'''

@app.route("/file/<path:path>")
def serve_image(path):
    
    if os.path.exists(rootdir+ path):
        return send_file(rootdir+path)
    else:
        return "File not found", 404

@app.route("/update", methods=['POST'])
def update_json_payload():
    '''
    Example payload:
    {
            "StoryWritingJobStatus": “a string”, # IN_PROGRESS | FAILED | COMPLETED
            "Data": [
            {
                “id”: 0,
                “pageType”: “front”,    # front/body/end
                “Text”: “input string”,
            "MediaImgUris": ["s3://DOC-EXAMPLE-BUCKET/your-image-file"],
            }...
            ...
            ...
                    ]    
    }
    '''

    payload = request.json
    #Upload image uris with the query
    for page in payload["Data"]:
        phrases = page["phrases"] 
        kw = page["keywords_all"] 

        top_imgs=[]
        if phrases:
            for lin in phrases:
                img = get_topK_images(lin, K=1)[0]
                top_imgs.append(img)

        #Generate images for all keywords and keep the one with highest score
        imgs =[get_topK_images(k)[0] for k in kw]
        top_imgs.extend(imgs)

        #Find the highest scoring phrase or keyword
        ret_imgs = max(top_imgs, key = operator.itemgetter(0))
        ret_imgs = [ret_imgs[1]]

        page["MediaImgUris"] =  ret_imgs

    return payload


def get_topK_images(query, K=1):
    global model, device

    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text).numpy()[0]
        logits_per_text = torch.FloatTensor(np.dot(img_feats, text_features))
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        img_ind = probs.argsort()
        ret = [(probs[i], fnames[i])  for i in img_ind]
        topK_img = ret[-K:]
        return topK_img


@app.route("/search", methods=['GET','POST'])
def query_images():
    query = request.args.get('text')
    logging.info("Query:" + query)

    top_img = get_topK_images(query)

    #Send the top image
    print(top_img[0][1])
    return send_file("static/images/"+top_img[0][1])


if __name__ == "__main__":
    load_clip_model()
    logging.info("Models loaded..")

    load_image_db("data/img-hash.json")
    logging.info("Database loaded..")

    app.run(host="0.0.0.0", port=8000, debug=True)
