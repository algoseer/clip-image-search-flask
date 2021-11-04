FROM pytorch/pytorch

RUN apt-get update && apt-get install -y git

COPY ./ /clipsearch

RUN pip3 install -r /clipsearch/req.txt
RUN python3 /clipsearch/dl_model.py

WORKDIR /clipsearch

CMD [ "python3", "/clipsearch/serve.py" ]
