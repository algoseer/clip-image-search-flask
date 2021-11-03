FROM pytorch/pytorch

RUN apt-get update && apt-get install -y git

COPY ./ /clipsearch

RUN pip3 install -r /clipsearch/req.txt

WORKDIR /clipsearch

CMD [ "python3", "/clipsearch/serve.py" ]
