FROM pytorch/pytorch

COPY ./ /clipsearch

RUN pip3 install -r /clipsearch/req.txt

WORKDIR /clipsearch

CMD [ "python3", "/clipsearch/serve.py" ]
