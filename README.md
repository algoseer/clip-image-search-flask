## Query images using text using OpenAI clip

This projects queries a database of images using a NL query and find the most fitting image according to OpenAI CLIP model. Images are prehashed by computing their embeddings and stored in a local file. Text embeddings are computed at serve time and logits calculated to find the top image.

## Flask component
Flask is used to build a REST API that takes text queries and serves the top image in the database. All images are locally stored or can alternatively be served from another location.

## Other files needed
To run the flask container you will also need image features for your dataset precomputed. 
The following files/folders are required to run: static/images/, data/img-hash.json

More instructions on that coming soon..

## Building the container

```
docker build -f dockerfile -t clip-img-srch:<tag> .
```

To run
```
docker run -p 8000:8000 clip-img-srch:<tag>
```
