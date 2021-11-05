## Query images using text using OpenAI clip

This projects queries a database of images using a NL query and find the most fitting image according to OpenAI CLIP model. Images are prehashed by computing their embeddings and stored in a local file. Text embeddings are computed at serve time and logits calculated to find the top image.

## Flask component
Flask is used to build a REST API that takes text queries and serves the top image in the database. All images are locally stored or can alternatively be served from another location.

## Other files needed
To run the flask container you will also need image features for your dataset precomputed. 
The following files/folders are required to run: static/images/, data/img-hash.json ( see below )

More instructions on that coming soon..

## Building image hash for search

1. Download the images to `static/images`
2. Run the script for generating image hash
Generate features for all \*.jpg files
```
find static/images -name *.jpg | python build_hash.py
```
3. Verify that the file `data/img-hash.json` was created.

## Running search in the container

1. Download the images to `static/images`

2. Download the precompute image hash to `data/img-hash.json`.

3. Build the container. This will pre-download the model one time so that it is not pulled at start each time.
```
docker build -f dockerfile -t clip-img-srch:<tag> .
```

4. Run the container
```
docker run --name clip-img-srch -d -p 8000:8000 clip-img-srch:<tag>
```
