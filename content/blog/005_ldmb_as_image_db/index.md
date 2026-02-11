---
title: LMDB as an image database
author: Avinash Mallya
date: 2026-02-10
tags: [python, machine-learning, storage, images, dvc]
---

# Premise

I was building a model for processing images (OCR, classification, object detection all bundled in), and I found myself with another problem - too many images to store efficiently as individual files on disk.

I had several thousand images already.
I was expecting several thousand more.
My repository was tracking these images via DVC.
My computer was also slowing down massively because of the sheer number of files.
DVC itself was slowing down (after all, randomly accessing many files isn't going to be fast).
I also needed to access files at random for training/evaluating the model (lots of shuffling).
Lastly, these images had their own associated metadata (labels, bounding boxes, "correct" text etc.), and they need to be stored along with the images - or least easily linkable to them.

I was primarily aiming for a "simple" solution, and didn't need a productionizable codebase.

# Potential Solutions

## Partitioning

A typical solution for "too many files" is to partition them by their name. It's ideal if their
name is a hash, so you can store the first character of the hash, then the second character, and
then the actual file. So, for example, the directory changes from:

```bash
files/
├── a1b2c3d4e5.txt
├── b7f8a9c0d1.txt
├── b7e4d2f1a0.txt
├── cf1a2b3c4d.txt
├── c0d1e2f3a4.txt
└── a1f5e6d7c8.txt
```

to

```bash
files/
├── a/
│   └── 1/
│       ├── a1b2c3d4e5.txt
│       └── a1f5e6d7c8.txt
├── b/
│   └── 7/
│       ├── b7f8a9c0d1.txt
│       └── b7e4d2f1a0.txt
└── c/
    ├── 0/
    │   └── c0d1e2f3a4.txt
    └── f/
        └── cf1a2b3c4d.txt
```

This isn't novel - `git` and `dvc` both store their objects this way.
This limits directory size, so file system look ups take less time.

This isn't a perfect solution - it required that I store the images as their hash,
and handle the directory structure correctly. I will also need to maintain my own
mechanism to maintain the link between the hash and the metadata, which means creating
some sort of index. Lastly, DVC will still track files individually, which means that
its `push`, `diff` and `pull` commands will still be slow.

## Separate (object) storage, maintain only the index

Another solution would be to offload the storage to a medium capable of handling a large number of files in any order,
while maintaining random access - for example, S3. Now my task will reduce to just correctly maintaining the index so
that I know how many images are stored on S3, and link their metadata via their hash.

If you've experienced retrieving a list of a large number of files stored on S3, you'd have first encountered the limit
of 1000 objects that `boto3` enforces per request. You'll need to work around it with pagination, which while standard,
is still more work. You would have also realized that even after all this, S3 will take quite some time to give you the
list even after you've optimized as much as you can.

However, that means I need an active internet connection to access any data. It also introduces latency during training,
and quite wasteful bandwidth in running multiple training sessions for the models. I can minimize these if I store the
images in the same AWS region as I would be running the training in (say, an EC2), but that means I needed access to an
EC2.

This still left me the task of maintaining my own index, which I really wanted to avoid, as it would mean additional
maintenance burden for a relatively nascent project that hasn't reached production status, while demanding production
code for an even more nascent pipeline.

## What about a... database?

This feels much like reaching for your nose by looping your hand behind your head instead of touching it directly with
your fingers. A database is great for many things, but setting one up and maintaining it (even a local SQLite one) isn't
really a quick and painless process, and has many gotchas. For instance, most databases aren't optimized for storing a
large number of binary blobs.

I considered, and tested using a more modern embeddable analytical database like DuckDB for this purpose (I'm quite
biased to using DuckDB and/or Polars to solve a large number of my data processing problems). I quickly
found out that storing large binary blobs in it causes it to choke (which is fair, it isn't really designed for that). Storing
the files elsewhere while maintaining just the index in it still had the original problem - I needed to write the mechanism
of maintaining the index.

> Note: HuggingFace now provides many images datasets (such as [MNIST](https://huggingface.co/datasets/ylecun/mnist)) in
the Parquet format, with the images stored using Arrow's extension types (but still as binary blobs). My experience with
storing binary data in Parquet hasn't been great, but you could check this out to see if it meets your requirements.

# The solution I landed on

## What about a... *different* kind of database?

Let's get down to first principles. What did I want to do? I wanted to store images. With those images, I also wanted
to store its metadata. I wanted to access said data quickly. It became clearer to me that I was looking for a fast key-value
store, and I stumbled upon [LMDB](https://www.symas.com/mdb).

## LMDB

Wikipedia's [entry](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) on LMDB indicates that it's an incredibly
small (64kB) piece of software that does one thing really well - be a ridiculously fast key-value store. I won't pretend to
understand how it works, the writeup on the Wiki provides plenty of good detail. I'll focus, rather, on how I used it to
solve my problem.

## Storing and retrieving image data along with its metadata

LMDB is rather barebones. It exposes few features - the ability to write, and read a particular key (stored as a bytestring),
that itself points to arbitrary bytes. I used its [Python bindings](https://lmdb.readthedocs.io/en/release/).

I wrote a tiny class (~200 LoC) that did the following:

1. Read the file names and the data of the images that I had, along with their metadata as it currently was, into Python. Batched to avoid running out of memory.
2. Serialize the metadata, and read the image files as bytes, and link them to the keys f"{file_name}_image" and f"{file_name}_metadata" respectively.
3. Store these as key-value pairs into the LMDB database, which is a **single** file.
4. Provided a method to read the keys to identify all the images present in the database.
5. Provided a method to retrieve an arbitrary set of images and their metadata quickly from the saved database.

This has many advantages:

1. LMDB is fast - really, really fast. Random access? Check. Fast retrieval of available images? Check.
2. DVC tracking becomes simple - maintain a single file, and just version control that. No slow downs due to sheer number of files - either for DVC, or my computer.
3. No index to maintain - the images and their metadata are stored in the same location, and linkable via a mere change in the suffix to their file name.
4. Local access, practically zero latency.

Which solves... all of the problems I had! When new files come in, all I needed to do was add them to the DB. LMDB has a few options available - you can
avoid overwriting the same key, ensure that the database is de-duplicated.

# The code

I've provided a sample code below that demonstrates storing just the images (not metadata) for the [Oxford 102 Category Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
dataset, which has around 8000 images.

```py
from pathlib import Path
import lmdb


class ImageDB:
    def __init__(self, env_path: Path, max_size_as_mb: int):
        self.env_path = str(env_path)
        self.env = lmdb.open(self.env_path, map_size=max_size_as_mb * (2**20))
        self.db = self.env.open_db()

    def save_image(
        self,
        name: str,
        image_path: Path,
    ) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(name.encode(), image_path.read_bytes())

    def read_image(self, name: str) -> bytes:
        with self.env.begin(write=False) as txn:
            if image_as_bytes := txn.get(name.encode()):
                return image_as_bytes
            else:
                raise KeyError()

    def save_images(
        self,
        name_image: dict[str, Path],
    ) -> None:
        # Note: you might need to enforce a batch size here
        # to aovid running out of memory because this loads
        # all images sent to this function as bytes.
        with self.env.begin(write=True) as txn:
            item_tuples = [
                (k.encode(), image_path.read_bytes())
                for k, image_path in name_image.items()
            ]
            cursor = txn.cursor()
            consumed, added = cursor.putmulti(
                item_tuples, dupdata=False, overwrite=False
            )
            print(
                f"Saved {added:,} out of {len(name_image):,} images to the DB ({consumed - added:,} seem to already exist)."
            )

    def load_images(
        self,
        names: list[str],
    ) -> dict[str, bytes]:
        names_as_bytestrings = [x.encode() for x in names]
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            return {
                k.decode(): image_as_bytes
                for k, image_as_bytes in cursor.getmulti(names_as_bytestrings)
            }

    def delete_image(self, name: str):
        with self.env.begin(write=True) as txn:
            if txn.delete(name.encode()):
                print(f"Image {name} deleted successfully")
            else:
                raise KeyError()

    def retrieve_names(self) -> list[str]:
        with self.env.begin(write=False) as txn:
            return [x.decode() for x in txn.cursor().iternext(keys=True, values=False)]


if __name__ == "__main__":
    db = ImageDB(Path("./db/"), 512)
    name_image: dict[str, Path] = dict()

    # Save the results
    for image_path in Path("./data/jpg/").glob("*.jpg"):
        name_image[image_path.name] = image_path
        if len(name_image) >= 1000:
            db.save_images(name_image)
            name_image.clear()
    # Add last batch also
    db.save_images(name_image)

    del name_image
    # How many images have been stored?
    print(f"The DB has {len(db.retrieve_names()):,} images stored")

    name_image: dict[str, bytes] = dict()
    # Load the results from the DB and check if they match the files on disk
    for image_path in Path("./data/jpg").glob("*.jpg"):
        name_image[image_path.name] = image_path.read_bytes()
        if len(name_image) >= 1000:
            saved_name_image = db.load_images(list(name_image.keys()))
            assert name_image == saved_name_image
            name_image.clear()
    # Verify last batch also
    saved_name_image = db.load_images(list(name_image.keys()))
    assert name_image == saved_name_image

    print("All images stored are byte identical to the original ones!")

    db.env.close()
```

This should provide you a good starting point to implement additional features,
such as storing metadata, filtering required input by metadata (such as extracting
a specific label for evaluation) and so on.

# Caveats

## Avoid PIL, or pay the (small) price

One gotcha that I initially faced is that the images I saved wasn't the same as
the images that I retrieved. This wasn't LMDB's fault, this was because I was
reading the images from disk via PIL, and storing them as bytes in LMDB. PIL
decodes and encodes the image, so a roundtrip will not necessarily be identical,
even for lossless file formats (other than bitmap images).

Don't encode/re-encode the image before you store it, or be prepared for the
stored data to not be byte-identical.

## The `max_size_as_mb` argument

LMDB has an unusual design. You need to specify the upper bound of the DB size
upon creation, and if it exceeds this size, it will fail. You can edit this
later, with some caveats (on Windows, this will actually allocate the full
size).

## Concurrency and LMDB

LMDB, while extremely fast, has some considerations with concurrency. See
[the documentation](https://lmdb.readthedocs.io/en/release/#threads) for details.
It may not be suited for distributed workloads.

# Alternatives

This article covers a "quick and dirty" solution, and was before more purpose-built
solutions were available. Some alternatives are:

1. If you're comfortable operating directly on archives, a simple `tar` file will
do - it can provide an offset index to provide random access to data.
2. [Nvidia's WebDataset](https://github.com/webdataset/webdataset). Modern, open
source and purpose built for large scale deep learning.
3. [LanceDB](https://lancedb.com/), which describes itself as "designed for multimodal"
and "built for scale". It's built on top of Arrow, closely related to Parquet.
4. As mentioned, HuggingFace has multiple solutions to this, starting with Arrow backed
storage, and their own [`datasets`](https://huggingface.co/docs/datasets/index).


Use these if you want to scale to production level training.
