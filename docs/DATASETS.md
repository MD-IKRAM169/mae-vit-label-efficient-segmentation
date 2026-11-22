# Data installation

Do not commit full datasets, annotations, private images, or GPU checkpoints to this repository.

## ADE20K

Dataset information: https://groups.csail.mit.edu/vision/datasets/ADE20K/

Expected root \`/path/to/ADEChallengeData2016\`:
~~~text
images/training/*.jpg
images/validation/*.jpg
annotations/training/*.png
annotations/validation/*.png
~~~
Standard ADE20K semantic mask IDs 1..150 become zero-based 0..149. Void/0 is ignored (mapped to 255). Keep nearest-neighbor resizing for masks.

## Cityscapes

Official site and licensing: https://www.cityscapes-dataset.com/

Expected root:
~~~text
leftImg8bit/train/<city>/*_leftImg8bit.png
leftImg8bit/val/<city>/*_leftImg8bit.png
gtFine/train/<city>/*_gtFine_labelIds.png
gtFine/val/<city>/*_gtFine_labelIds.png
~~~
The loader also accepts \`_gtFine_labelTrainIds.png\` if generated separately. Official raw label IDs are mapped to the 19 semantic train IDs; all other values become ignore 255.

For a smoke test without downloads: \`python scripts/make_toy_data.py --output-dir data/toy_ade20k\`. Synthetic examples have **no relation to benchmark accuracy**.

The 10%-subset code samples training images with a fixed random seed (default 42). It is not an original split-ID manifest from the 2025 experiments.
