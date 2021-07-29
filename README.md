# tyler

Python module to extract tiles from whole slide images.

![GitHub Workflow Status](https://github.com/kaczmarj/tyler/actions/workflows/ci.yml/badge.svg)
![Codecov](https://img.shields.io/codecov/c/github/kaczmarj/tyler)

# Examples

## Command-line

```bash
python -m tyler \
    -o test-tiles \
    --tile-size 1000 1000 \
    wsi.svs mask.png
```

## In a Python script

```python
import openslide
import PIL
import tyler

slide = openslide.OpenSlide(
    "TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs"
)
print("Slide dimensions (cols, rows): ", slide.dimensions)
tiles = tyler.get_tiles(
    oslide=slide,
    wsi_id="TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7",
    level=0,
    tile_size=(1000, 1000),
)
print("Number of tiles (cols, rows):", tiles.shape)

# Plot the top-left tile.
tiles[0, 0].to_pil_image()

# Save to disk.
tiles[0, 0].to_png("top-left-tile.png")

# Read the PNG and get its metadata.
# Tyler is generous and saves important metadata within the PNG file.
img = PIL.Image.open("top-left-tile.png")
print("Image metadata")
for key, value in img.text.items():
    print(f"  {key}: {value}")
```

The output of the preceding code

```
Slide dimensions (cols, rows):  (32001, 38474)
Number of tiles (cols, rows): (32, 38)
Image metadata
  oslide: OpenSlide('TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs')
  wsi_id: TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7
  c: 0
  r: 0
  cols: 1000
  rows: 1000
  level: 0
  timestamp: 1619631433
```

# Installation

Install with `pip`:

```bash
python -m pip install --no-cache-dir https://github.com/kaczmarj/tyler/tarball/main
```

Tyler depends on [OpenSlide](https://openslide.org/), which must be installed separately. See [OpenSlide's download page](https://openslide.org/download/) for more information.

## Developer installation

Create a virtual environment, and install development dependencies.

```bash
git clone https://github.com/kaczmarj/tyler
cd tyler
python -m venv venv
source ./venv/bin/activate
python -m pip install --no-cache-dir --editable .[dev]
```

# Musings on data orientation

- Openslide `slide.dimensions` orients data as `(cols, rows)`.
- PIL orients data as `(cols, rows)`.
- Scikit-image orients data as `(rows, cols, channels)`.
- Matplotlib pyplot imshow expects data as `(rows, cols, [channels])`.

where `[]` indicates optional axis.

---

<p align="center">
    <img src="https://i.redd.it/avif889dhh751.jpg" alt="Dog in hoodie." width="200">
</p>
