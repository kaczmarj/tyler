from pathlib import Path
import time
from typing import Tuple

import skimage.exposure
import skimage.io
import numpy as np
from numpy.testing import assert_array_equal
import openslide
import PIL
import pytest

from tyler import get_tiles
from tyler import Tile


@pytest.fixture
def test_image_and_slide(tmp_path):
    """Create an image of a gradient."""
    rows, cols = 200, 150
    img = np.arange(rows * cols * 3).reshape(rows, cols, 3)
    img = skimage.exposure.rescale_intensity(img, out_range="uint8")
    img_path = tmp_path / "foobar.tiff"
    skimage.io.imsave(img_path, img, plugin="tifffile", tile=(16, 16))
    slide = openslide.OpenSlide(str(img_path))
    assert slide.dimensions == (cols, rows)
    return img, img_path


# tile_size and strides are cols, rows.
@pytest.mark.parametrize(
    "tile_size,strides,num_tiles",
    [
        ((15, 20), (15, 20), (10, 10)),
        # Tile is same size as image.
        ((150, 200), (150, 200), (1, 1)),
        # Overlapping in cols, non-overlapping in rows.
        ((15, 20), (5, 20), (28, 10)),
        # Overlapping in rows, non-overlapping in cols.
        ((15, 20), (15, 10), (10, 19)),
        # Overlapping in rows and cols.
        ((15, 20), (3, 5), (46, 37)),
        # Overlapping in rows and cols.
        ((15, 20), (3, 5), (46, 37)),
        # Tiles that don't divide evenly into image.
        ((20, 15), (20, 15), (7, 13)),
    ],
)
def test_integration(
    tmp_path: Path,
    test_image_and_slide: Tuple[np.ndarray, Path],
    tile_size: Tuple[int, int],
    strides: Tuple[int, int],
    num_tiles: Tuple[int, int],
):
    img, img_path = test_image_and_slide
    slide = openslide.OpenSlide(str(img_path))
    assert slide.dimensions == (img.shape[1], img.shape[0])

    # Non-overlapping tiles.
    tiles = get_tiles(slide, "012", level=0, tile_size=tile_size, strides=strides)
    assert tiles.shape == num_tiles, "number of tiles not correct"

    stride_c, stride_r = strides
    tile_size_c, tile_size_r = tile_size
    for tile_c in range(tiles.shape[0]):
        for tile_r in range(tiles.shape[1]):
            # Figure out bounds.
            ref_start_c = stride_c * tile_c
            ref_end_c = ref_start_c + tile_size_c
            ref_start_r = stride_r * tile_r
            ref_end_r = ref_start_r + tile_size_r
            # Get data for tests.
            # Note that openslide/PIL use (col,row) format and skimage uses
            # (row,col) format.
            tst = np.array(tiles[tile_c, tile_r].to_pil_image())[..., :3]
            ref = img[ref_start_r:ref_end_r, ref_start_c:ref_end_c]
            assert_array_equal(tst, ref)


def test_tile_info(test_image_and_slide: Tuple[np.ndarray, Path]):
    img, img_path = test_image_and_slide
    slide = openslide.OpenSlide(str(img_path))
    assert slide.dimensions == (img.shape[1], img.shape[0])

    tile = Tile(oslide=slide, wsi_id="0123", c=0, r=0, cols=50, rows=10, level=0)
    assert tile.info == dict(
        oslide=str(tile.oslide),
        wsi_id=str(tile.wsi_id),
        c=str(tile.c),
        r=str(tile.r),
        cols=str(tile.cols),
        rows=str(tile.rows),
        level=str(tile.level),
    )


def test_tile_save(tmp_path: Path, test_image_and_slide: Tuple[np.ndarray, Path]):
    img, img_path = test_image_and_slide
    slide = openslide.OpenSlide(str(img_path))
    assert slide.dimensions == (img.shape[1], img.shape[0])

    tile = Tile(oslide=slide, wsi_id="0123", c=0, r=0, cols=50, rows=10, level=0)
    path = tile.to_png(tmp_path / "tile.png")

    tst_img = PIL.Image.open(path)
    tst = np.array(tst_img)[..., :3]

    # Test that data are the same.
    ref = img[:10, :50, :3]
    assert_array_equal(ref, tst)

    # Test that metadata were saved properly.
    ref_text = dict(
        oslide=str(tile.oslide),
        wsi_id=str(tile.wsi_id),
        c=str(tile.c),
        r=str(tile.r),
        cols=str(tile.cols),
        rows=str(tile.rows),
        level=str(tile.level),
    )
    # The png text has timestamp as well but Tile.info does not.
    tst_time = tst_img.text.pop("timestamp")
    assert ref_text == tst_img.text
    now = int(time.time())
    assert (now - int(tst_time)) < 5, "wrong timestamp"
