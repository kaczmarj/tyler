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

from tyler import Tile
from tyler import get_tiles
from tyler import main


@pytest.fixture
def sample_img_and_wsi_path(tmp_path):
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
def test_get_tiles(
    sample_img_and_wsi_path: Tuple[np.ndarray, Path],
    tile_size: Tuple[int, int],
    strides: Tuple[int, int],
    num_tiles: Tuple[int, int],
):
    img, img_path = sample_img_and_wsi_path
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


def test_tile_info(sample_img_and_wsi_path: Tuple[np.ndarray, Path]):
    img, img_path = sample_img_and_wsi_path
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
        mppx="-1.0",
        mppy="-1.0",
    )


def test_tile_save(tmp_path: Path, sample_img_and_wsi_path: Tuple[np.ndarray, Path]):
    img, img_path = sample_img_and_wsi_path
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
        mppx="-1.0",
        mppy="-1.0",
    )
    # The png text has timestamp as well but Tile.info does not.
    tst_time = tst_img.text.pop("timestamp")
    assert ref_text == tst_img.text
    now = int(time.time())
    assert (now - int(tst_time)) < 5, "wrong timestamp"


def test_tile_in_mask(sample_img_and_wsi_path: Tuple[np.ndarray, Path]):
    _, img_path = sample_img_and_wsi_path
    slide = openslide.OpenSlide(str(img_path))
    # tiles have same shape as the upsampled mask (3, 4)
    tiles = get_tiles(slide, "foo", tile_size=(50, 50), strides=None, level=0)
    # TODO: this part depends on the shape of the whole slide image...
    mask = np.zeros((4, 3))  # shape of mask is rows,cols but slide is cols,rows
    mask[0, 0] = 1
    mask[:, 2] = 1
    mask[3, 1] = 1
    assert tiles[0, 0].is_in_mask(mask)
    assert not tiles[0, 1].is_in_mask(mask)
    assert not tiles[0, 2].is_in_mask(mask)
    assert not tiles[0, 3].is_in_mask(mask)
    assert not tiles[1, 0].is_in_mask(mask)
    assert not tiles[1, 1].is_in_mask(mask)
    assert not tiles[1, 2].is_in_mask(mask)
    assert tiles[1, 3].is_in_mask(mask)
    assert tiles[2, 0].is_in_mask(mask)
    assert tiles[2, 1].is_in_mask(mask)
    assert tiles[2, 2].is_in_mask(mask)
    assert tiles[2, 3].is_in_mask(mask)

    # bad mask shape
    mask = np.zeros((3, 3))
    with pytest.raises(ValueError):
        tiles[0, 0].is_in_mask(mask)

    # TODO: add test of strided tiles


def test_main(tmp_path: Path, sample_img_and_wsi_path: Tuple[np.ndarray, Path]):
    _, img_path = sample_img_and_wsi_path
    mask = np.zeros((4, 3), dtype="uint8")
    mask[0, 0] = 255
    mask[:, 2] = 255
    mask[3, 1] = 255
    mask_path = tmp_path / "mask.png"
    PIL.Image.fromarray(mask).save(mask_path)

    tiles_png_dir = tmp_path / "tiles"
    tiles_png_dir.mkdir()
    args = [
        "--output",
        str(tiles_png_dir),
        "--tile-size",
        "50",
        "50",
        str(img_path),
        str(mask_path),
    ]
    main(args)

    # The filenames we expect...
    # 0_0_50_50_-1.0_-1.0.png    100_100_50_50_-1.0_-1.0.png  100_50_50_50_-1.0_-1.0.png
    # 100_0_50_50_-1.0_-1.0.png  100_150_50_50_-1.0_-1.0.png  50_150_50_50_-1.0_-1.0.png
    tile_pngs = list((tiles_png_dir / img_path.name).glob("*"))
    assert len(tile_pngs) == 6

    oslide = openslide.OpenSlide(str(img_path))
    # read the images and make sure they are the same as original.
    for tile_png in tile_pngs:
        c, r, cols, rows, _ = tile_png.name.split("_", maxsplit=4)
        t = Tile(
            oslide=oslide,
            wsi_id="",
            c=int(c),
            r=int(r),
            cols=int(cols),
            rows=int(rows),
            level=0,
        )
        reference_img = t.to_pil_image()
        loaded_img = PIL.Image.open(tile_png)
        np.testing.assert_array_equal(loaded_img, reference_img)

    # And a sanity check...
    assert not np.array_equal(reference_img, PIL.Image.open(tile_pngs[0]))
