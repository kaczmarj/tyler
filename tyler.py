"""Extract tiles from whole slide images."""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict
from typing import NamedTuple
from typing import Tuple
from typing import Union

import numpy as np
import openslide
import PIL
from PIL.PngImagePlugin import PngInfo
import tqdm

PathType = Union[Path, str]


def _get_num_tiles(
    slide_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    strides: Tuple[int, int] = None,
) -> Tuple[int, int]:
    """Get number of tiles that can be taken from a whole slide.

    Does not include partial tiles (i.e., at edges). This also does not consider any
    mask.

    Parameters
    ----------
    slide_size : (int, int)
        Size of slide as (cols, rows).
    tile_size : (int, int)
        Size of tile as (cols, rows).
    strides : (int, int)
        Strides as (cols, rows). By default, sets strides for
        non-overlapping tiles.

    Returns
    -------
    (int, int)
        Number of tiles as (cols, rows).
    """

    def f(slide, tile, stride):
        return (slide - tile) // stride + 1

    # By default, no overlap.
    strides = strides or tile_size
    num_tiles_w = f(slide_size[0], tile_size[0], strides[0])
    num_tiles_h = f(slide_size[1], tile_size[1], strides[1])
    return num_tiles_w, num_tiles_h


class Tile(NamedTuple):
    """Metadata for one tile.

    Parameters
    ----------
    oslide : openslide.OpenSlide
        The whole slide image with which this tile is associated.
    wsi_id : str
        A unique ID for the whole slide image.
    c, r : int
        Column and row of the top-left corner of the tile.
    cols, rows : int
        The number of columns and rows in the tile.
    level : int
        The level in the whole slide image from which the tile is taken.
    """

    oslide: openslide.OpenSlide
    wsi_id: str
    c: int  # column of the top-left
    r: int  # row of the top-left
    cols: int  # columns in the tile
    rows: int  # rows in the tile
    level: int

    @property
    def info(self) -> Dict[str, str]:
        d = self._asdict()
        d = {str(k): str(v) for k, v in d.items()}
        d["mppx"] = str(self.mpp[0])
        d["mppy"] = str(self.mpp[1])
        return d

    @property
    def mpp(self) -> Tuple[float, float]:
        """Microns per pixel in the x, y dimensions, respectively."""
        mppx = float(self.oslide.properties.get(openslide.PROPERTY_NAME_MPP_X, -1))
        mppy = float(self.oslide.properties.get(openslide.PROPERTY_NAME_MPP_Y, -1))
        return mppx, mppy

    def to_pil_image(self) -> PIL.Image.Image:
        """Return a region of the whole slide image as a Pillow image object."""
        img = self.oslide.read_region(
            location=(self.c, self.r),
            level=self.level,
            size=(self.cols, self.rows),
        )
        return img

    def to_png(self, path: PathType) -> Path:
        """Save a tile to disk as a PNG image with associated metadata.

        Parameters
        ----------
        path : Pathlike
            Path to save the PNG file.

        Returns
        -------
        pathlib.Path
        """
        # Save metadata directly into the PNG file.
        info = PngInfo()
        for k, v in self.info.items():
            info.add_text(k, v)
        info.add_text("timestamp", str(int(time.time())))
        img = self.to_pil_image()
        path = Path(path).with_suffix(".png")
        img.save(path, pnginfo=info)
        return path

    def is_in_mask(self, mask: np.ndarray) -> bool:
        """Return whether this tile is contained inside a boolean mask.

        Parameters
        ----------
        mask : array-like
            Boolean mask. The shape must be proportional to the shape of the whole
            slide image.

        Returns
        -------
        bool
        """
        # openslide shape is (cols, rows), and numpy is (rows, cols).
        mask = np.asarray(mask)
        down_sample_ratio: float = self.oslide.dimensions[0] / mask.shape[1]
        # check that the down_sample ratio is the same in both dimensions
        down_sample_ratio_rows: float = self.oslide.dimensions[1] / mask.shape[0]
        # sometimes the down sample ratios are very close but not exactly equal.
        if round(down_sample_ratio, 2) != round(down_sample_ratio_rows, 2):
            err = (
                f"Shape of mask {mask.shape[::-1]} is not proportional to shape of"
                f" whole slide image {self.oslide.dimensions}."
            )
            raise ValueError(err)
        smallc = round(self.c / down_sample_ratio)
        smallr = round(self.r / down_sample_ratio)
        smallcols = round(self.cols / down_sample_ratio)
        smallrows = round(self.rows / down_sample_ratio)
        return mask[smallr : smallr + smallrows, smallc : smallc + smallcols].any()


def get_tiles(
    oslide: openslide.OpenSlide,
    wsi_id: str,
    level: int,
    tile_size: Tuple[int, int],
    strides: Tuple[int, int] = None,
) -> np.ndarray:
    """Return 2D array of Tile instances.

    These Tile instances include metadata
    (e.g., position, size) for each tile in the whole slide image.

    Parameters
    ----------
    oslide : openslide.OpenSlide
        The whole slide image.
    wsi_id : str
        The unique ID of this whole slide image.
    level : int
        Level of the whole slide image from which to extract tiles.
    tile_size : (int, int)
        Size of tiles as (cols, rows).
    strides : (int, int)
        Size of strides as (cols, rows). By default, uses
        strides equal to tile size, to create non-overlapping tiles.

    Returns
    -------
    np.ndarray
        Two-dimensional array of Tile instances. The Tile
        instance of position col,row in the array corresponds to its
        position on the whole slide image.
    """

    num_tiles_w, num_tiles_h = _get_num_tiles(
        slide_size=oslide.dimensions, tile_size=tile_size, strides=strides
    )
    # Get position for each tile (top-left).
    strides = strides or tile_size
    cs = list(range(0, oslide.dimensions[0], strides[0]))
    rs = list(range(0, oslide.dimensions[1], strides[1]))
    cs = cs[:num_tiles_w]
    rs = rs[:num_tiles_h]

    # sanity check that last tile does not go past slide dimensions.
    assert (cs[-1] + tile_size[0]) <= oslide.dimensions[0]
    assert (rs[-1] + tile_size[1]) <= oslide.dimensions[1]

    # Shape is in format (cols, rows).
    result = np.empty((num_tiles_w, num_tiles_h), dtype=object)
    for i, col in zip(range(num_tiles_w), cs):
        for j, row in zip(range(num_tiles_h), rs):
            result[i, j] = Tile(
                oslide=oslide,
                wsi_id=wsi_id,
                c=col,
                r=row,
                cols=tile_size[0],
                rows=tile_size[1],
                level=level,
            )
    if any(t is None for t in result.flat):
        raise ValueError("Array not filled. This should have never happened...")

    return result


def _get_parsed_args(args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("input", help="Path to whole slide image.")
    p.add_argument(
        "mask", help="Path to corresponding tissue mask. Excludes black regions."
    )
    p.add_argument(
        "-o", "--output", required=True, help="Root directory for output images."
    )
    p.add_argument(
        "-t",
        "--tile-size",
        required=True,
        type=int,
        nargs=2,
        help="(width, height) of each tile in pixels.",
    )
    p.add_argument(
        "-s",
        "--strides",
        type=int,
        nargs=2,
        help="(cols, rows) of overlap between tiles in pixels.",
    )
    p.add_argument(
        "--format",
        default="{t.c}_{t.r}_{t.cols}_{t.rows}_{t.mpp[0]}_{t.mpp[1]}.png",
    )
    p.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite image if they exist.",
    )
    return p.parse_args(args)


def main(args=None):
    ns = _get_parsed_args(args)
    ns.input = Path(ns.input)
    ns.output = Path(ns.output)
    output_dir = ns.output / ns.input.name

    if output_dir.exists() and not ns.force:
        print("Output directory exists. To potentially overwrite, re-run with --force.")
        return

    oslide = openslide.OpenSlide(str(ns.input))
    tiles = get_tiles(
        oslide=oslide,
        wsi_id=ns.input.stem,
        level=0,
        tile_size=ns.tile_size,
        strides=ns.strides,
    )
    output_dir.mkdir(exist_ok=True)

    mask = PIL.Image.open(ns.mask)
    # remove optional transparency and then convert to grayscale
    mask = mask.convert("RGB").convert("L")
    mask = np.asarray(mask)

    for tile in tqdm.tqdm(tiles.flat):
        if tile.is_in_mask(mask):
            output_name = ns.format.format(t=tile)
            tile.to_png(output_dir / output_name)


if __name__ == "__main__":
    sys.exit(main())
