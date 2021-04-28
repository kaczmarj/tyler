"""Extract tiles from whole slide images."""

from pathlib import Path
import time
from typing import Dict, NamedTuple, Tuple, Union
import uuid

import numpy as np
import openslide
import PIL
from PIL.PngImagePlugin import PngInfo

PathType = Union[Path, str]


def get_num_tiles(
    slide_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    strides: Tuple[int, int] = None,
) -> Tuple[int, int]:
    """Get number of tiles that can be taken from a whole slide.

    Does not include partial tiles (i.e., at edges).

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


class TileMeta(NamedTuple):
    """Metadata for one tile.

    Includes information like position and size.
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
        return {str(k): str(v) for k, v in d.items()}

    def to_pil_image(self) -> PIL.Image.Image:
        img = self.oslide.read_region(
            location=(self.c, self.r),
            level=self.level,
            size=(self.cols, self.rows),
        )
        return img


def get_tilemetas(
    oslide: openslide.OpenSlide,
    wsi_id: str,
    level: int,
    tile_size: Tuple[int, int],
    strides: Tuple[int, int] = None,
    tile_path: PathType = Path("tiles"),
) -> np.ndarray:
    """Return 2D array of TileMeta instances.

    These TileMeta instances include metadata
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
        Two-dimensional array of TileMeta instances. The TileMeta
        instance of position col,row in the array corresponds to its
        position on the whole slide image.
    """

    num_tiles_w, num_tiles_h = get_num_tiles(
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
            result[i, j] = TileMeta(
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


def save_one(directory: PathType, tilemeta: TileMeta) -> Path:
    """Save one tile to disk as a PNG image with associated metadata."""
    directory = Path(directory)
    # Save metadata directly into the PNG file.
    info = PngInfo()
    for k, v in tilemeta.info.items():
        info.add_text(k, v)
    info.add_text("timestamp", str(int(time.time())))
    img = tilemeta.to_pil_image()
    name = f"{uuid.uuid4()}.png"
    path = directory / name
    img.save(path, pnginfo=info)
    return path
