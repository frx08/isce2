#!/usr/bin/env python3
"""High-level helpers to orchestrate Sentinel-1 shadow/layover generation."""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import isce
from calcShadowLayover import generate_shadow_layover
from contrib.demUtils import createDemStitcher
from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1

DEFAULT_SAFE_PRODUCT = (
    "/nas/products/0b2b1dc4-8366-4b69-baa4-b65b77ec3757/"
    "S1A_IW_SLC__1SDV_20230109T172319_20230109T172346_046710_059966_7103.zip"
)
DEFAULT_SWATHS = (1, 2, 3)


def _load_swath_reader(safe_path: str, swath: int, polarization: str) -> Sentinel1:
    reader = Sentinel1()
    reader.configure()
    reader.safe = [safe_path]
    reader.swathNumber = swath
    reader.polarization = polarization.lower()
    reader.parse()
    return reader


def _collect_scene_bbox(safe_path: str, swaths: Iterable[int], polarization: str) -> List[float]:
    bbox: Optional[List[float]] = None
    for swath in swaths:
        reader = _load_swath_reader(safe_path, swath, polarization)
        if reader.product.numberOfBursts == 0:
            continue
        swath_bbox = reader.product.getBbox()
        if bbox is None:
            bbox = list(swath_bbox)
        else:
            bbox[0] = min(bbox[0], swath_bbox[0])
            bbox[1] = max(bbox[1], swath_bbox[1])
            bbox[2] = min(bbox[2], swath_bbox[2])
            bbox[3] = max(bbox[3], swath_bbox[3])
    if bbox is None:
        raise RuntimeError("Unable to determine bbox; no bursts were available in the SAFE input")
    return bbox


def _expand_bbox_to_integers(snwe: Sequence[float]) -> List[int]:
    south = math.floor(snwe[0])
    north = math.ceil(snwe[1])
    west = math.floor(snwe[2])
    east = math.ceil(snwe[3])
    return [south, north, west, east]


def _ensure_dem(
    *,
    bbox_snwe: Sequence[float],
    dem_dir: str,
    source: int,
    existing_dem: Optional[str] = None,
) -> str:
    if existing_dem:
        dem_path = os.path.abspath(existing_dem)
        xml_path = dem_path if dem_path.endswith(".xml") else dem_path + ".xml"
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"DEM XML sidecar not found: {xml_path}")
        return dem_path

    os.makedirs(dem_dir, exist_ok=True)

    stitcher = createDemStitcher("version3")
    stitcher.configure()
    stitcher.setCreateXmlMetadata(True)
    stitcher.setKeepDems(True)

    snwe_int = _expand_bbox_to_integers(bbox_snwe)
    lat = snwe_int[0:2]
    lon = snwe_int[2:4]
    dem_name = stitcher.defaultName(snwe_int)

    if not stitcher.stitchDems(lat, lon, source, dem_name, downloadDir=dem_dir, keep=True):
        raise RuntimeError("DEM stitching failed; check network credentials or bbox coverage")

    return os.path.join(dem_dir, dem_name)


def generate_scene_shadow_masks(
    *,
    safe_path: str = DEFAULT_SAFE_PRODUCT,
    polarization: str = "vv",
    swaths: Sequence[int] = DEFAULT_SWATHS,
    output_dir: str = os.path.join(os.getcwd(), "shadow_layover"),
    dem_directory: str = os.path.join(os.getcwd(), "dem"),
    dem_source: int = 1,
    dem_path: Optional[str] = None,
    burst_range: Optional[Tuple[int, int]] = None,
    dem_interp: str = "BIQUINTIC",
    manifest_paths: Optional[Sequence[str]] = None,
    orbit_file: Optional[str] = None,
    orbit_dir: Optional[str] = None,
    aux_dir: Optional[str] = None,
) -> Dict[int, Dict[str, object]]:
    """Generate shadow/layover masks for all requested swaths.

    Returns a dictionary keyed by swath number with the results from
    :func:`applications.calcShadowLayover.generate_shadow_layover`.
    """

    bbox = _collect_scene_bbox(safe_path, swaths, polarization)
    dem_path_resolved = _ensure_dem(
        bbox_snwe=bbox,
        dem_dir=dem_directory,
        source=dem_source,
        existing_dem=dem_path,
    )

    os.makedirs(output_dir, exist_ok=True)

    burst_range_tuple = tuple(burst_range) if burst_range else None

    results: Dict[int, Dict[str, object]] = {}
    for swath in swaths:
        results[swath] = generate_shadow_layover(
            dem_path=dem_path_resolved,
            output_dir=output_dir,
            swath=swath,
            polarization=polarization,
            safe_paths=[safe_path],
            annotation_paths=None,
            manifest_paths=manifest_paths,
            orbit_file=orbit_file,
            orbit_dir=orbit_dir,
            aux_dir=aux_dir,
            burst_range=burst_range_tuple,
            dem_interp=dem_interp,
        )

    return results

generate_scene_shadow_masks()
