#!/usr/bin/env python3
"""Helpers to produce Sentinel-1 shadow/layover masks with ISCE2."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import isceobj
from isceobj.Planet.Planet import Planet
from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
from zerodop.topozero import createTopozero

# Accepted Topozero interpolation identifiers
_VALID_DEM_METHODS = {"SINC", "BILINEAR", "BICUBIC", "NEAREST", "AKIMA", "BIQUINTIC"}


def _load_dem(dem_path: str):
    xml_path = dem_path if dem_path.endswith(".xml") else dem_path + ".xml"
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"DEM XML not found: {xml_path}")

    dem_image = isceobj.createDemImage()
    dem_image.load(xml_path)
    return dem_image


def _configure_reader(
    *,
    safe_paths: Optional[Sequence[str]],
    annotation_paths: Optional[Sequence[str]],
    manifest_paths: Optional[Sequence[str]],
    swath: int,
    polarization: str,
    orbit_file: Optional[str],
    orbit_dir: Optional[str],
    aux_dir: Optional[str],
) -> Sentinel1:
    reader = Sentinel1()
    reader.configure()

    if annotation_paths:
        reader.xml = list(annotation_paths)
    elif safe_paths:
        reader.safe = list(safe_paths)
    else:
        raise ValueError("Provide either SAFE paths or annotation XML paths")

    if manifest_paths:
        reader.manifest = list(manifest_paths)

    reader.swathNumber = swath
    reader.polarization = polarization.lower()
    reader.orbitFile = orbit_file
    reader.orbitDir = orbit_dir
    reader.auxDir = aux_dir

    reader.parse()
    return reader


def _burst_slice(
    bursts: Sequence,
    burst_range: Optional[Tuple[int, int]],
) -> Iterable[Tuple[int, object]]:
    if burst_range is None:
        start = 1
        stop = len(bursts)
    else:
        start, stop = burst_range
        if start < 1 or stop < start or stop > len(bursts):
            raise ValueError(
                f"Invalid burst range {start}-{stop}; product provides {len(bursts)} bursts",
            )

    for offset, burst in enumerate(bursts[start - 1:stop], start=start):
        yield offset, burst


def _ensure_outdir(outdir: str, swath: int) -> str:
    swath_dir = os.path.join(outdir, f"IW{swath}")
    os.makedirs(swath_dir, exist_ok=True)
    return swath_dir


def _topo_output_paths(swath_dir: str, burst_id: int) -> Dict[str, str]:
    tag = f"{burst_id:02d}"
    return {
        "lat": os.path.join(swath_dir, f"lat_{tag}.rdr"),
        "lon": os.path.join(swath_dir, f"lon_{tag}.rdr"),
        "hgt": os.path.join(swath_dir, f"hgt_{tag}.rdr"),
        "los": os.path.join(swath_dir, f"los_{tag}.rdr"),
        "inc": os.path.join(swath_dir, f"incLocal_{tag}.rdr"),
        "mask": os.path.join(swath_dir, f"shadowMask_{tag}.rdr"),
    }


def _run_topo_for_burst(
    burst,
    burst_id: int,
    dem_image,
    planet,
    swath_dir: str,
    dem_interp: str,
) -> Tuple[Tuple[float, float, float, float], str]:
    outputs = _topo_output_paths(swath_dir, burst_id)

    topo = createTopozero()
    topo.slantRangePixelSpacing = burst.rangePixelSize
    topo.prf = 1.0 / burst.azimuthTimeInterval
    topo.radarWavelength = burst.radarWavelength
    topo.orbit = burst.orbit
    topo.width = burst.numberOfSamples
    topo.length = burst.numberOfLines
    topo.lookSide = -1  # Sentinel-1 is right-looking
    topo.sensingStart = burst.sensingStart
    topo.rangeFirstSample = burst.startingRange
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = 1
    topo.demInterpolationMethod = dem_interp
    topo.latFilename = outputs["lat"]
    topo.lonFilename = outputs["lon"]
    topo.heightFilename = outputs["hgt"]
    topo.losFilename = outputs["los"]
    topo.incFilename = outputs["inc"]
    topo.maskFilename = outputs["mask"]

    topo.wireInputPort(name="dem", object=dem_image)
    topo.wireInputPort(name="planet", object=planet)

    topo.topo()

    bbox = (
        topo.minimumLatitude,
        topo.maximumLatitude,
        topo.minimumLongitude,
        topo.maximumLongitude,
    )
    return bbox, outputs["mask"]


def generate_shadow_layover(
    *,
    dem_path: str,
    output_dir: str,
    swath: int,
    polarization: str = "vv",
    safe_paths: Optional[Sequence[str]] = None,
    annotation_paths: Optional[Sequence[str]] = None,
    manifest_paths: Optional[Sequence[str]] = None,
    orbit_file: Optional[str] = None,
    orbit_dir: Optional[str] = None,
    aux_dir: Optional[str] = None,
    burst_range: Optional[Tuple[int, int]] = None,
    dem_interp: str = "BIQUINTIC",
) -> Dict[str, object]:
    """Generate shadow/layover masks for a given Sentinel-1 swath.

    Parameters are passed explicitly so the function can be consumed in
    external environments without relying on command-line parsing.

    Returns
    -------
    dict
        Summary dictionary containing per-burst outputs and overall
        bounding boxes.
    """

    method = dem_interp.upper()
    if method not in _VALID_DEM_METHODS:
        raise ValueError(f"Unsupported DEM interpolation method: {dem_interp}")

    if swath not in (1, 2, 3):
        raise ValueError("Swath number must be one of 1, 2, or 3")

    dem_image = _load_dem(dem_path)
    reader = _configure_reader(
        safe_paths=safe_paths,
        annotation_paths=annotation_paths,
        manifest_paths=manifest_paths,
        swath=swath,
        polarization=polarization,
        orbit_file=orbit_file,
        orbit_dir=orbit_dir,
        aux_dir=aux_dir,
    )

    swath_dir = _ensure_outdir(output_dir, swath)
    planet = Planet(pname="Earth")

    burst_iter = _burst_slice(reader.product.bursts, burst_range)

    bboxes: List[Tuple[float, float, float, float]] = []
    burst_results: List[Dict[str, object]] = []

    for burst_id, burst in burst_iter:
        bbox, mask_path = _run_topo_for_burst(
            burst=burst,
            burst_id=burst_id,
            dem_image=dem_image,
            planet=planet,
            swath_dir=swath_dir,
            dem_interp=method,
        )
        bboxes.append(bbox)
        burst_results.append({"burst_id": burst_id, "mask_path": mask_path, "bbox": bbox})

    if not burst_results:
        raise RuntimeError("No bursts processed; check burst range or input coverage")

    overall_bbox = (
        min(b[0] for b in bboxes),
        max(b[1] for b in bboxes),
        min(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )

    return {
        "swath": swath,
        "polarization": polarization,
        "output_directory": swath_dir,
        "bursts": burst_results,
        "overall_bbox": overall_bbox,
    }
