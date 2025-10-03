#!/usr/bin/env python3
"""Helpers to produce radar shadow/layover masks with ISCE2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import isceobj
from isceobj.Planet.Planet import Planet
from zerodop.topozero import createTopozero

_VALID_DEM_METHODS = {"SINC", "BILINEAR", "BICUBIC", "NEAREST", "AKIMA", "BIQUINTIC"}


@dataclass
class GeometryTask:
    """Container describing a single geometry computation."""

    label: str
    tag: str
    width: int
    length: int
    range_pixel_spacing: float
    prf: float
    radar_wavelength: float
    orbit: object
    sensing_start: object
    range_first_sample: float
    look_side: int
    polarization: Optional[str] = None


def _load_dem(dem_path: str):
    xml_path = dem_path if dem_path.endswith(".xml") else dem_path + ".xml"
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"DEM XML not found: {xml_path}")

    dem_image = isceobj.createDemImage()
    dem_image.load(xml_path)
    return dem_image


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


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _run_topo_task(
    task: GeometryTask,
    dem_image,
    planet,
    outdir: str,
    dem_interp: str,
) -> Tuple[Tuple[float, float, float, float], str]:
    outputs = {
        "lat": os.path.join(outdir, f"lat_{task.tag}.rdr"),
        "lon": os.path.join(outdir, f"lon_{task.tag}.rdr"),
        "hgt": os.path.join(outdir, f"hgt_{task.tag}.rdr"),
        "los": os.path.join(outdir, f"los_{task.tag}.rdr"),
        "inc": os.path.join(outdir, f"incLocal_{task.tag}.rdr"),
        "mask": os.path.join(outdir, f"shadowMask_{task.tag}.rdr"),
    }

    topo = createTopozero()
    topo.slantRangePixelSpacing = task.range_pixel_spacing
    topo.prf = task.prf
    topo.radarWavelength = task.radar_wavelength
    topo.orbit = task.orbit
    topo.width = task.width
    topo.length = task.length
    topo.lookSide = task.look_side
    topo.sensingStart = task.sensing_start
    topo.rangeFirstSample = task.range_first_sample
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


def _prepare_geometry_tasks(
    reader,
    burst_range: Optional[Tuple[int, int]],
) -> Tuple[List[GeometryTask], Dict[str, Optional[str]]]:
    tasks: List[GeometryTask] = []
    meta: Dict[str, Optional[str]] = {
        "sensor": getattr(reader, "family", None),
        "polarization": None,
    }

    if hasattr(reader, "product") and getattr(reader.product, "bursts", None):
        bursts = reader.product.bursts
        burst_iter = _burst_slice(bursts, burst_range)
        for burst_id, burst in burst_iter:
            pol = getattr(burst, "polarization", None)
            if pol and not meta["polarization"]:
                meta["polarization"] = pol

            tasks.append(
                GeometryTask(
                    label=f"burst_{burst_id:02d}",
                    tag=f"{burst_id:02d}",
                    width=burst.numberOfSamples,
                    length=burst.numberOfLines,
                    range_pixel_spacing=burst.rangePixelSize,
                    prf=1.0 / burst.azimuthTimeInterval,
                    radar_wavelength=burst.radarWavelength,
                    orbit=burst.orbit,
                    sensing_start=burst.sensingStart,
                    range_first_sample=burst.startingRange,
                    look_side=-1,
                    polarization=pol,
                )
            )
    elif hasattr(reader, "frame") and reader.frame is not None:
        frame = reader.frame
        instrument = frame.getInstrument()
        platform = instrument.getPlatform()
        pol = None
        if hasattr(frame, "getPolarization"):
            pol = frame.getPolarization()
        elif hasattr(frame, "polarization"):
            pol = frame.polarization
        meta["polarization"] = pol

        tasks.append(
            GeometryTask(
                label="frame",
                tag="frame",
                width=frame.getNumberOfSamples(),
                length=frame.getNumberOfLines(),
                range_pixel_spacing=instrument.getRangePixelSize(),
                prf=instrument.getPulseRepetitionFrequency(),
                radar_wavelength=instrument.getRadarWavelength(),
                orbit=frame.getOrbit(),
                sensing_start=frame.getSensingStart(),
                range_first_sample=frame.getStartingRange(),
                look_side=platform.pointingDirection,
                polarization=pol,
            )
        )
    else:
        raise ValueError("Reader does not expose bursts or frame metadata required for geometry generation")

    return tasks, meta


def generate_shadow_layover(
    *,
    reader,
    dem_path: str,
    output_dir: str,
    burst_range: Optional[Tuple[int, int]] = None,
    dem_interp: str = "BIQUINTIC",
) -> Dict[str, object]:
    """Generate shadow/layover masks for a parsed sensor reader."""

    method = dem_interp.upper()
    if method not in _VALID_DEM_METHODS:
        raise ValueError(f"Unsupported DEM interpolation method: {dem_interp}")

    dem_image = _load_dem(dem_path)
    tasks, meta = _prepare_geometry_tasks(reader, burst_range)

    outdir = _ensure_dir(output_dir)
    planet = Planet(pname="Earth")

    bboxes: List[Tuple[float, float, float, float]] = []
    unit_results: List[Dict[str, object]] = []

    for task in tasks:
        bbox, mask_path = _run_topo_task(task, dem_image, planet, outdir, method)
        bboxes.append(bbox)
        unit_results.append(
            {
                "label": task.label,
                "mask_path": mask_path,
                "bbox": bbox,
                "polarization": task.polarization,
            }
        )

    if not unit_results:
        raise RuntimeError("No valid geometry tasks were generated from the sensor reader")

    overall_bbox = (
        min(b[0] for b in bboxes),
        max(b[1] for b in bboxes),
        min(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )

    return {
        "output_directory": outdir,
        "units": unit_results,
        "overall_bbox": overall_bbox,
        "metadata": meta,
    }


__all__ = ["generate_shadow_layover"]
