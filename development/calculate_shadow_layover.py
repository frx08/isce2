#!/usr/bin/env python3
"""High-level helpers to orchestrate radar shadow/layover generation."""

import os
from typing import Dict, Optional, Sequence, Tuple

# isce needs to be imported before any isceobj modules
import isce
from calcShadowLayover import generate_shadow_layover
from isceobj.Sensor import createSensor
from isceobj.Sensor.TOPS import createSentinel1 as create_tops_sentinel1
from isceobj.StripmapProc.runVerifyDEM import getBbox as stripmap_bbox
from contrib.demUtils import createDemStitcher

DEFAULT_SAFE_PRODUCT = (
    "/nas/products/0b2b1dc4-8366-4b69-baa4-b65b77ec3757/"
    "S1A_IW_SLC__1SDV_20230109T172319_20230109T172346_046710_059966_7103.zip"
)
DEFAULT_SWATHS = (1, 2, 3)

def _create_sensor(sensor_type: str, params: Dict[str, object]):
    sensor_upper = sensor_type.upper()
    if sensor_upper in {"SENTINEL1", "TOPS_SENTINEL1", "SENTINEL1_TOPS", "S1_TOPS"}:
        reader = create_tops_sentinel1()
    else:
        reader = createSensor(sensor_upper)
    if reader is None:
        raise ValueError(f"Unsupported sensor type: {sensor_type}")

    if hasattr(reader, "configure"):
        reader.configure()

    for key, value in params.items():
        if value is None:
            continue
        setattr(reader, key, value)

    reader.parse()
    return reader

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

    south, north, west, east = bbox_snwe
    print(f"Requesting DEM for bbox: S={south}, N={north}, W={west}, E={east}")
    bbox_int = [int(south), int(north + 0.9999), int(west), int(east + 0.9999)]
    lat = bbox_int[0:2]
    lon = bbox_int[2:4]
    dem_name = stitcher.defaultName(bbox_int)

    if not stitcher.stitchDems(lat, lon, source, dem_name, downloadDir=dem_dir, keep=True):
        raise RuntimeError("DEM stitching failed; check network credentials or bbox coverage")

    return os.path.join(dem_dir, dem_name)

def _union_bbox(bboxes: Sequence[Sequence[float]]) -> Sequence[float]:
    south = min(b[0] for b in bboxes)
    north = max(b[1] for b in bboxes)
    west = min(b[2] for b in bboxes)
    east = max(b[3] for b in bboxes)
    return [south, north, west, east]

def generate_scene_shadow_masks(
    *,
    sensor_type: str = "SENTINEL1",
    product: Optional[str] = DEFAULT_SAFE_PRODUCT,
    polarization: str = "vv",
    swaths: Optional[Sequence[int]] = None,
    output_dir: str = os.path.join(os.getcwd(), "shadow_layover"),
    dem_directory: str = os.path.join(os.getcwd(), "dem"),
    dem_source: int = 1,
    dem_path: Optional[str] = None,
    burst_range: Optional[Tuple[int, int]] = None,
    dem_interp: str = "BIQUINTIC",
    sensor_kwargs: Optional[Dict[str, object]] = None,
    manifest_paths: Optional[Sequence[str]] = None,
    orbit_file: Optional[str] = None,
    orbit_dir: Optional[str] = None,
    aux_dir: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Generate shadow/layover masks for the requested sensor acquisition."""

    sensor_type_upper = sensor_type.upper()
    params_base: Dict[str, object] = dict(sensor_kwargs or {})

    os.makedirs(output_dir, exist_ok=True)

    if sensor_type_upper == "SENTINEL1":
        safe_value = params_base.get("safe")
        if not safe_value:
            if product is None:
                raise ValueError("product must be provided for Sentinel-1 processing")
            safe_value = product
        if isinstance(safe_value, str):
            params_base["safe"] = [safe_value]
        else:
            params_base["safe"] = list(safe_value)

        if manifest_paths and "manifest" not in params_base:
            params_base["manifest"] = list(manifest_paths)
        elif isinstance(params_base.get("manifest"), str):
            params_base["manifest"] = [params_base["manifest"]]
        params_base.setdefault("orbitFile", orbit_file)
        params_base.setdefault("orbitDir", orbit_dir)
        params_base.setdefault("auxDir", aux_dir)
        params_base.setdefault("polarization", polarization.lower())

        swath_list = tuple(swaths) if swaths is not None else DEFAULT_SWATHS
        readers: Dict[str, object] = {}
        bboxes = []

        for swath in swath_list:
            params = dict(params_base)
            params["swathNumber"] = swath
            reader = _create_sensor(sensor_type_upper, params)
            readers[f"IW{swath}"] = reader
            bboxes.append(reader.product.getBbox())

    elif sensor_type_upper == "COSMO_SKYMED_SLC":
        params_base["hdf5"] = product
        params_base["output"] = os.path.join(output_dir, "scene.slc")
        if manifest_paths and "manifest" not in params_base:
            params_base["manifest"] = list(manifest_paths)
        reader = _create_sensor(sensor_type_upper, params_base)
        reader.extractImage()
        reader.frame.getImage().renderHdr()
        reader.extractDoppler()
        readers = {"frame": reader}
        print(reader.frame)
        bboxes = [stripmap_bbox(reader.frame)]
    
    elif sensor_type_upper == "SAOCOM_SLC":
        ''' TODO: add kwargs support 
        imgname = glob.glob(os.path.join(fname,'S1*/Data/slc*-vv'))[0]
        xmlname = glob.glob(os.path.join(fname,'S1*/Data/slc*-vv.xml'))[0]
        xemtname = glob.glob(os.path.join(fname,'S1*.xemt'))[0]

        obj = createSensor('SAOCOM_SLC')
        obj._imageFileName = imgname
        obj.xmlFile = xmlname
        obj.xemtFile = xemtname
        obj.output = os.path.join(outputdir, date+'.slc')
        '''
        pass
    
    else:
        #if not params_base:
        #    raise ValueError("sensor_kwargs must be provided for non-Sentinel sensors")

        if manifest_paths and "manifest" not in params_base:
            params_base["manifest"] = list(manifest_paths)
        params_base.setdefault("orbitFile", orbit_file)
        params_base.setdefault("orbitDir", orbit_dir)
        params_base.setdefault("auxDir", aux_dir)
        reader = _create_sensor(sensor_type_upper, params_base)
        readers = {"frame": reader}
        bboxes = [stripmap_bbox(reader.frame)]

    bbox = _union_bbox(bboxes)
    dem_path_resolved = _ensure_dem(
        bbox_snwe=bbox,
        dem_dir=dem_directory,
        source=dem_source,
        existing_dem=dem_path,
    )

    burst_range_tuple = tuple(burst_range) if burst_range else None

    results: Dict[str, Dict[str, object]] = {}
    for label, reader in readers.items():
        outdir = os.path.join(output_dir, label)
        results[label] = generate_shadow_layover(
            reader=reader,
            dem_path=dem_path_resolved,
            output_dir=outdir,
            burst_range=burst_range_tuple,
            dem_interp=dem_interp,
        )

    return results

# SENTINEL1
#generate_scene_shadow_masks(sensor_type="SENTINEL1", swaths=[1], burst_range=(1, 3))

# COSMO_SKYMED_SLC
csk = "/nas/products/485266a4-7b84-4b3e-be97-d447c7e190a7/CSKS1_SCS_B_HI_04_HH_RD_SF_20241121173220_20241121173226.h5"
generate_scene_shadow_masks(product=csk, sensor_type="COSMO_SKYMED_SLC")

# SAOCOM_SLC
#saocom = "/nas/products/48398db0-a4d1-4972-9cef-11a9f756fb88/EOL1ASARSAO1B10041305.zip"
