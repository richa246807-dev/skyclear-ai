
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from contextlib import contextmanager
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds, Window
from rasterio.vrt import WarpedVRT

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure process logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@contextmanager
def open_dataset_vrt(url: str):
    """Context manager to open a raster dataset, wrapping it in a WarpedVRT if it has GCPs but no CRS."""
    src = rasterio.open(url)
    vrt = None
    try:
        # Check if dataset has GCPs but no CRS (typical for Sentinel-1 GRD raw TIFFs)
        if src.crs is None and src.gcps and len(src.gcps) > 1 and src.gcps[1]:
            vrt = WarpedVRT(src, src_crs=src.gcps[1])
            yield vrt
        else:
            yield src
    finally:
        if vrt is not None:
            vrt.close()
        src.close()


def get_asset_url(item, band_key: str) -> str | None:
    """Retrieve asset URL with fallback checks for key naming conventions."""
    assets = item.assets
    # Try direct key and lowercase key
    for k in (band_key, band_key.lower()):
        if k in assets:
            return assets[k].href

    # Common mappings
    mappings = {
        "B01": "coastal",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "rededge1",
        "B06": "rededge2",
        "B07": "rededge3",
        "B08": "nir",
        "B8A": "nir08",
        "B09": "nir09",
        "B10": "cirrus",
        "B11": "swir16",
        "B12": "swir22",
        "VV": "vv",
        "VH": "vh",
    }
    alt_key = mappings.get(band_key)
    if alt_key:
        for k in (alt_key, alt_key.lower(), alt_key.upper()):
            if k in assets:
                return assets[k].href

    return None


def crop_and_stack_remote_cogs(
    band_urls: list[str],
    bbox_wgs84: list[float],
    output_path: Path,
) -> None:
    """Crop a set of remote COG bands using WGS84 bbox, resample, and stack into a single GeoTIFF."""
    with rasterio.Env(aws_no_sign_request=True):
        first_url = band_urls[0]
        LOGGER.info("Defining target grid using band: %s", first_url)
        with open_dataset_vrt(first_url) as src:
            minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
            window = from_bounds(minx, miny, maxx, maxy, src.transform).round()
            col_off = max(0, min(src.width - 1, int(window.col_off)))
            row_off = max(0, min(src.height - 1, int(window.row_off)))
            width = max(1, min(src.width - col_off, int(window.width)))
            height = max(1, min(src.height - row_off, int(window.height)))
            window = Window(col_off, row_off, width, height)
            
            transform = rasterio.windows.transform(window, src.transform)
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": transform,
                "crs": src.crs,
                "count": len(band_urls),
                "dtype": "float32",
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing cropped and stacked raster to %s", output_path)
        with rasterio.open(output_path, "w", **profile) as dst:
            for idx, url in enumerate(band_urls):
                LOGGER.info("Cropping and resampling band %d/%d: %s", idx + 1, len(band_urls), url)
                with open_dataset_vrt(url) as src:
                    b_minx, b_miny, b_maxx, b_maxy = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
                    b_window = from_bounds(b_minx, b_miny, b_maxx, b_maxy, src.transform).round()
                    b_col_off = max(0, min(src.width - 1, int(b_window.col_off)))
                    b_row_off = max(0, min(src.height - 1, int(b_window.row_off)))
                    b_width = max(1, min(src.width - b_col_off, int(b_window.width)))
                    b_height = max(1, min(src.height - b_row_off, int(b_window.height)))
                    b_window = Window(b_col_off, b_row_off, b_width, b_height)
                    
                    band_data = src.read(
                        1,
                        window=b_window,
                        out_shape=(height, width),
                        resampling=rasterio.enums.Resampling.bilinear,
                    ).astype(np.float32)
                    dst.write(band_data, idx + 1)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Acquire real Sentinel-2 and Sentinel-1 crops over a target region.")
    parser.add_argument("--bbox", type=float, nargs=4, default=[11.5, 48.1, 11.6, 48.2],
                        help="WGS84 bounding box [min_lon, min_lat, max_lon, max_lat]")
    parser.add_argument("--date", default="2023-06-01/2023-08-31", help="Date range for S2 search")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Output raw data folder")
    args = parser.parse_args()

    try:
        from pystac_client import Client
    except ImportError:
        LOGGER.error("pystac-client is not installed. Please run setup first.")
        sys.exit(1)

    stac_url = "https://earth-search.aws.element84.com/v1"
    LOGGER.info("Connecting to Element 84 Earth Search STAC API: %s", stac_url)
    client = Client.open(stac_url)

    # 1. Search Sentinel-2 L2A scenes
    LOGGER.info("Searching sentinel-2-l2a items...")
    s2_search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=args.bbox,
        datetime=args.date,
    )
    s2_items = list(s2_search.items())
    if not s2_items:
        LOGGER.error("No Sentinel-2 items found for bbox %s and date %s", args.bbox, args.date)
        sys.exit(1)

    LOGGER.info("Found %d Sentinel-2 scenes. Filtering for clear and cloudy scenes...", len(s2_items))
    # Filter items by cloud cover
    clear_item = None
    cloudy_item = None
    
    # Sort items by cloud cover ascending
    sorted_s2 = sorted(s2_items, key=lambda x: x.properties.get("eo:cloud_cover", 100.0))
    
    for item in sorted_s2:
        cc = item.properties.get("eo:cloud_cover", 100.0)
        if cc < 5.0 and clear_item is None:
            clear_item = item
        if 20.0 <= cc <= 80.0 and cloudy_item is None:
            cloudy_item = item
        if clear_item and cloudy_item:
            break
            
    # Fallbacks if we can't satisfy strict thresholds
    if clear_item is None:
        clear_item = sorted_s2[0]
        LOGGER.warning("Could not find scene with <5%% cloud cover. Using lowest cloud cover: %.2f%%", 
                       clear_item.properties.get("eo:cloud_cover", 100.0))
    if cloudy_item is None:
        # Find something with >15% cloud cover
        cloudy_candidates = [item for item in sorted_s2 if item.properties.get("eo:cloud_cover", 0.0) > 15.0]
        if cloudy_candidates:
            cloudy_item = cloudy_candidates[0]
        else:
            cloudy_item = sorted_s2[-1]
        LOGGER.warning("Could not find scene with 20%%-80%% cloud cover. Using cloud cover: %.2f%%", 
                       cloudy_item.properties.get("eo:cloud_cover", 100.0))

    LOGGER.info("Selected Clear S2 Item ID: %s (Cloud Cover: %.2f%%)", clear_item.id, clear_item.properties.get("eo:cloud_cover", 100.0))
    LOGGER.info("Selected Cloudy S2 Item ID: %s (Cloud Cover: %.2f%%)", cloudy_item.id, cloudy_item.properties.get("eo:cloud_cover", 100.0))

    # 2. Search Sentinel-1 GRD scene close in time to cloudy scene (+/- 5 days)
    from datetime import timedelta
    cloudy_date = cloudy_item.datetime
    s1_start = (cloudy_date - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    s1_end = (cloudy_date + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOGGER.info("Searching sentinel-1-grd items between %s and %s...", s1_start, s1_end)
    
    s1_search = client.search(
        collections=["sentinel-1-grd"],
        bbox=args.bbox,
        datetime=f"{s1_start}/{s1_end}",
    )
    s1_items = list(s1_search.items())
    if not s1_items:
        # Retry with wider date range (+/- 15 days) if not found
        s1_start_wide = (cloudy_date - timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        s1_end_wide = (cloudy_date + timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        LOGGER.warning("No S1 items found in +/-5 days. Retrying with +/-15 days: %s to %s...", s1_start_wide, s1_end_wide)
        s1_search = client.search(
            collections=["sentinel-1-grd"],
            bbox=args.bbox,
            datetime=f"{s1_start_wide}/{s1_end_wide}",
        )
        s1_items = list(s1_search.items())
        
    if not s1_items:
        LOGGER.error("No Sentinel-1 items found.")
        sys.exit(1)

    s1_item = s1_items[0]
    LOGGER.info("Selected S1 Item ID: %s", s1_item.id)

    # 3. Crop and download Clear S2 (B02, B03, B04, B08)
    clear_bands = ["B02", "B03", "B04", "B08"]
    clear_urls = []
    for b in clear_bands:
        url = get_asset_url(clear_item, b)
        if not url:
            LOGGER.error("Could not find band %s in clear S2 item. Available keys: %s", b, list(clear_item.assets.keys()))
            sys.exit(1)
        clear_urls.append(url)
        
    crop_and_stack_remote_cogs(clear_urls, args.bbox, args.output_dir / "sentinel2_clear.tif")

    # 4. Crop and download Cloudy S2 (B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12)
    cloudy_bands = ["B01", "B02", "B03", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
    cloudy_urls = []
    for b in cloudy_bands:
        url = get_asset_url(cloudy_item, b)
        if not url:
            if b == "B10":
                # Fallback to B09 (nir09 or wvp) if B10 is not available in surface reflectance (L2A)
                url = get_asset_url(cloudy_item, "B09")
                if not url:
                    url = get_asset_url(cloudy_item, "wvp")
                if url:
                    LOGGER.warning("Band B10 (cirrus) not found in Sentinel-2 L2A item. Falling back to B09/wvp as placeholder.")
            if not url:
                LOGGER.error("Could not find band %s in cloudy S2 item. Available keys: %s", b, list(cloudy_item.assets.keys()))
                sys.exit(1)
        cloudy_urls.append(url)

    crop_and_stack_remote_cogs(cloudy_urls, args.bbox, args.output_dir / "sentinel2_cloudy_raw.tif")

    # 5. Crop and download S1 GRD (VV, VH)
    s1_bands = ["VV", "VH"]
    s1_urls = []
    for b in s1_bands:
        url = get_asset_url(s1_item, b)
        if not url:
            LOGGER.error("Could not find band %s in S1 item.", b)
            sys.exit(1)
        s1_urls.append(url)

    crop_and_stack_remote_cogs(s1_urls, args.bbox, args.output_dir / "sentinel1_grd_raw.tif")

    # 6. Align cloudy and S1 to clear
    from src.alignment import align_and_save
    
    align_and_save(
        args.output_dir / "sentinel2_cloudy_raw.tif",
        args.output_dir / "sentinel2_clear.tif",
        args.output_dir / "sentinel2_cloudy.tif"
    )
    align_and_save(
        args.output_dir / "sentinel1_grd_raw.tif",
        args.output_dir / "sentinel2_clear.tif",
        args.output_dir / "sentinel1_grd.tif"
    )

    LOGGER.info("Successfully completed real data acquisition and saved crops to %s", args.output_dir)


if __name__ == "__main__":
    main()
