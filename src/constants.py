"""Canonical constants for SkyClearAI band ordering and channel configuration."""

# Optical bands (Sentinel-2 10m bands order in processed numpy arrays)
BAND_BLUE = 0
BAND_GREEN = 1
BAND_RED = 2
BAND_NIR = 3

# SAR bands (Sentinel-1 VV/VH order in processed numpy arrays)
BAND_VV = 0
BAND_VH = 1

# Stacked input channel indices for Model 1 (7 channels total)
CH_CLOUDY_B02 = 0  # Cloudy Blue
CH_CLOUDY_B03 = 1  # Cloudy Green
CH_CLOUDY_B04 = 2  # Cloudy Red
CH_CLOUDY_B08 = 3  # Cloudy NIR
CH_SAR_VV = 4      # SAR VV
CH_SAR_VH = 5      # SAR VH
CH_CLOUD_MASK = 6  # Cloud Mask
