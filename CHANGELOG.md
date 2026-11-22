# Changelog

## 0.3.4

- Added provider-native Pan-STARRS and SkyMapper FITS cutout fallbacks after CDS HiPS outages.
- Restricted provider fallbacks to named providers and supported bands.
- Provider fallbacks are now usable by the normal subtraction pipeline and remain experimentally calibrated.
- Removed the NASA SkyView and DSS2R fallback implementation.

## 0.3.3

- Added Pan-STARRS and SkyMapper named-provider HiPS selection.
- Preserved explicit HiPS identifiers as non-substituted requests.

## 0.3.2

- Added HiPS survey-reference acquisition and cache provenance.
- Added Pan-STARRS and SkyMapper HiPS provider selection.
- Added provider-native acquisition-only fallback reporting.
