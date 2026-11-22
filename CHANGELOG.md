# Changelog

## 0.3.3

- Restricted the NASA SkyView outage fallback to R-band named-provider requests.
- Added the official DSS2R SkyView survey identifier.
- Kept DSS2R acquisition-only and prohibited it from photometric subtraction.
- Removed G-, I-, near-IR, and other DSS/DSS2 fallback substitutions.
- Preserved explicit HiPS identifiers as non-substituted requests.

## 0.3.2

- Added HiPS survey-reference acquisition and cache provenance.
- Added Pan-STARRS and SkyMapper HiPS provider selection.
- Added acquisition-only SkyView fallback reporting.
