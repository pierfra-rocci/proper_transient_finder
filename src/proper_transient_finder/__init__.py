# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Standalone PSF-matched astronomical difference imaging."""

import logging

from ._version import __version__
from .api import (
    BatchItemResult,
    BatchJob,
    BatchResult,
    PipelineResult,
    compare_batch,
    compare_images,
    load_batch_manifest,
)
from .astro import PipelineConfig, ProperTransientFinder, compare_cross_filter, main
from .survey import (
    SurveyTemplate,
    TemplateRequest,
    compare_survey_batch,
    compare_survey_images,
    fetch_survey_template,
    load_science_list,
    prepare_template_request,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ProperTransientFinder",
    "BatchItemResult",
    "BatchJob",
    "BatchResult",
    "PipelineConfig",
    "PipelineResult",
    "SurveyTemplate",
    "TemplateRequest",
    "__version__",
    "compare_batch",
    "compare_cross_filter",
    "compare_images",
    "compare_survey_batch",
    "compare_survey_images",
    "fetch_survey_template",
    "load_batch_manifest",
    "load_science_list",
    "main",
    "prepare_template_request",
]
