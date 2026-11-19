# Copyright (c) Alibaba, Inc. and its affiliates.
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
from typing import Literal, Sequence

from PIL import Image
from pydantic import BaseModel, RootModel

from evalscope.api.judge import CaseVerdict, JudgeContext, OutputContract, ReducedVerdict
from evalscope.utils.io_utils import PIL_to_base64
from evalscope.utils.uri_utils import file_as_data

from .checklists import DIM_TO_CHECKLIST

SCORE_NAME = 'qwen_image_bench_score'
SCORE_MAP = {0: 0.0, 1: 60.0, 2: 100.0}


def prepare_judge_image(image_path: str, artifact_dir: Path | None) -> str:
    """Apply the official preprocessing and keep one PNG for all judge requests on this image."""
    image_bytes, _ = file_as_data(image_path)
    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert('RGB')
        if max(image.size) > 1024:
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    if artifact_dir is None:
        return PIL_to_base64(image, format='PNG', add_header=True)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / f'{sha256(image_bytes).hexdigest()}.png'
    if not target.is_file():
        # Publish only complete PNGs when repeated images are reviewed concurrently.
        with NamedTemporaryFile(dir=artifact_dir, delete=False) as output:
            pending = Path(output.name)
        try:
            image.save(pending, format='PNG')
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
    return str(target)


class FacetGrade(BaseModel):
    score: Literal[0, 1, 2, 'N/A']


class DimensionGrade(RootModel[dict[str, FacetGrade | dict[str, FacetGrade]]]):
    """The official nested rubric verdict and the flat variant accepted by the upstream scorer."""


GRADE_CONTRACT = OutputContract(schema_model=DimensionGrade, allow_json_fences=True)


def _facet_parents(checklist: str) -> dict[str, str]:
    parents = {}
    for line in checklist.splitlines():
        if line.startswith('## '):
            parent = line[3:]
        elif line.startswith('- '):
            facet = line[2:].split(':', 1)[0]
            parents[facet] = parent
    return parents


FACET_PARENTS = {dimension: _facet_parents(checklist) for dimension, checklist in DIM_TO_CHECKLIST.items()}


def _subdimension_scores(dimension: str, grade: DimensionGrade) -> dict[str, float]:
    """Repair known facet placement before aggregating non-N/A grades."""
    grouped: dict[str, dict[str, FacetGrade]] = {}
    parents = FACET_PARENTS[dimension]
    for label, value in grade.root.items():
        facets = {label: value} if isinstance(value, FacetGrade) else value
        for facet, facet_grade in facets.items():
            if dimension == 'Creative Generation' and facet == 'Feature Mapping':
                facet = 'Feature Matching'
            grouped.setdefault(parents.get(facet, label), {})[facet] = facet_grade

    scores = {}
    for subdimension, facets in grouped.items():
        applicable = [SCORE_MAP[facet.score] for facet in facets.values() if facet.score in SCORE_MAP]
        if applicable:
            scores[subdimension] = mean(applicable)
    return scores


def reduce_grades(verdicts: Sequence[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
    """Average subdimensions within dimensions and scored dimensions within the image."""
    values = {}
    dimension_scores = []
    for verdict in verdicts:
        dimension = verdict.metadata['dimension']
        subdimensions = _subdimension_scores(dimension, verdict.value)
        if not subdimensions:
            continue
        dimension_score = mean(subdimensions.values())
        dimension_scores.append(dimension_score)
        values[f'{SCORE_NAME}:{dimension}'] = dimension_score
        for label, score in subdimensions.items():
            values[f'{SCORE_NAME}:{dimension}/{label}'] = score
    if dimension_scores:
        values[SCORE_NAME] = mean(dimension_scores)
    return ReducedVerdict(value=values)
