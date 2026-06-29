from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

_XML_NS = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
_RELS_NS = {'rel': 'http://schemas.openxmlformats.org/package/2006/relationships'}
_OFFICE_RELS_NS = {'rel': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
_XLSX_EXTENSIONS = {'.xlsx', '.xlsm', '.xls'}


@dataclass
class RubricItem:
    criterion: str
    score: float
    rubric_item_id: str = ''
    tags: Optional[List[str]] = None


@dataclass
class RubricItemResult:
    criterion: str
    score: float
    criteria_met: Optional[bool]
    awarded_points: float
    explanation: str
    rubric_item_id: str = ''

    @property
    def evaluated(self) -> bool:
        return self.criteria_met is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rubric_item_id': self.rubric_item_id,
            'criterion': self.criterion,
            'score': self.score,
            'criteria_met': self.criteria_met,
            'awarded_points': self.awarded_points,
            'explanation': self.explanation,
            'evaluated': self.evaluated,
        }


@dataclass
class LocalRubricScore:
    score: float
    score_all_items: float
    coverage: float
    achieved_points: float
    possible_points: float
    all_possible_points: float
    evaluated_items: int
    total_items: int
    item_results: List[RubricItemResult]

    def summary(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'score_all_items': self.score_all_items,
            'coverage': self.coverage,
            'achieved_points': self.achieved_points,
            'possible_points': self.possible_points,
            'all_possible_points': self.all_possible_points,
            'evaluated_items': self.evaluated_items,
            'total_items': self.total_items,
        }


@dataclass
class XlsxSheet:
    name: str
    rows: List[List[Any]]

    @property
    def headers(self) -> List[str]:
        if not self.rows:
            return []
        return [_normalize_header(value) for value in self.rows[0]]

    @property
    def data_rows(self) -> List[List[Any]]:
        return self.rows[1:]

    def column_index(self, candidates: List[str]) -> Optional[int]:
        normalized_candidates = {_normalize_text(candidate) for candidate in candidates}
        for idx, header in enumerate(self.headers):
            if _normalize_text(header) in normalized_candidates:
                return idx
        return None


@dataclass
class XlsxWorkbook:
    path: Path
    sheets: List[XlsxSheet]

    @property
    def sheet_names(self) -> List[str]:
        return [sheet.name for sheet in self.sheets]

    @property
    def first_sheet(self) -> Optional[XlsxSheet]:
        return self.sheets[0] if self.sheets else None

    def get_sheet(self, name: str) -> Optional[XlsxSheet]:
        target = _normalize_text(name)
        for sheet in self.sheets:
            if _normalize_text(sheet.name) == target:
                return sheet
        return None


class GDPvalLocalScorer:
    """EvalScope-owned approximate scorer for locally checkable GDPval rubric items."""

    score_name = 'local_rubric_score'

    def __init__(self, metadata: Dict[str, Any], prediction: str) -> None:
        self.metadata = metadata
        self.prediction = prediction
        self.deliverables = _deliverable_paths(metadata)
        self.sample_workbook = self._find_workbook('Sample')
        self.reference_workbook = self._load_reference_workbook()
        self._workbook_cache: Dict[Path, Optional[XlsxWorkbook]] = {}

    def score(self) -> LocalRubricScore:
        rubric_items = parse_rubric_items(self.metadata.get('rubric_json'))
        item_results = [self._score_item(item) for item in rubric_items]
        evaluated_results = [result for result in item_results if result.evaluated]
        all_positive_points = sum(item.score for item in rubric_items if item.score > 0)
        possible_points = sum(result.score for result in evaluated_results if result.score > 0)
        achieved_points = sum(result.awarded_points for result in evaluated_results)
        score = achieved_points / possible_points if possible_points else 0.0
        score_all_items = achieved_points / all_positive_points if all_positive_points else 0.0
        coverage = possible_points / all_positive_points if all_positive_points else 0.0
        return LocalRubricScore(
            score=score,
            score_all_items=score_all_items,
            coverage=coverage,
            achieved_points=achieved_points,
            possible_points=possible_points,
            all_possible_points=all_positive_points,
            evaluated_items=len(evaluated_results),
            total_items=len(rubric_items),
            item_results=item_results,
        )

    def _score_item(self, item: RubricItem) -> RubricItemResult:
        criterion = item.criterion
        normalized = _normalize_text(criterion)
        checks = [
            self._check_sample_workbook_file,
            self._check_named_sheet,
            self._check_sample_size_values,
            self._check_population_size,
            self._check_sampling_formula,
            self._check_first_sheet_headers,
            self._check_first_sheet_rows_match_reference,
            self._check_reference_column_positions,
            self._check_variance_column,
            self._check_zero_variance,
            self._check_division_by_zero_convention,
            self._check_no_excel_errors,
            self._check_sample_column_exists,
            self._check_non_sample_values,
            self._check_sample_count,
            self._check_variance_threshold,
            self._check_specific_row_combo,
            self._check_metric_present,
            self._check_conditional_reference_value_present,
            self._check_all_values_covered,
            self._check_first_sheet_name,
        ]
        for check in checks:
            result = check(normalized)
            if result is not None:
                criteria_met, explanation = result
                return _result(item, criteria_met, explanation)
        return _result(item, None, 'Not evaluated by EvalScope local deterministic scorer.')

    def _sample_book(self) -> Optional[XlsxWorkbook]:
        return self._workbook(self.sample_workbook) if self.sample_workbook else None

    def _reference_book(self) -> Optional[XlsxWorkbook]:
        return self.reference_workbook

    def _workbook(self, path: Optional[Path]) -> Optional[XlsxWorkbook]:
        if path is None:
            return None
        if path not in self._workbook_cache:
            self._workbook_cache[path] = read_xlsx_workbook(path)
        return self._workbook_cache[path]

    def _find_workbook(self, stem: str) -> Optional[Path]:
        target = _normalize_text(stem)
        for path in self.deliverables:
            if path.suffix.lower() in _XLSX_EXTENSIONS and _normalize_text(path.stem) == target:
                return path
        return None

    def _load_reference_workbook(self) -> Optional[XlsxWorkbook]:
        for host_path in self.metadata.get('host_reference_files') or []:
            path = Path(str(host_path))
            if path.suffix.lower() in _XLSX_EXTENSIONS and path.is_file():
                workbook = read_xlsx_workbook(path)
                if workbook is not None:
                    return workbook
        return None

    def _first_sample_sheet(self) -> Optional[XlsxSheet]:
        workbook = self._sample_book()
        return workbook.first_sheet if workbook else None

    def _calc_sheet(self) -> Optional[XlsxSheet]:
        workbook = self._sample_book()
        return workbook.get_sheet('Sample Size Calculation') if workbook else None

    def _reference_sheet(self) -> Optional[XlsxSheet]:
        workbook = self._reference_book()
        return workbook.first_sheet if workbook else None

    def _check_sample_workbook_file(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if "basename is 'sample'" not in criterion and 'basename is "sample"' not in criterion:
            return None
        if self.sample_workbook is not None:
            return True, f'Found Sample workbook: {self.sample_workbook.name}.'
        return False, 'No Excel workbook named Sample was found in deliverable_files.'

    def _check_named_sheet(self, criterion: str) -> Optional[Tuple[bool, str]]:
        match = re.search(r"worksheet named exactly '([^']+)'", criterion)
        if not match:
            return None
        expected = match.group(1)
        workbook = self._sample_book()
        found = workbook.get_sheet(expected) is not None if workbook else False
        return found, f'Worksheet {expected!r} {"found" if found else "not found"} in Sample workbook.'

    def _check_sample_size_values(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'confidence level of 90%' not in criterion or 'tolerable error' not in criterion:
            return None
        text = self._sheet_text(self._calc_sheet())
        found = ('90%' in text or '0.9' in text) and ('10%' in text or '0.1' in text)
        return found, 'Sample Size Calculation sheet contains 90% confidence and 10% tolerable error.' if found else (
            'Sample Size Calculation sheet does not clearly contain both 90% confidence and 10% tolerable error.'
        )

    def _check_population_size(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'population size n' not in criterion:
            return None
        reference_rows = len(self._reference_sheet().data_rows) if self._reference_sheet() else None
        text = self._sheet_text(self._calc_sheet())
        found = reference_rows is not None and str(reference_rows) in text
        return found, f'Population size {reference_rows} {"found" if found else "not found"} in calculation sheet.'

    def _check_sampling_formula(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'z = 1.645' not in criterion or 'finite population correction' not in criterion:
            return None
        text = self._sheet_text(self._calc_sheet())
        required_bits = ['1.645', '0.5', '0.10']
        has_formula_bits = all(bit in text
                               for bit in required_bits) or all(bit.rstrip('0') in text for bit in required_bits)
        has_fpc = 'finite' in text or 'fpc' in text or 'population correction' in text
        has_integer = any(
            _is_integer_like(value) for row in (self._calc_sheet().rows if self._calc_sheet() else []) for value in row
        )
        found = has_formula_bits and has_fpc and has_integer
        return found, 'Sampling formula includes z, p, e, FPC, and an integer sample size.' if found else (
            'Sampling formula is missing z/p/e, finite population correction, or integer sample size.'
        )

    def _check_first_sheet_headers(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'preserving columns a-h' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        reference_sheet = self._reference_sheet()
        if not sample_sheet or not reference_sheet:
            return False, 'Sample or reference workbook is unavailable.'
        found = sample_sheet.headers[:8] == reference_sheet.headers[:8]
        return found, 'First eight headers match the reference workbook.' if found else (
            f'First eight headers differ: {sample_sheet.headers[:8]} vs {reference_sheet.headers[:8]}.'
        )

    def _check_first_sheet_rows_match_reference(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'values in columns a' not in criterion or 'exactly match' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        reference_sheet = self._reference_sheet()
        if not sample_sheet or not reference_sheet:
            return False, 'Sample or reference workbook is unavailable.'
        mismatches = _count_reference_mismatches(sample_sheet, reference_sheet, 8)
        found = mismatches == 0
        return found, 'All sampled rows match reference columns A-H.' if found else (
            f'{mismatches} sampled rows do not match reference columns A-H.'
        )

    def _check_reference_column_positions(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'columns g and h' not in criterion or 'consistent with the population reference' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        reference_sheet = self._reference_sheet()
        if not sample_sheet or not reference_sheet:
            return False, 'Sample or reference workbook is unavailable.'
        found = sample_sheet.headers[6:8] == reference_sheet.headers[6:8]
        return found, f'Columns G-H are {sample_sheet.headers[6:8]} and match the reference.' if found else (
            f'Columns G-H do not match reference: {sample_sheet.headers[6:8]} vs {reference_sheet.headers[6:8]}.'
        )

    def _check_variance_column(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'computes quarter' not in criterion and 'variance as' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        if not sample_sheet:
            return False, 'Sample workbook is unavailable.'
        variance_idx = _variance_index(sample_sheet)
        if variance_idx is None:
            return False, 'No variance column was found.'
        checked, mismatches = _check_variance_values(sample_sheet, variance_idx)
        found = checked > 0 and mismatches == 0
        return found, f'Checked {checked} non-zero Q2 variance values with {mismatches} mismatches.'

    def _check_zero_variance(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'q2 = 0 and q3 = 0' not in criterion or 'records 0' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        variance_idx = _variance_index(sample_sheet) if sample_sheet else None
        if sample_sheet is None or variance_idx is None:
            return False, 'Sample workbook or variance column is unavailable.'
        checked, mismatches = _check_zero_zero_variance(sample_sheet, variance_idx)
        found = checked == 0 or mismatches == 0
        return found, f'Checked {checked} Q2=0/Q3=0 rows with {mismatches} non-zero variance values.'

    def _check_division_by_zero_convention(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'q2 = 0 and q3' not in criterion or 'non-numeric convention' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        variance_idx = _variance_index(sample_sheet) if sample_sheet else None
        if sample_sheet is None or variance_idx is None:
            return False, 'Sample workbook or variance column is unavailable.'
        checked, mismatches = _check_zero_nonzero_variance(sample_sheet, variance_idx)
        found = checked == 0 or mismatches == 0
        return found, f'Checked {checked} Q2=0/Q3!=0 rows with {mismatches} invalid conventions.'

    def _check_no_excel_errors(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'no cells in column i' not in criterion or 'excel errors' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        variance_idx = _variance_index(sample_sheet) if sample_sheet else None
        if sample_sheet is None or variance_idx is None:
            return False, 'Sample workbook or variance column is unavailable.'
        errors = sum(1 for row in sample_sheet.data_rows if _is_excel_error(_get(row, variance_idx)))
        return errors == 0, f'Found {errors} Excel error values in the variance column.'

    def _check_sample_column_exists(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'sampled rows are flagged' not in criterion and 'column j exists' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        sample_idx = _sample_index(sample_sheet) if sample_sheet else None
        found = sample_idx is not None and _sample_count(sample_sheet, sample_idx) > 0
        return found, 'Sample indicator column exists and contains at least one 1.' if found else (
            'Sample indicator column is missing or has no sampled rows.'
        )

    def _check_non_sample_values(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'non' not in criterion or 'sampled rows' not in criterion or 'left blank or set to 0' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        sample_idx = _sample_index(sample_sheet) if sample_sheet else None
        if sample_sheet is None or sample_idx is None:
            return False, 'Sample indicator column is unavailable.'
        invalid = sum(
            1 for row in sample_sheet.data_rows
            if not _is_sampled(row, sample_idx) and _get(row, sample_idx) not in ('', None, 0, '0')
        )
        return invalid == 0, f'Found {invalid} non-sampled indicator values that are not blank or 0.'

    def _check_sample_count(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'sum of 1s' not in criterion and 'sample count' not in criterion:
            return None
        sample_sheet = self._first_sample_sheet()
        sample_idx = _sample_index(sample_sheet) if sample_sheet else None
        if sample_sheet is None or sample_idx is None:
            return False, 'Sample indicator column is unavailable.'
        count = _sample_count(sample_sheet, sample_idx)
        required = _required_sample_size(self._calc_sheet())
        found = count > 0 and (required is None or count >= required)
        return found, f'Sample count is {count}; required sample size is {required}.'

    def _check_variance_threshold(self, criterion: str) -> Optional[Tuple[bool, str]]:
        match = re.search(r'absolute variance.*?(?:>=?|≥)\s*(\d+)%', criterion)
        if not match:
            return None
        threshold = float(match.group(1))
        sample_sheet = self._first_sample_sheet()
        variance_idx = _variance_index(sample_sheet) if sample_sheet else None
        sample_idx = _sample_index(sample_sheet) if sample_sheet else None
        if sample_sheet is None or variance_idx is None or sample_idx is None:
            return False, 'Sample, variance, or sample indicator column is unavailable.'
        hits = _count_sampled_variance_hits(sample_sheet, variance_idx, sample_idx, threshold)
        return hits > 0, f'Found {hits} sampled rows with absolute variance >= {threshold}%.'

    def _check_specific_row_combo(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'first tab' not in criterion and 'first worksheet' not in criterion:
            return None
        combo = _extract_combo_criterion(criterion)
        if combo is None:
            return None
        sample_sheet = self._first_sample_sheet()
        if sample_sheet is None:
            return False, 'Sample workbook is unavailable.'
        count = _count_matching_rows(sample_sheet, combo)
        return count > 0, f'Found {count} sampled rows matching {combo}.'

    def _check_metric_present(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'where the metric is' not in criterion:
            return None
        match = re.search(r'metric is ([a-z0-9 \-/]+)', criterion)
        if not match:
            return None
        metric = match.group(1).strip()
        sample_sheet = self._first_sample_sheet()
        if sample_sheet is None:
            return False, 'Sample workbook is unavailable.'
        count = _count_column_value(sample_sheet, 'KRIs', metric)
        return count > 0, f'Found {count} sampled rows where metric is {metric!r}.'

    def _check_conditional_reference_value_present(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'if any rows have q2 = 0 and q3 = 0' in criterion:
            sample_sheet = self._first_sample_sheet()
            if sample_sheet is None:
                return False, 'Sample workbook is unavailable.'
            count = _count_zero_zero_rows(sample_sheet)
            return count > 0, f'Found {count} selected rows with Q2=0 and Q3=0.'
        match = re.search(r"if '([^']+)' (?:appears|occurs)", criterion)
        if not match:
            return None
        value = match.group(1)
        sample_sheet = self._first_sample_sheet()
        if sample_sheet is None:
            return False, 'Sample workbook is unavailable.'
        count = _count_value_anywhere(sample_sheet, value)
        return count > 0, f'Found {count} selected rows containing {value!r}.'

    def _check_all_values_covered(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'for each distinct division' in criterion:
            return self._check_distinct_values_covered('Division')
        if 'for each distinct sub division' in criterion or 'for each distinct sub-division' in criterion:
            return self._check_distinct_values_covered('Sub-Division')
        return None

    def _check_distinct_values_covered(self, column: str) -> Tuple[bool, str]:
        sample_sheet = self._first_sample_sheet()
        reference_sheet = self._reference_sheet()
        if not sample_sheet or not reference_sheet:
            return False, 'Sample or reference workbook is unavailable.'
        sample_values = _column_values(sample_sheet, column)
        reference_values = _column_values(reference_sheet, column)
        missing = sorted(reference_values - sample_values)
        return not missing, f'Missing {column} values: {missing}.' if missing else f'All {column} values are covered.'

    def _check_first_sheet_name(self, criterion: str) -> Optional[Tuple[bool, str]]:
        if 'first worksheet is named' not in criterion and 'first worksheet is named' not in criterion:
            return None
        match = re.search(r"named '([^']+)'", criterion)
        expected = match.group(1) if match else 'Sample'
        workbook = self._sample_book()
        actual = workbook.first_sheet.name if workbook and workbook.first_sheet else ''
        found = _normalize_text(actual) == _normalize_text(expected)
        return found, f'First worksheet is named {actual!r}; expected {expected!r}.'

    @staticmethod
    def _sheet_text(sheet: Optional[XlsxSheet]) -> str:
        if sheet is None:
            return ''
        return ' '.join(str(value) for row in sheet.rows for value in row if value is not None).lower()


def parse_rubric_items(raw_rubric: Any) -> List[RubricItem]:
    if not raw_rubric:
        return []
    if isinstance(raw_rubric, str):
        try:
            raw_items = json.loads(raw_rubric)
        except json.JSONDecodeError:
            return []
    else:
        raw_items = raw_rubric
    if not isinstance(raw_items, list):
        return []
    items = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        criterion = str(raw_item.get('criterion') or '').strip()
        if not criterion:
            continue
        items.append(
            RubricItem(
                criterion=criterion,
                score=float(raw_item.get('score') or raw_item.get('points') or 0.0),
                rubric_item_id=str(raw_item.get('rubric_item_id') or ''),
                tags=[str(tag) for tag in raw_item.get('tags') or []],
            )
        )
    return items


def read_xlsx_workbook(path: Path) -> Optional[XlsxWorkbook]:
    if not path.is_file():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _read_shared_strings(archive)
            sheet_refs = _read_sheet_refs(archive)
            sheets = []
            for name, target in sheet_refs:
                rows = _read_sheet_rows(archive, target, shared_strings)
                sheets.append(XlsxSheet(name=name, rows=rows))
            return XlsxWorkbook(path=path, sheets=sheets)
    except Exception:
        return None


def _read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        root = ElementTree.fromstring(archive.read('xl/sharedStrings.xml'))
    except KeyError:
        return []
    values = []
    for item in root.findall('main:si', _XML_NS):
        text_parts = [node.text or '' for node in item.findall('.//main:t', _XML_NS)]
        values.append(''.join(text_parts))
    return values


def _read_sheet_refs(archive: zipfile.ZipFile) -> List[Tuple[str, str]]:
    workbook = ElementTree.fromstring(archive.read('xl/workbook.xml'))
    rels = ElementTree.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    rel_map = {
        rel.attrib['Id']: rel.attrib['Target']
        for rel in rels.findall('rel:Relationship', _RELS_NS)
        if 'Id' in rel.attrib and 'Target' in rel.attrib
    }
    refs = []
    for sheet in workbook.findall('main:sheets/main:sheet', _XML_NS):
        name = sheet.attrib.get('name') or ''
        rel_id = sheet.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        target = rel_map.get(rel_id or '', '')
        if target:
            refs.append((name, f'xl/{target.lstrip("/")}'))
    return refs


def _read_sheet_rows(archive: zipfile.ZipFile, target: str, shared_strings: List[str]) -> List[List[Any]]:
    root = ElementTree.fromstring(archive.read(target))
    rows = []
    for row in root.findall('.//main:sheetData/main:row', _XML_NS):
        values: Dict[int, Any] = {}
        max_idx = -1
        for cell in row.findall('main:c', _XML_NS):
            ref = cell.attrib.get('r', '')
            idx = _cell_column_index(ref)
            max_idx = max(max_idx, idx)
            values[idx] = _cell_value(cell, shared_strings)
        if max_idx >= 0:
            rows.append([values.get(idx, '') for idx in range(max_idx + 1)])
    return rows


def _cell_value(cell: ElementTree.Element, shared_strings: List[str]) -> Any:
    cell_type = cell.attrib.get('t')
    if cell_type == 'inlineStr':
        texts = [node.text or '' for node in cell.findall('.//main:t', _XML_NS)]
        return ''.join(texts)
    value_node = cell.find('main:v', _XML_NS)
    raw_value = (value_node.text or '') if value_node is not None else ''
    if cell_type == 's':
        try:
            return shared_strings[int(raw_value)]
        except (ValueError, IndexError):
            return raw_value
    if cell_type in {'str', 'e'}:
        return raw_value
    return _to_number(raw_value)


def _cell_column_index(ref: str) -> int:
    letters = ''.join(ch for ch in ref if ch.isalpha()).upper()
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord('A') + 1
    return max(value - 1, 0)


def _to_number(value: str) -> Any:
    if value == '':
        return ''
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _deliverable_paths(metadata: Dict[str, Any]) -> List[Path]:
    paths = []
    for deliverable in metadata.get('deliverable_files') or []:
        path = Path(str(deliverable.get('local_path') or ''))
        if path.is_file():
            paths.append(path)
    return paths


def _result(item: RubricItem, criteria_met: Optional[bool], explanation: str) -> RubricItemResult:
    awarded_points = item.score if criteria_met else 0.0
    return RubricItemResult(
        criterion=item.criterion,
        score=item.score,
        criteria_met=criteria_met,
        awarded_points=awarded_points,
        explanation=explanation,
        rubric_item_id=item.rubric_item_id,
    )


def _normalize_header(value: Any) -> str:
    return str(value).strip()


def _normalize_text(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value).strip().lower().replace('‑', '-').replace('–', '-'))


def _get(row: List[Any], idx: Optional[int]) -> Any:
    if idx is None or idx >= len(row):
        return ''
    return row[idx]


def _is_integer_like(value: Any) -> bool:
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    return str(value).strip().isdigit()


def _numeric(value: Any) -> Optional[float]:
    if value in ('', None):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isinf(value):
            return value
        return float(value)
    try:
        return float(str(value).strip().replace('%', ''))
    except ValueError:
        return None


def _variance_index(sheet: Optional[XlsxSheet]) -> Optional[int]:
    if sheet is None:
        return None
    return sheet.column_index(['Variance %', 'Variance (%)', '% Var Q3 vs Q2', 'Variance'])


def _sample_index(sheet: Optional[XlsxSheet]) -> Optional[int]:
    if sheet is None:
        return None
    return sheet.column_index(['Sample', 'Sample Indicator', 'Selected'])


def _q2_q3_indices(sheet: XlsxSheet) -> Tuple[Optional[int], Optional[int]]:
    q2_idx = sheet.column_index(['Q2 2024 KRI', 'Q2'])
    q3_idx = sheet.column_index(['Q3 2024 KRI', 'Q3'])
    return q2_idx, q3_idx


def _check_variance_values(sheet: XlsxSheet, variance_idx: int) -> Tuple[int, int]:
    q2_idx, q3_idx = _q2_q3_indices(sheet)
    if q2_idx is None or q3_idx is None:
        return 0, 0
    checked = 0
    mismatches = 0
    for row in sheet.data_rows:
        q2 = _numeric(_get(row, q2_idx))
        q3 = _numeric(_get(row, q3_idx))
        observed = _numeric(_get(row, variance_idx))
        if q2 in (None, 0) or q3 is None or observed is None:
            continue
        expected = (q3 - q2) / q2
        checked += 1
        if not (_close(observed, expected) or _close(observed, expected * 100)):
            mismatches += 1
    return checked, mismatches


def _check_zero_zero_variance(sheet: XlsxSheet, variance_idx: int) -> Tuple[int, int]:
    q2_idx, q3_idx = _q2_q3_indices(sheet)
    if q2_idx is None or q3_idx is None:
        return 0, 0
    checked = 0
    mismatches = 0
    for row in sheet.data_rows:
        q2 = _numeric(_get(row, q2_idx))
        q3 = _numeric(_get(row, q3_idx))
        if q2 == 0 and q3 == 0:
            checked += 1
            value = _numeric(_get(row, variance_idx))
            if value not in (0, 0.0):
                mismatches += 1
    return checked, mismatches


def _check_zero_nonzero_variance(sheet: XlsxSheet, variance_idx: int) -> Tuple[int, int]:
    q2_idx, q3_idx = _q2_q3_indices(sheet)
    if q2_idx is None or q3_idx is None:
        return 0, 0
    checked = 0
    mismatches = 0
    for row in sheet.data_rows:
        q2 = _numeric(_get(row, q2_idx))
        q3 = _numeric(_get(row, q3_idx))
        if q2 == 0 and q3 not in (None, 0):
            checked += 1
            value = _get(row, variance_idx)
            if _numeric(value) is not None or _is_excel_error(value):
                mismatches += 1
    return checked, mismatches


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-4, abs_tol=1e-4)


def _is_excel_error(value: Any) -> bool:
    return str(value).strip().upper() in {'#DIV/0!', '#VALUE!', '#N/A', '#REF!', '#NAME?', '#NUM!', '#NULL!'}


def _sample_count(sheet: XlsxSheet, sample_idx: int) -> int:
    return sum(1 for row in sheet.data_rows if _is_sampled(row, sample_idx))


def _is_sampled(row: List[Any], sample_idx: int) -> bool:
    value = _get(row, sample_idx)
    return value == 1 or str(value).strip() == '1'


def _required_sample_size(sheet: Optional[XlsxSheet]) -> Optional[int]:
    if sheet is None:
        return None

    preferred = []
    candidates = []
    sample_size_columns = [idx for idx, header in enumerate(sheet.headers) if _is_sample_size_label(header)]
    for row in sheet.data_rows:
        for idx in sample_size_columns:
            value = _positive_integer(_get(row, idx))
            if value is not None:
                candidates.append(value)
        for idx, cell in enumerate(row):
            if not _is_sample_size_label(cell):
                continue
            row_values = row[idx + 1:] or row
            row_candidates = [value for value in (_positive_integer(item) for item in row_values) if value is not None]
            if _is_preferred_sample_size_label(cell):
                preferred.extend(row_candidates)
            else:
                candidates.extend(row_candidates)
    if preferred:
        return preferred[-1]
    return candidates[-1] if candidates else None


def _is_sample_size_label(value: Any) -> bool:
    text = _normalize_text(value)
    if not text or 'population size' in text:
        return False
    return 'sample size' in text or 'sample count' in text or 'samples required' in text


def _is_preferred_sample_size_label(value: Any) -> bool:
    text = _normalize_text(value)
    return any(token in text for token in ('final', 'required', 'minimum', 'rounded'))


def _positive_integer(value: Any) -> Optional[int]:
    if not _is_integer_like(value):
        return None
    parsed = int(float(value))
    return parsed if parsed > 0 else None


def _count_sampled_variance_hits(sheet: XlsxSheet, variance_idx: int, sample_idx: int, threshold: float) -> int:
    hits = 0
    for row in sheet.data_rows:
        value = _numeric(_get(row, variance_idx))
        if value is None:
            continue
        if abs(value) >= threshold and _is_sampled(row, sample_idx):
            hits += 1
    return hits


def _extract_combo_criterion(criterion: str) -> Optional[Dict[str, str]]:
    combo: Dict[str, str] = {}
    division_match = re.search(r'division is ([a-z ]+),', criterion)
    if division_match:
        combo['Division'] = division_match.group(1).strip()
    sub_match = re.search(r'sub-division is ([a-z &-]+?)(?:,| and )', criterion)
    if sub_match:
        combo['Sub-Division'] = sub_match.group(1).strip()
    country_match = re.search(r'country is ([a-z ]+)', criterion)
    if country_match:
        combo['Country'] = country_match.group(1).strip()
    return combo or None


def _count_matching_rows(sheet: XlsxSheet, combo: Dict[str, str]) -> int:
    header_map = {_normalize_text(header): idx for idx, header in enumerate(sheet.headers)}
    count = 0
    for row in sheet.data_rows:
        matched = True
        for column, expected in combo.items():
            idx = header_map.get(_normalize_text(column))
            if idx is None or not _values_match(_get(row, idx), expected):
                matched = False
                break
        if matched:
            count += 1
    return count


def _values_match(actual: Any, expected: str) -> bool:
    left = _normalize_text(actual).replace('banking', 'bank')
    right = _normalize_text(expected).replace('banking', 'bank')
    return left == right


def _count_column_value(sheet: XlsxSheet, column: str, expected: str) -> int:
    idx = sheet.column_index([column])
    if idx is None:
        return 0
    return sum(1 for row in sheet.data_rows if _normalize_text(_get(row, idx)) == _normalize_text(expected))


def _count_zero_zero_rows(sheet: XlsxSheet) -> int:
    q2_idx, q3_idx = _q2_q3_indices(sheet)
    return sum(1 for row in sheet.data_rows if _numeric(_get(row, q2_idx)) == 0 and _numeric(_get(row, q3_idx)) == 0)


def _count_value_anywhere(sheet: XlsxSheet, value: str) -> int:
    expected = _normalize_text(value)
    return sum(1 for row in sheet.data_rows if any(_normalize_text(cell) == expected for cell in row))


def _column_values(sheet: XlsxSheet, column: str) -> set[str]:
    idx = sheet.column_index([column])
    if idx is None:
        return set()
    return {_normalize_header(_get(row, idx)) for row in sheet.data_rows if _normalize_header(_get(row, idx))}


def _count_reference_mismatches(sample_sheet: XlsxSheet, reference_sheet: XlsxSheet, width: int) -> int:
    sample_no_idx = sample_sheet.column_index(['No'])
    ref_no_idx = reference_sheet.column_index(['No'])
    if sample_no_idx is None or ref_no_idx is None:
        return 0
    ref_rows = {str(_get(row, ref_no_idx)): row for row in reference_sheet.data_rows}
    mismatches = 0
    for sample_row in sample_sheet.data_rows:
        ref_row = ref_rows.get(str(_get(sample_row, sample_no_idx)))
        if ref_row is None:
            mismatches += 1
            continue
        if [_get(sample_row, idx) for idx in range(width)] != [_get(ref_row, idx) for idx in range(width)]:
            mismatches += 1
    return mismatches
