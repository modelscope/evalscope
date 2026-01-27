# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CLI command for benchmark information and documentation management.

This module provides CLI commands to:
- Display benchmark metadata and statistics
- Update benchmark data in the unified JSON file
- Translate README content to Chinese
- Generate documentation files

Usage:
    # Show info for a specific benchmark
    evalscope benchmark-info gsm8k

    # List all available benchmarks
    evalscope benchmark-info --list

    # Update benchmark data (metadata only)
    evalscope benchmark-info gsm8k --update

    # Update benchmark data with statistics (requires dataset download)
    evalscope benchmark-info gsm8k --update --compute-stats

    # Update all benchmarks
    evalscope benchmark-info --all --update

    # Translate benchmarks that need translation
    evalscope benchmark-info --translate

    # Translate specific benchmark
    evalscope benchmark-info gsm8k --translate

    # Force re-translate
    evalscope benchmark-info --translate --force

    # Generate documentation from persisted data
    evalscope benchmark-info --generate-docs
"""

import json
from argparse import ArgumentParser, Namespace

from evalscope.cli.base import CLICommand
from evalscope.utils.logger import get_logger

logger = get_logger()


class BenchmarkInfoCMD(CLICommand):
    """
    Command line tool for benchmark information and documentation management.

    Examples:
        # Show info for a specific benchmark
        evalscope benchmark-info gsm8k

        # List all available benchmarks
        evalscope benchmark-info --list

        # Update benchmark data (metadata, statistics, sample)
        evalscope benchmark-info gsm8k --update --compute-stats

        # Update all benchmarks (metadata only)
        evalscope benchmark-info --all --update

        # Translate benchmarks that need it
        evalscope benchmark-info --translate

        # Generate documentation from persisted data
        evalscope benchmark-info --generate-docs
    """

    name = 'benchmark-info'

    def __init__(self, args: Namespace):
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParser):
        parser = parsers.add_parser(
            BenchmarkInfoCMD.name,
            help='Display benchmark information and manage documentation',
        )

        parser.add_argument(
            'benchmark',
            nargs='?',
            default=None,
            help='Name of the benchmark (e.g., gsm8k, mmlu). Use --all for all benchmarks.',
        )

        parser.add_argument(
            '--all',
            action='store_true',
            help='Process all registered benchmarks',
        )

        parser.add_argument(
            '--list',
            action='store_true',
            help='List all available benchmarks',
        )

        parser.add_argument(
            '--update',
            action='store_true',
            help='Update benchmark data in the unified JSON file',
        )

        parser.add_argument(
            '--compute-stats',
            action='store_true',
            help='Compute data statistics when updating (requires dataset download)',
        )

        parser.add_argument(
            '--translate',
            action='store_true',
            help='Translate README content to Chinese',
        )

        parser.add_argument(
            '--force',
            action='store_true',
            help='Force recompute/re-translate even if data exists',
        )

        parser.add_argument(
            '--generate-docs',
            action='store_true',
            help='Generate documentation files from persisted benchmark data',
        )

        parser.add_argument(
            '--format',
            choices=['text', 'json', 'markdown'],
            default='text',
            help='Output format for benchmark info (default: text)',
        )

        parser.add_argument(
            '--max-samples',
            type=int,
            default=5000,
            help='Maximum samples per subset when computing statistics (default: 5000)',
        )

        parser.add_argument(
            '--workers',
            type=int,
            default=8,
            help='Number of parallel workers for update/translation (default: 8)',
        )

        parser.set_defaults(func=BenchmarkInfoCMD)

    def execute(self):
        """Execute the benchmark-info command."""
        from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark

        # Handle --list flag
        if self.args.list:
            self._list_benchmarks()
            return

        # Handle --generate-docs flag
        if self.args.generate_docs:
            self._generate_docs()
            return

        # Handle --translate flag
        if self.args.translate:
            self._translate_benchmarks()
            return

        # Validate arguments for update operation
        if not self.args.benchmark and not self.args.all:
            if self.args.update:
                print('Error: Please specify a benchmark name or use --all for update')
                return
            print('Error: Please specify a benchmark name or use --all')
            print('Use --list to see available benchmarks')
            return

        # Get benchmarks to process
        if self.args.all:
            benchmark_names = list(BENCHMARK_REGISTRY.keys())
        else:
            benchmark_names = [self.args.benchmark]

        # Update benchmark data if requested
        if self.args.update:
            self._update_benchmark_data(benchmark_names)
            return

        # Display info
        for name in benchmark_names:
            try:
                logger.info(f'Processing benchmark: {name}')
                adapter = get_benchmark(name)
                self._display_info(adapter)
            except Exception as e:
                logger.error(f'Error processing {name}: {e}')
                if not self.args.all:
                    raise

    def _list_benchmarks(self):
        """List all available benchmarks."""
        from evalscope.api.registry import BENCHMARK_REGISTRY

        print('\nAvailable Benchmarks:')
        print('=' * 60)

        # Group by tags
        benchmarks_by_tag = {}
        for name, meta in sorted(BENCHMARK_REGISTRY.items()):
            for tag in meta.tags or ['Other']:
                if tag not in benchmarks_by_tag:
                    benchmarks_by_tag[tag] = []
                benchmarks_by_tag[tag].append((name, meta.pretty_name or name))

        for tag in sorted(benchmarks_by_tag.keys()):
            print(f'\n{tag}:')
            for name, pretty_name in sorted(benchmarks_by_tag[tag]):
                print(f'  - {name:30s} ({pretty_name})')

        print(f'\nTotal: {len(BENCHMARK_REGISTRY)} benchmarks')

    def _update_benchmark_data(self, benchmark_names: list):
        """
        Update benchmark data in the unified JSON file.

        This updates the benchmark_data.json file with:
        - Metadata from the adapter
        - Statistics (if --compute-stats is specified)
        - Sample example
        - Generated README content
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        from evalscope.utils.doc_utils.generate_dataset_md import update_benchmark_data

        def update_single(name: str) -> tuple:
            """Update a single benchmark and return (name, error or None)."""
            try:
                update_benchmark_data(
                    benchmark_name=name,
                    force=self.args.force,
                    compute_stats=self.args.compute_stats,
                    max_samples=self.args.max_samples,
                )
                return (name, None)
            except Exception as e:
                logger.error(f'Error updating {name}: {e}')
                return (name, str(e))

        failed_benchmarks = []
        workers = self.args.workers

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(update_single, name): name for name in benchmark_names}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Updating benchmarks'):
                name, error = future.result()
                if error:
                    failed_benchmarks.append(name)

        if failed_benchmarks:
            logger.error(f'Failed to update benchmarks: {failed_benchmarks}')

    def _translate_benchmarks(self):
        """Translate README content to Chinese."""
        from evalscope.utils.doc_utils.translate_description import translate_benchmarks

        # Determine which benchmarks to translate
        benchmark_names = None
        if self.args.benchmark:
            benchmark_names = [self.args.benchmark]
        elif self.args.all:
            # All benchmarks that need translation
            benchmark_names = None

        result = translate_benchmarks(
            benchmark_names=benchmark_names,
            force=self.args.force,
            workers=self.args.workers,
        )

        print(f'\nTranslation complete:')
        print(f'  Total: {result.get("total", 0)}')
        print(f'  Translated: {result.get("translated", 0)}')
        print(f'  Skipped: {result.get("skipped", 0)}')
        if result.get('errors'):
            print(f'  Errors: {len(result["errors"])}')
            for name, error in result['errors'][:5]:
                print(f'    - {name}: {error}')

    def _generate_docs(self):
        """Generate documentation from persisted benchmark data."""
        from evalscope.utils.doc_utils.generate_dataset_md import generate_docs
        generate_docs()

    def _display_info(self, adapter):
        """Display benchmark information."""
        meta = adapter._benchmark_meta

        if self.args.format == 'json':
            self._display_json(adapter)
        elif self.args.format == 'markdown':
            self._display_markdown(adapter)
        else:
            self._display_text(meta)

    def _display_text(self, meta):
        """Display info in text format."""
        print(f'\n{"=" * 60}')
        print(f'Benchmark: {meta.pretty_name or meta.name}')
        print(f'{"=" * 60}')
        print(f'Name:          {meta.name}')
        print(f'Dataset ID:    {meta.dataset_id}')
        print(f'Tags:          {", ".join(meta.tags) if meta.tags else "N/A"}')
        print(f'Few-shot:      {meta.few_shot_num}-shot')
        print(f'Eval Split:    {meta.eval_split or "N/A"}')
        print(f'Subsets:       {", ".join(meta.subset_list) if meta.subset_list else "N/A"}')

        if meta.paper_url:
            print(f'Paper:         {meta.paper_url}')
        if meta.homepage:
            print(f'Homepage:      {meta.homepage}')
        if meta.data_license:
            print(f'License:       {meta.data_license}')

        if meta.description:
            print(f'\nDescription:')
            for line in meta.description.split('\n'):
                print(f'  {line}')

        if meta.data_statistics:
            stats = meta.data_statistics
            print(f'\nStatistics:')
            print(f'  Total samples:        {stats.total_samples:,}')
            print(f'  Prompt length (mean): {stats.prompt_length_mean:.1f}')
            print(f'  Prompt length (range): {stats.prompt_length_min} - {stats.prompt_length_max}')

        print()

    def _display_json(self, adapter):
        """Display info in JSON format."""
        from evalscope.utils.doc_utils import load_benchmark_data

        # Try to load from persisted data first
        try:
            data = load_benchmark_data()
            entry = data.get(adapter.name, {})

            if entry:
                print(json.dumps(entry, indent=2, ensure_ascii=False))
                return
        except Exception:
            pass

        # Fall back to adapter metadata
        meta = adapter._benchmark_meta
        data = {
            'name': meta.name,
            'pretty_name': meta.pretty_name,
            'dataset_id': meta.dataset_id,
            'tags': meta.tags,
            'description': meta.description,
            'paper_url': meta.paper_url,
            'few_shot_num': meta.few_shot_num,
            'eval_split': meta.eval_split,
            'subset_list': meta.subset_list,
            'metric_list': meta.metric_list,
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))

    def _display_markdown(self, adapter):
        """Display info in markdown format."""
        from evalscope.utils.doc_utils import load_benchmark_data
        from evalscope.utils.doc_utils.generate_dataset_md import generate_readme_content

        # Try to load from persisted data first
        try:
            data = load_benchmark_data()
            entry = data.get(adapter.name, {})

            if entry and entry.get('readme', {}).get('en'):
                print(entry['readme']['en'])
                return
        except Exception:
            pass

        # Fall back to generating from adapter
        from evalscope.utils.readme_generator import generate_benchmark_readme
        readme = generate_benchmark_readme(adapter, compute_if_missing=False)
        print(readme)
