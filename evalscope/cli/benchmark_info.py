# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CLI command for benchmark information and documentation management.

This module provides CLI commands to:
- Display benchmark metadata and statistics
- Update benchmark data in individual JSON files per benchmark
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
            nargs='*',
            default=None,
            help=
            'Name(s) of the benchmark(s) (e.g., gsm8k, mmlu). Multiple benchmarks can be separated by spaces. Use --all for all benchmarks.',
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
            '--tag',
            nargs='+',
            default=None,
            help='Filter benchmarks by tag(s) when listing (e.g., --tag Agent Coding). Case-insensitive.',
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
            '--workers',
            type=int,
            default=8,
            help='Number of parallel workers for update/translation (default: 8)',
        )

        parser.set_defaults(func=BenchmarkInfoCMD)

    def execute(self):
        """Execute the benchmark-info command."""
        import os

        from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
        os.environ['BUILD_DOC'] = '1'

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
            benchmark_names = self.args.benchmark if isinstance(self.args.benchmark, list) else [self.args.benchmark]

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

    @staticmethod
    def _format_metric_list(metric_list):
        """Format metric_list (mixed strings and dicts) into readable strings."""
        formatted = []
        for item in (metric_list or []):
            if isinstance(item, str):
                formatted.append(item)
            elif isinstance(item, dict):
                for metric_name, config in item.items():
                    if isinstance(config, dict) and config:
                        params = ', '.join(f'{k}={v}' for k, v in config.items())
                        formatted.append(f'{metric_name} ({params})')
                    else:
                        formatted.append(str(metric_name))
        return formatted

    def _list_benchmarks(self):
        """List all available benchmarks."""
        from tabulate import tabulate

        from evalscope.api.registry import BENCHMARK_REGISTRY
        from evalscope.utils.doc_utils.generate_dataset_md import get_category_from_adapter_class

        # Collect flat rows: one row per benchmark
        filter_tags = None
        if hasattr(self.args, 'tag') and self.args.tag:
            filter_tags = {t.lower() for t in self.args.tag}

        rows = []
        for name, meta in sorted(BENCHMARK_REGISTRY.items()):
            tags = meta.tags or ['Other']
            # Apply tag filter (case-insensitive)
            if filter_tags:
                if not any(t.lower() in filter_tags for t in tags):
                    continue
            category = get_category_from_adapter_class(meta.data_adapter)
            tags_str = ', '.join(tags)
            n_metrics = len(meta.metric_list) if meta.metric_list else 0
            n_subsets = len(meta.subset_list) if meta.subset_list else 0
            rows.append([name, meta.pretty_name or name, category, tags_str, n_metrics, n_subsets])

        headers = ['Name', 'Pretty Name', 'Category', 'Tags', 'Metrics', 'Subsets']
        print(tabulate(rows, headers=headers, tablefmt='simple'))
        print(f'\nTotal: {len(rows)} benchmarks')

    def _update_benchmark_data(self, benchmark_names: list):
        """
        Update benchmark data in individual JSON files.

        This updates the individual JSON files in evalscope/benchmarks/_meta/ with:
        - Metadata from the adapter
        - Statistics (if --compute-stats is specified)
        - Sample example
        - Generated README content
        """
        from evalscope.utils.doc_utils.generate_dataset_md import update_benchmark_data

        # Pass the list of benchmarks to update_benchmark_data
        # It will handle parallel processing based on workers parameter
        update_benchmark_data(
            benchmark_name=benchmark_names,
            force=self.args.force,
            compute_stats=self.args.compute_stats,
            workers=self.args.workers,
        )

    def _translate_benchmarks(self):
        """Translate README content to Chinese."""
        from evalscope.utils.doc_utils.translate_description import translate_benchmarks

        # Determine which benchmarks to translate
        benchmark_names = None
        if self.args.benchmark:
            benchmark_names = self.args.benchmark if isinstance(self.args.benchmark, list) else [self.args.benchmark]
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
        from evalscope.utils.doc_utils.generate_dataset_md import get_adapter_category

        meta = adapter._benchmark_meta
        category = get_adapter_category(adapter)

        if self.args.format == 'json':
            self._display_json(adapter, category)
        elif self.args.format == 'markdown':
            self._display_markdown(adapter)
        else:
            self._display_text(meta, category)

    def _display_text(self, meta, category=None):
        """Display info in text format."""
        print(f'\n{"=" * 60}')
        print(f'Benchmark: {meta.pretty_name or meta.name}')
        print(f'{"=" * 60}')
        print(f'Name:          {meta.name}')
        print(f'Dataset ID:    {meta.dataset_id}')
        print(f'Category:      {category or "N/A"}')
        print(f'Tags:          {", ".join(meta.tags) if meta.tags else "N/A"}')
        print(f'Output Types:  {", ".join(meta.output_types) if meta.output_types else "N/A"}')
        print(f'Few-shot:      {meta.few_shot_num}-shot')
        print(f'Aggregation:   {meta.aggregation or "mean"}')
        print(f'Train Split:   {meta.train_split or "N/A"}')
        print(f'Eval Split:    {meta.eval_split or "N/A"}')
        print(f'Subsets:       {", ".join(meta.subset_list) if meta.subset_list else "N/A"}')

        if meta.paper_url:
            print(f'Paper:         {meta.paper_url}')

        # Metrics
        if meta.metric_list:
            formatted_metrics = self._format_metric_list(meta.metric_list)
            print(f'\nMetrics:')
            for m in formatted_metrics:
                print(f'  - {m}')

        if meta.description:
            print(f'\nDescription:')
            for line in meta.description.split('\n'):
                print(f'  {line}')

        # Prompt template (truncated for readability)
        if meta.prompt_template:
            print(f'\nPrompt Template:')
            template = meta.prompt_template
            if len(template) > 200:
                template = template[:200] + '... [TRUNCATED]'
            for line in template.split('\n'):
                print(f'  {line}')

        # System prompt (truncated for readability)
        if meta.system_prompt:
            print(f'\nSystem Prompt:')
            prompt = meta.system_prompt
            if len(prompt) > 200:
                prompt = prompt[:200] + '... [TRUNCATED]'
            for line in prompt.split('\n'):
                print(f'  {line}')

        # Configurable parameters (extra_params with full spec)
        if meta.extra_params:
            print(f'\nConfigurable Parameters:')
            for param_name, param_spec in meta.extra_params.items():
                print(f'  {param_name}:')
                if meta._is_spec_entry(param_spec):
                    print(f'    Type:        {param_spec.get("type", "N/A")}')
                    print(f'    Default:     {param_spec.get("value", "N/A")}')
                    print(f'    Description: {param_spec.get("description", "N/A")}')
                    if param_spec.get('choices'):
                        print(f'    Choices:     {param_spec["choices"]}')
                else:
                    print(f'    Value:       {param_spec}')

        if meta.data_statistics:
            stats = meta.data_statistics
            print(f'\nStatistics:')
            print(f'  Total samples:        {stats.total_samples:,}')
            print(f'  Prompt length (mean): {stats.prompt_length_mean:.1f}')
            print(f'  Prompt length (range): {stats.prompt_length_min} - {stats.prompt_length_max}')

        print()

    def _display_json(self, adapter, category=None):
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
            'category': category,
            'tags': meta.tags,
            'output_types': meta.output_types,
            'description': meta.description,
            'paper_url': meta.paper_url,
            'few_shot_num': meta.few_shot_num,
            'train_split': meta.train_split,
            'eval_split': meta.eval_split,
            'subset_list': meta.subset_list,
            'metric_list': meta.metric_list,
            'aggregation': meta.aggregation,
            'prompt_template': meta.prompt_template,
            'system_prompt': meta.system_prompt,
            'extra_params': meta.extra_params,
            'sandbox_config': meta.sandbox_config if meta.sandbox_config else None,
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
