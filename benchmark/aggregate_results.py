"""
Aggregate experiment results from log files, LongBench result.json, and RULER summary.csv.

Usage:
    python benchmark/aggregate_results.py [--throughput-dir throughput_eval/different_lengths_logs]
                                         [--longbench-dir benchmark/LongBench/results]
                                         [--ruler-dir benchmark/ruler/ruler_eval_result]
                                         [--output summary.csv]
"""

import os
import re
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def parse_throughput_log(log_path):
    """Parse a single throughput log file and extract metrics."""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    result = {}
    # Decoding latency
    m = re.search(r'Decoding latency:\s+([\d.]+)\s+ms/step', content)
    if m:
        result['ms_per_step'] = float(m.group(1))
    # Throughput
    m = re.search(r'Throughput:\s+([\d.]+)\s+tokens/s', content)
    if m:
        result['tokens_per_sec'] = float(m.group(1))
    # Cache hit rate
    m = re.search(r'Cache hit rate:\s+([\d.]+)\s+\(hits=(\d+),\s*misses=(\d+)\)', content)
    if m:
        result['cache_hit_rate'] = float(m.group(1))
        result['cache_hits'] = int(m.group(2))
        result['cache_misses'] = int(m.group(3))
    # Reuse hit rate
    m = re.search(r'Reuse hit rate:\s+([\d.]+)\s+\(hits=(\d+),\s*total=(\d+)\)', content)
    if m:
        result['reuse_hit_rate'] = float(m.group(1))
        result['reuse_hits'] = int(m.group(2))
        result['reuse_total'] = int(m.group(3))
    # Prefill latency
    m = re.search(r'Prefilling latency:\s+([\d.]+)\s+s', content)
    if m:
        result['prefill_s'] = float(m.group(1))

    return result if result else None


def aggregate_throughput_logs(log_dir):
    """Aggregate throughput logs, grouping by configuration and computing mean/std across rounds."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"Warning: {log_dir} does not exist, skipping throughput logs")
        return []

    # Pattern: {attn}_{length}_bsz{N}_{round}.log
    groups = defaultdict(list)
    for log_file in sorted(log_dir.glob('*.log')):
        fname = log_file.name
        m = re.match(r'(.+)_(\d+k?)_bsz(\d+)_(\d+)\.log', fname)
        if m:
            attn = m.group(1)
            length = m.group(2)
            bsz = int(m.group(3))
            round_num = int(m.group(4))
            result = parse_throughput_log(log_file)
            if result:
                result['_file'] = fname
                result['_round'] = round_num
                result['_attn'] = attn
                result['_length'] = length
                result['_bsz'] = bsz
                groups[(attn, length, bsz)].append(result)

    rows = []
    for (attn, length, bsz), entries in sorted(groups.items()):
        n = len(entries)
        metrics = {}
        for key in ['tokens_per_sec', 'ms_per_step', 'cache_hit_rate', 'reuse_hit_rate']:
            vals = [e[key] for e in entries if key in e]
            if vals:
                mean_val = sum(vals) / len(vals)
                if len(vals) >= 2:
                    std_val = (sum((v - mean_val) ** 2 for v in vals) / len(vals)) ** 0.5
                else:
                    std_val = 0
                metrics[f'{key}_mean'] = round(mean_val, 4)
                metrics[f'{key}_std'] = round(std_val, 4)
                metrics[f'{key}_n'] = len(vals)

        row = {
            'attn_type': attn,
            'context_len': length,
            'batch_size': bsz,
            'num_rounds': n,
        }
        row.update(metrics)
        rows.append(row)

    return rows


def aggregate_longbench_results(results_dir):
    """Parse LongBench result.json files."""
    results_dir = Path(results_dir)
    pred_dir = results_dir / 'pred'
    if not pred_dir.exists():
        print(f"Warning: {pred_dir} does not exist, skipping LongBench results")
        return []

    rows = []
    for model_dir in sorted(pred_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for attn_dir in sorted(model_dir.iterdir()):
            if not attn_dir.is_dir():
                continue
            attn_type = attn_dir.name
            result_file = attn_dir / 'result.json'
            if not result_file.exists():
                continue
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            row = {'model': model_name, 'attn_type': attn_type}
            for task, score in data.items():
                row[task] = round(score, 4)
            rows.append(row)

    return rows


def aggregate_ruler_results(results_dir):
    """Parse RULER summary CSV files."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist, skipping RULER results")
        return []

    rows = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for benchmark_dir in sorted(model_dir.iterdir()):
            if not benchmark_dir.is_dir():
                continue
            for seq_dir in sorted(benchmark_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                seq_len = seq_dir.name
                for attn_dir in sorted(seq_dir.iterdir()):
                    if not attn_dir.is_dir():
                        continue
                    attn_type = attn_dir.name
                    pred_dir = attn_dir / 'pred'
                    if not pred_dir.exists():
                        continue
                    summary_files = list(pred_dir.glob('summary*.csv'))
                    if not summary_files:
                        continue
                    for sf in summary_files:
                        with open(sf, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row_data in reader:
                                row = {
                                    'model': model_name,
                                    'attn_type': attn_type,
                                    'seq_len': seq_len,
                                }
                                row.update(row_data)
                                rows.append(row)

    return rows


def write_csv(rows, output_path):
    """Write aggregated rows to CSV."""
    if not rows:
        print("No data to write")
        return

    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate RetroInfer experiment results')
    parser.add_argument('--throughput-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'throughput_eval', 'different_lengths_logs'),
                        help='Directory containing throughput log files')
    parser.add_argument('--longbench-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'LongBench', 'results'),
                        help='Directory containing LongBench results')
    parser.add_argument('--ruler-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'ruler', 'ruler_eval_result'),
                        help='Directory containing RULER evaluation results')
    parser.add_argument('--output', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'summary.csv'),
                        help='Output CSV file for main aggregated table')
    parser.add_argument('--output-throughput', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'throughput_summary.csv'),
                        help='Output CSV for throughput results')
    parser.add_argument('--output-longbench', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'longbench_summary.csv'),
                        help='Output CSV for LongBench results')
    parser.add_argument('--output-ruler', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'ruler_summary.csv'),
                        help='Output CSV for RULER results')
    args = parser.parse_args()

    # Throughput logs (mean/std across rounds)
    tp_rows = aggregate_throughput_logs(args.throughput_dir)
    if tp_rows:
        write_csv(tp_rows, args.output_throughput)

    # LongBench results
    lb_rows = aggregate_longbench_results(args.longbench_dir)
    if lb_rows:
        write_csv(lb_rows, args.output_longbench)

    # RULER results
    ru_rows = aggregate_ruler_results(args.ruler_dir)
    if ru_rows:
        write_csv(ru_rows, args.output_ruler)

    # Print summary to console
    print("\n=== Summary ===")
    if tp_rows:
        print(f"Throughput configs: {len(tp_rows)} (see {args.output_throughput})")
    if lb_rows:
        print(f"LongBench configs: {len(lb_rows)} (see {args.output_longbench})")
    if ru_rows:
        print(f"RULER configs: {len(ru_rows)} (see {args.output_ruler})")


if __name__ == '__main__':
    main()
