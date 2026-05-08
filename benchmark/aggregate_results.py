"""
Aggregate experiment results into per-experiment summary tables matching
the templates in ablation_experiments.md.

Usage:
    python benchmark/aggregate_results.py
"""

import os
import re
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# Config parsing
# ============================================================
def parse_config(attn_dir_name):
    """Parse 'RetroInfer_top-p_True_lru_0.4' into components."""
    name = attn_dir_name.rstrip('/\\')
    parts = name.split('_')
    if len(parts) >= 5:
        try:
            return {
                'full': name,
                'attn_type': parts[0],
                'cluster_select': parts[1],
                'cluster_reuse': parts[2],
                'eviction_policy': parts[3],
                'top_p': parts[4],
            }
        except (ValueError, IndexError):
            pass
    return {'full': name, 'attn_type': name}


def parse_config_from_filename(filename):
    """Try to extract config from a filename with pattern *_top-p_True_lru_0.4*"""
    m = re.search(r'(RetroInfer|retroinfer|Full_Flash_Attn)_(top-[kp])_(True|False)_(lru|sclru|arc)_([\d.]+)', filename)
    if m:
        return m.group(2), m.group(3), m.group(4), m.group(5)
    return None


# ============================================================
# Task group definitions (matching experiment scripts)
# ============================================================
E1_HIGH_SENS_LB = ["musique", "gov_report", "passage_count"]
E1_LOW_SENS_LB = ["passage_retrieval_en", "trec", "triviaqa"]
E2_DIFFUSE_LB = ["gov_report", "musique", "passage_count"]
E2_FOCUSED_LB = ["passage_retrieval_en", "narrativeqa", "qasper"]
E3_LB = ["gov_report", "musique", "qasper"]
E4_LB_DIFFUSE = "gov_report"
E4_LB_FOCUSED = "passage_retrieval_en"


def avg_score(row, tasks):
    """Average score across specified tasks."""
    vals = [row.get(t) for t in tasks if row.get(t) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


# ============================================================
# Runtime metrics extraction from log files
# ============================================================
def _parse_runtime_log(content):
    """Extract runtime metrics from a single log content string."""
    result = {}
    m = re.search(r'Decoding latency:\s+([\d.]+)\s+ms/step', content)
    if m:
        result['ms_per_step'] = float(m.group(1))
    m = re.search(r'Throughput:\s+([\d.]+)\s+tokens/s', content)
    if m:
        result['tokens_per_sec'] = float(m.group(1))
    m = re.search(r'Cache hit rate:\s+([\d.]+)\s+\(hits=(\d+),\s*misses=(\d+)\)', content)
    if m:
        result['cache_hit_rate'] = float(m.group(1))
        result['cache_hits'] = int(m.group(2))
        result['cache_misses'] = int(m.group(3))
    m = re.search(r'Reuse hit rate:\s+([\d.]+)\s+\(hits=(\d+),\s*total=(\d+)\)', content)
    if m:
        result['reuse_hit_rate'] = float(m.group(1))
        result['reuse_hits'] = int(m.group(2))
        result['reuse_total'] = int(m.group(3))
    return result if result else None


def _extract_length_from_name(name):
    """Extract context length from a filename like ..._60k_... or ..._120k_..."""
    m = re.search(r'_(\d+k)\b', name)
    if m:
        return m.group(1)
    return None


def _collect_runtime(log_file, raw_entries, has_length=False):
    """Parse one log file and append runtime metrics to raw_entries.

    raw_entries: list of (config_key, length_or_None, metrics_dict)
    config_key = (cluster_select, cluster_reuse, eviction_policy, top_p)
    """
    config = parse_config_from_filename(log_file.name)
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return

    if not config:
        # fallback: parse config from log content (CLI args)
        sel = re.search(r'--cluster_select\s+(\S+)', content)
        reu = re.search(r'--cluster_reuse\s+(\S+)', content)
        pol = re.search(r'--eviction_policy\s+(\S+)', content)
        tp_p = re.search(r'--top_p\s+([\d.]+)', content)
        if not all([sel, reu, pol, tp_p]):
            return
        config = (sel.group(1), reu.group(1), pol.group(1), tp_p.group(1))

    metrics = _parse_runtime_log(content)
    if not metrics:
        return

    length = _extract_length_from_name(log_file.name) if has_length else None
    raw_entries.append((config, length, metrics))


def load_runtime_metrics(exp_log_dir, tp_log_dir):
    """Scan experiment logs and throughput logs for runtime metrics.

    Returns two structures:
      runtime[config_key] = {cache_hit_rate, reuse_hit_rate, tokens_per_sec, ms_per_step}
          aggregated across all runs for that config.
      tp_by_len[config_key] = {length: {tokens_per_sec, ...}}
          per-length breakdown from throughput logs only.

    config_key = (cluster_select, cluster_reuse, eviction_policy, top_p)
    """
    exp_log_dir = Path(exp_log_dir)
    tp_log_dir = Path(tp_log_dir)

    # Collect raw entries: (config_key, length_or_None, metrics_dict)
    raw_entries = []

    # Scan experiment logs
    if exp_log_dir.exists():
        for exp_dir in sorted(exp_log_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            for log_file in sorted(exp_dir.glob('*.log')):
                _collect_runtime(log_file, raw_entries)

    # Scan throughput logs (with length info)
    if tp_log_dir.exists():
        for log_file in sorted(tp_log_dir.glob('*.log')):
            _collect_runtime(log_file, raw_entries, has_length=True)

    # Aggregate: compute mean per config_key
    by_config = defaultdict(list)
    by_config_len = defaultdict(lambda: defaultdict(list))

    for config, length, metrics in raw_entries:
        by_config[config].append(metrics)
        if length:
            by_config_len[config][length].append(metrics)

    runtime = {}
    for config, entries in by_config.items():
        agg = {}
        for key in ['cache_hit_rate', 'reuse_hit_rate', 'tokens_per_sec', 'ms_per_step']:
            vals = [e[key] for e in entries if key in e]
            if vals:
                agg[key] = round(sum(vals) / len(vals), 4)
        runtime[config] = agg

    tp_by_len = {}
    for config, len_dict in by_config_len.items():
        tp_by_len[config] = {}
        for length, entries in len_dict.items():
            agg = {}
            for key in ['tokens_per_sec', 'ms_per_step', 'cache_hit_rate', 'reuse_hit_rate']:
                vals = [e[key] for e in entries if key in e]
                if vals:
                    agg[key] = round(sum(vals) / len(vals), 4)
            tp_by_len[config][length] = agg

    return runtime, tp_by_len


# ============================================================
# Data loaders
# ============================================================
def load_longbench(results_dir):
    """Load LongBench result.json files. Returns list of dicts."""
    results_dir = Path(results_dir)
    pred_dir = results_dir / 'pred'
    if not pred_dir.exists():
        print(f"Warning: {pred_dir} does not exist")
        return []

    rows = []
    for model_dir in sorted(pred_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for attn_dir in sorted(model_dir.iterdir()):
            if not attn_dir.is_dir():
                continue
            result_file = attn_dir / 'result.json'
            if not result_file.exists():
                continue
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            config = parse_config(attn_dir.name)
            row = {'model': model_dir.name, **config}
            for task, score in data.items():
                if isinstance(score, dict):
                    row[task] = round(score.get('8k+', score.get(list(score.keys())[0], 0)), 4)
                else:
                    row[task] = round(score, 4)
            rows.append(row)
    return rows


def load_ruler(results_dir):
    """Load RULER summary CSV files. Returns list of dicts."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist")
        return []

    rows = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for benchmark_dir in sorted(model_dir.iterdir()):
            if not benchmark_dir.is_dir():
                continue
            for seq_dir in sorted(benchmark_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                for attn_dir in sorted(seq_dir.iterdir()):
                    if not attn_dir.is_dir():
                        continue
                    pred_dir = attn_dir / 'pred'
                    if not pred_dir.exists():
                        continue
                    summary_files = list(pred_dir.glob('summary*.csv'))
                    if not summary_files:
                        continue
                    for sf in summary_files:
                        scores = {}
                        try:
                            with open(sf, 'r', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                task_names = None
                                score_values = None
                                for line in reader:
                                    if len(line) < 2:
                                        continue
                                    if line[0].strip() == 'Tasks':
                                        task_names = [c.strip() for c in line[1:]]
                                    elif line[0].strip() == 'Score':
                                        score_values = [c.strip() for c in line[1:]]
                                if task_names and score_values:
                                    for t, s in zip(task_names, score_values):
                                        try:
                                            scores[t] = round(float(s), 4)
                                        except ValueError:
                                            scores[t] = s
                        except Exception:
                            continue

                        if not scores:
                            continue

                        config = parse_config(attn_dir.name)
                        row = {
                            'model': model_dir.name,
                            'seq_len': seq_dir.name,
                            **config,
                            **scores,
                        }
                        rows.append(row)
    return rows


# ============================================================
# Helpers for build functions
# ============================================================
def _lb_scores_by_key(lb_rows, key_field):
    """Index LB rows by a config key."""
    out = {}
    for r in lb_rows:
        k = r.get(key_field, '?')
        if k not in out:
            out[k] = r
    return out


def _ruler_scores_by_key(ruler_rows, key_field):
    """Index RULER rows by config key."""
    out = {}
    for r in ruler_rows:
        k = r.get(key_field, '?')
        if k not in out:
            out[k] = r
    return out


def _rt(config_key, runtime, tp_by_len, length=None):
    """Get runtime metrics for a config key.
    If length is given, prefers tp_by_len[config][length], falls back to runtime.
    """
    if length and config_key in tp_by_len and length in tp_by_len[config_key]:
        return tp_by_len[config_key][length]
    return runtime.get(config_key, {})


# ============================================================
# Table builders
# ============================================================
def build_e1(lb_rows, ruler_rows, runtime, tp_by_len):
    """E1: 缓存替换策略."""
    lb_idx = _lb_scores_by_key(lb_rows, 'eviction_policy')
    ru_idx = _ruler_scores_by_key(ruler_rows, 'eviction_policy')

    rows = []
    for policy in ['lru', 'sclru', 'arc']:
        lb = lb_idx.get(policy, {})
        ru = ru_idx.get(policy, {})
        ck = ('top-p', 'True', policy, '0.4')
        rt = runtime.get(ck, {})

        rows.append({
            'eviction_policy': policy,
            'high_sens_LB': avg_score(lb, E1_HIGH_SENS_LB),
            'low_sens_LB': avg_score(lb, E1_LOW_SENS_LB),
            'multikey_RULER': ru.get('niah_multikey_1'),
            'single_RULER': ru.get('niah_single_1'),
            'cache_hit_rate': rt.get('cache_hit_rate'),
        })
    return rows


def build_e2(lb_rows, ruler_rows, runtime, tp_by_len):
    """E2: 聚类选择方式."""
    lb_idx = _lb_scores_by_key(lb_rows, 'cluster_select')
    ru_idx = _ruler_scores_by_key(ruler_rows, 'cluster_select')

    rows = []
    for mode in ['top-k', 'top-p']:
        lb = lb_idx.get(mode, {})
        ru = ru_idx.get(mode, {})
        ck = (mode, 'True', 'sclru', '0.4')
        rt = runtime.get(ck, {})

        rows.append({
            'cluster_select': mode,
            'diffuse_LB': avg_score(lb, E2_DIFFUSE_LB),
            'focused_LB': avg_score(lb, E2_FOCUSED_LB),
            'single_RULER': ru.get('niah_single_1'),
            'multi_RULER': ru.get('niah_multikey_1'),
            'throughput_tok_s': rt.get('tokens_per_sec'),
            'latency_ms': rt.get('ms_per_step'),
        })
    return rows


def build_e3(lb_rows, ruler_rows, runtime, tp_by_len):
    """E3: 聚类复用."""
    lb_idx = _lb_scores_by_key(lb_rows, 'cluster_reuse')
    ru_idx = _ruler_scores_by_key(ruler_rows, 'cluster_reuse')

    rows = []
    for reuse in ['False', 'True']:
        lb = lb_idx.get(reuse, {})
        ru = ru_idx.get(reuse, {})
        ck = ('top-p', reuse, 'sclru', '0.4')
        rt = runtime.get(ck, {})

        row = {'cluster_reuse': reuse}
        for length in ['60k', '120k', '240k', '480k']:
            row[f'tp_{length}'] = _rt(ck, runtime, tp_by_len, length).get('tokens_per_sec')
        row['latency_ms'] = rt.get('ms_per_step')
        row['LB_acc'] = avg_score(lb, E3_LB)
        row['RULER_acc'] = ru.get('niah_multikey_1')
        row['reuse_hit_rate'] = rt.get('reuse_hit_rate')
        rows.append(row)
    return rows


def build_e4(lb_rows, ruler_rows, runtime, tp_by_len):
    """E4: Top-p 阈值敏感性."""
    lb_idx = _lb_scores_by_key(lb_rows, 'top_p')
    ru_idx = _ruler_scores_by_key(ruler_rows, 'top_p')

    rows = []
    for tp_val in ['0.3', '0.4', '0.5', '0.6']:
        lb = lb_idx.get(tp_val, {})
        ru = ru_idx.get(tp_val, {})
        ck = ('top-p', 'True', 'sclru', tp_val)
        rt = runtime.get(ck, {})

        rows.append({
            'top_p': tp_val,
            'gov_report_acc': lb.get(E4_LB_DIFFUSE),
            'passage_retrieval_acc': lb.get(E4_LB_FOCUSED),
            'niah_single_acc': ru.get('niah_single_1'),
            'niah_multikey_acc': ru.get('niah_multikey_1'),
            'throughput_tok_s': rt.get('tokens_per_sec'),
        })
    return rows


def build_e5(lb_rows, ruler_rows, runtime, tp_by_len):
    """E5: 参数交互效应."""
    evictions = ['lru', 'arc']
    selects = ['top-k', 'top-p']
    reuses = ['True', 'False']

    lb_by_config = {}
    for r in lb_rows:
        key = (r.get('eviction_policy', '?'), r.get('cluster_select', '?'), r.get('cluster_reuse', '?'))
        lb_by_config[key] = r

    ru_by_config = {}
    for r in ruler_rows:
        key = (r.get('eviction_policy', '?'), r.get('cluster_select', '?'), r.get('cluster_reuse', '?'))
        ru_by_config[key] = r

    rows = []
    for ev in evictions:
        for sel in selects:
            for reu_val in reuses:
                key = (ev, sel, reu_val)
                lb = lb_by_config.get(key, {})
                ru = ru_by_config.get(key, {})
                ck = (sel, reu_val, ev, '0.4')
                rt = runtime.get(ck, {})

                rows.append({
                    'eviction_policy': ev,
                    'cluster_select': sel,
                    'cluster_reuse': reu_val,
                    'musique_acc': lb.get('musique'),
                    'gov_report_acc': lb.get('gov_report'),
                    'niah_mk1_acc': ru.get('niah_multikey_1'),
                    'niah_s1_acc': ru.get('niah_single_1'),
                    'cache_hit_rate': rt.get('cache_hit_rate'),
                    'throughput_tok_s': rt.get('tokens_per_sec'),
                })
    return rows


# ============================================================
# CSV writer
# ============================================================
def write_csv(rows, output_path):
    if not rows:
        print(f"No data to write to {output_path}")
        return
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Aggregate RetroInfer experiment results')
    parser.add_argument('--exp-log-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'exp_logs'),
                        help='Directory containing experiment logs')
    parser.add_argument('--tp-log-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'throughput_eval', 'different_lengths_logs'),
                        help='Directory containing throughput log files')
    parser.add_argument('--longbench-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'LongBench', 'results'))
    parser.add_argument('--ruler-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark', 'ruler', 'ruler_eval_result'))
    parser.add_argument('--outdir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'benchmark'),
                        help='Output directory for experiment CSVs')
    args = parser.parse_args()

    print("Loading data...")
    lb_rows = load_longbench(args.longbench_dir)
    ruler_rows = load_ruler(args.ruler_dir)
    runtime, tp_by_len = load_runtime_metrics(args.exp_log_dir, args.tp_log_dir)

    print(f"  LongBench configs: {len(lb_rows)}")
    print(f"  RULER configs:     {len(ruler_rows)}")
    print(f"  Runtime configs:   {len(runtime)}")
    if tp_by_len:
        print(f"  TP-length configs: {sum(len(v) for v in tp_by_len.values())}")

    outdir = args.outdir

    print("\nBuilding experiment tables...")
    write_csv(build_e1(lb_rows, ruler_rows, runtime, tp_by_len),
              os.path.join(outdir, 'e1_eviction.csv'))
    write_csv(build_e2(lb_rows, ruler_rows, runtime, tp_by_len),
              os.path.join(outdir, 'e2_selection.csv'))
    write_csv(build_e3(lb_rows, ruler_rows, runtime, tp_by_len),
              os.path.join(outdir, 'e3_reuse.csv'))
    write_csv(build_e4(lb_rows, ruler_rows, runtime, tp_by_len),
              os.path.join(outdir, 'e4_topp.csv'))
    write_csv(build_e5(lb_rows, ruler_rows, runtime, tp_by_len),
              os.path.join(outdir, 'e5_interact.csv'))

    print("\nDone. Summary files written to benchmark/e[1-5]_*.csv")


if __name__ == '__main__':
    main()
