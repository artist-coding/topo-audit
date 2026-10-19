"""Read-only cost and sampling diagnostics; never selects an experimental condition."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def distribution(values):
    values = list(values)
    if not values:
        return {'n': 0, 'mean': None, 'p50': None, 'p90': None, 'max': None}
    return {'n': len(values), 'mean': float(np.mean(values)),
            'p50': float(np.percentile(values, 50)), 'p90': float(np.percentile(values, 90)),
            'max': float(np.max(values))}


def cost_summary(records, config, *, fresh_run_ids, wall_s, concurrency):
    """Physical node attempts, deduplicated by artifact run_id, including failures.

    Fresh means calls belonging to this invocation, not historical cache hits.
    Missing usage remains unknown. Wall time includes retries and cache overhead.
    """
    nodes, attempts, missing_artifacts = [], [], []
    unique = {r.run_id: r for r in records}
    for r in unique.values():
        path = Path(r.node_outputs_path)
        if not r.node_outputs_path or not path.is_file():
            missing_artifacts.append(r.run_id)
            continue
        artifact = json.loads(path.read_text())
        artifact_nodes = artifact.get('nodes', {})
        if r.split_role == 'pi_single':
            artifact_nodes = {'single': {'effective_role': 'pi_single',
                'attempts': artifact.get('attempts', []),
                'truncated': bool(r.truncated_nodes)}}
        for node in artifact_nodes.values():
            role = node.get('effective_role', 'unknown')
            node_attempts = node.get('attempts', [])
            nodes.append({'role': role, 'truncated': bool(node.get('truncated')),
                          'retries': max(0, len(node_attempts) - 1)})
            for a in node_attempts:
                attempts.append({**a, 'role': role, 'fresh': r.run_id in fresh_run_ids,
                                 'task_id': r.task_id, 'kind': 'pi_single' if r.split_role == 'pi_single' else 'graph'})
    success = [a for a in attempts if a.get('error') is None]
    def token(a): return (a.get('raw_usage') or {}).get('completion_tokens')
    fresh = [a for a in attempts if a['fresh']]
    fresh_tokens = sum(token(a) or 0 for a in fresh)
    missing_usage = sum(token(a) is None for a in fresh)
    throughput = fresh_tokens / wall_s if wall_s and fresh else None
    per_role = {}
    for role in sorted({n['role'] for n in nodes}):
        ns = [n for n in nodes if n['role'] == role]
        aa = [a for a in success if a['role'] == role]
        per_role[role] = {'completion_tokens': distribution(token(a) for a in aa if token(a) is not None),
                          'node_calls': len(ns), 'truncated_calls': sum(n['truncated'] for n in ns),
                          'truncation_rate': sum(n['truncated'] for n in ns) / len(ns)}
    lengths = distribution(token(a) for a in success if token(a) is not None)
    lengths_by_kind = {kind: distribution(token(a) for a in success
                       if a['kind'] == kind and token(a) is not None)
                       for kind in ('graph', 'pi_single')}
    by_task = {}
    for tid in sorted({a['task_id'] for a in attempts}):
        mine = [a for a in attempts if a['task_id'] == tid]
        by_task[tid] = {
            'attempts': len(mine),
            'completion_tokens_known': sum(token(a) or 0 for a in mine),
            'attempts_missing_usage': sum(token(a) is None for a in mine),
            'fresh_completion_tokens_known': sum(token(a) or 0 for a in mine if a['fresh']),
            'by_kind': {kind: sum(token(a) or 0 for a in mine if a['kind'] == kind)
                        for kind in ('graph', 'pi_single')}}
    total_known = sum(t['completion_tokens_known'] for t in by_task.values())
    for t in by_task.values():
        t['share_of_known_completion_tokens'] = t['completion_tokens_known'] / total_known if total_known else None

    # Derive graph calls from the actual space, rather than embedding 90 in estimates.
    from space.s0 import enumerate_space
    rows = {g.exec_hash: g for g in enumerate_space(model=config.get('model', {}).get('name', 'unspecified'))}
    node_sum = sum(len(g.nodes) for g in rows.values())
    sizes, reps = config.get('splits', {}), config.get('tensor', {}).get('repeats', {})
    tensor_cells = sum(sizes.get(s, 0) * reps.get(s, 1) for s in ('train', 'audit', 'val', 'test'))
    samples = max(config.get('single_agent', {}).get('budget_sweep', [8]))
    single_calls = 2 * samples + 6
    counts = {'p2_graph': node_sum * 150 * 2, 'p2_single': 150 * single_calls,
              'tensor': node_sum * tensor_cells, 'val_single': sizes.get('val', 0) * single_calls}
    counts['p2_total'] = counts['p2_graph'] + counts['p2_single']
    counts['tensor_plus_val_single'] = counts['tensor'] + counts['val_single']
    for val_repeats in (1, 2):
        counts[f'tensor_val_r{val_repeats}'] = node_sum * (
            tensor_cells + sizes.get('val', 0) * (val_repeats - reps.get('val', 1)))
    def days(calls, kind):
        mean = lengths_by_kind[kind]['mean']
        return calls * mean / throughput / 86400 if throughput and mean is not None else None
    estimates = {k: days(v, 'pi_single' if k in ('p2_single', 'val_single') else 'graph')
                 for k, v in counts.items() if k not in ('p2_total', 'tensor_plus_val_single')}
    for total, a, b in [('p2_total', 'p2_graph', 'p2_single'),
                         ('tensor_plus_val_single', 'tensor', 'val_single')]:
        estimates[total] = (estimates[a] + estimates[b]
                           if estimates[a] is not None and estimates[b] is not None else None)
    return {'wall_s': wall_s, 'concurrency': concurrency, 'completion_tokens': lengths,
            'by_effective_role': per_role,
            'completion_tokens_by_kind': lengths_by_kind,
            'by_task': by_task,
            'per_task_completion_tokens_known': distribution(t['completion_tokens_known'] for t in by_task.values()),
            'successful_attempt_tokens_per_second': distribution(
                token(a) / a['latency_s'] for a in success
                if token(a) is not None and a.get('latency_s', 0) > 0),
            'fresh_completion_tokens_known': fresh_tokens,
            'fresh_attempts': len(fresh), 'fresh_attempts_missing_usage': missing_usage,
            'throughput_tokens_per_second': throughput,
            'throughput_is_lower_bound': bool(missing_usage),
            'retries': sum(n['retries'] for n in nodes),
            'status_codes': dict(Counter(str(a.get('status_code')) for a in attempts)),
            'errors': dict(Counter(a['error'] for a in attempts if a.get('error'))),
            'failed_attempt_latency_s': distribution(a['latency_s'] for a in attempts if a.get('error')),
            'stream_failures': sum(a.get('error') == 'EndpointStreamError' for a in attempts),
            'missing_artifacts': missing_artifacts,
            'projection': {'calls': counts, 'days_at_measured_concurrency': estimates,
                           'assumptions': '150 P2 tasks x 2 repeats; configured tensor splits/repeats; '
                           'Graph and single-agent projections use their own observed means; unmeasured groups are null. No extrapolation to '
                           'unmeasured concurrency. Small, role-mixed smoke sample; not a runtime guarantee.'}}


def truncation_noise(records, max_spread, *, simulations=2000, seed=20261019):
    """Conditional row-label permutation reference, preserving task/repeat clusters.

    Tests exchangeable rows, not equivalence within the practical threshold.
    Non-rejection is not proof of safety. Fewer than 30 tasks is diagnostic only.
    """
    cells = defaultdict(list)
    for r in records:
        cells[(r.exec_hash, r.task_id)].append(float(bool(r.truncated_nodes)))
    rows = sorted({g for g, _ in cells}); tasks = sorted({t for _, t in cells})
    out = {'method': 'within_task_row_permutation', 'tasks': len(tasks),
           'simulations': simulations, 'seed': seed, 'reference_p95': None,
           'p_value': None, 'spread_ok': None,
           'note': 'Exchangeable row labels under the null; repeats stay together. '
                   'Non-rejection does not establish equivalence or adequate power.'}
    if max_spread is None or len(rows) < 2 or not tasks:
        return out
    if any((g,t) not in cells for g in rows for t in tasks):
        out['note'] += ' Incomplete row/task grid.'
        return out
    if any(len(cells[g,t]) != len(cells[rows[0],t]) for g in rows for t in tasks):
        out['note'] += ' Unequal repeat counts across rows.'
        return out
    total = sum(len(cells[rows[0],t]) for t in tasks)
    values = np.array([[sum(cells[g,t]) for t in tasks] for g in rows])
    spread = float(np.ptp(values.sum(axis=1) / total))
    out['runs_per_row'] = total
    rng = np.random.default_rng(seed)
    null = np.array([np.ptp(rng.permuted(values, axis=0).sum(axis=1) / total) for _ in range(simulations)])
    p95 = float(np.quantile(null, .95))
    out.update(observed_spread=spread, reference_p95=p95,
               p_value=float((1 + np.sum(null >= spread - 1e-12)) / (simulations + 1)))
    if len(tasks) >= 30:
        out['spread_ok'] = not (spread > max_spread and spread > p95 + 1e-12)
    else:
        out['note'] += ' Fewer than 30 tasks: diagnostic only, n/a.'
    return out
