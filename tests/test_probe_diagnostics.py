from dataclasses import replace
import json
import numpy as np
import pytest
from analysis.probe_diagnostics import cost_summary, truncation_noise
from scripts.p2_spread import probe_graphs, summarize_probe
from tests.test_p2 import _record


def test_noise_reference_preserves_shared_task_truncation():
    records = [replace(_record(g, str(t), 0), truncated_nodes=('n4',) if t < 5 else ())
               for g in ('a','b','c') for t in range(100)]
    d = truncation_noise(records, .03, simulations=100)
    assert d['reference_p95'] == 0 and d['spread_ok'] is True
    records = [replace(r, truncated_nodes=('n4',) if r.exec_hash == 'a' else ()) for r in records]
    assert truncation_noise(records, .03, simulations=100)['spread_ok'] is False
    assert truncation_noise(records[:5], .03)['spread_ok'] is None


def test_equal_rate_noise_not_treated_as_large_systematic_spread():
    rng = np.random.default_rng(9)
    records = [replace(_record(g, str(t), 0), truncated_nodes=('n4',) if rng.random()<.03 else ())
               for g in map(str, range(25)) for t in range(300)]
    d=truncation_noise(records,.03,simulations=500)
    assert d['reference_p95'] > .03
    assert d['spread_ok'] is True


def test_cost_counts_physical_attempts_and_unknown_usage(tmp_path):
    p=tmp_path/'call.json'
    p.write_text(json.dumps({'nodes': {'n4': {'effective_role':'solver','truncated':False,'attempts':[
        {'error':'EndpointStreamError','status_code':200,'latency_s':120,'raw_usage':{}},
        {'error':None,'status_code':200,'latency_s':2,'raw_usage':{'completion_tokens':100}}]}}}))
    r=replace(_record('g','t',1),node_outputs_path=str(p))
    config={'model':{'name':'m'},'splits':{'train':256,'audit':300,'val':1350,'test':1000},
            'tensor':{'repeats':{'train':8,'audit':2,'val':2,'test':1}}}
    cost=cost_summary([r,r],config,fresh_run_ids={r.run_id},wall_s=125,concurrency=4)
    assert cost['fresh_attempts']==2 and cost['retries']==1
    assert cost['fresh_completion_tokens_known']==100
    assert cost['throughput_tokens_per_second']==.8
    assert cost['throughput_is_lower_bound']
    assert cost['projection']['calls']['p2_total']==30300
    assert cost['projection']['calls']['tensor']==571320
    replay=cost_summary([r],config,fresh_run_ids=set(),wall_s=1,concurrency=4)
    assert replay['throughput_tokens_per_second'] is None
    assert replay['projection']['days_at_measured_concurrency']['tensor'] is None


def test_repeat_variance_projection_is_explicit_and_clips_negative_between():
    graphs = dict(list(probe_graphs('m','hand6').items())[:2])
    records=[replace(_record(g.exec_hash,str(t),float(rep)),repeat=rep)
             for g in graphs.values() for t in range(10) for rep in range(2)]
    s=summarize_probe(records,[],graphs,list(map(str,range(10))),repeats=2)
    p=s['power']['repeat_projection']
    assert p['within_task_variance']==1
    assert p['between_task_variance_raw']==-.5 and p['between_clipped']
    assert p['by_repeats']['1']['variance']==1
    assert p['by_repeats']['2']['variance']==.5


def test_zero_observed_variance_does_not_claim_zero_required_tasks():
    graphs = dict(list(probe_graphs('m','hand6').items())[:2])
    records=[replace(_record(g.exec_hash,str(t),1.),repeat=rep)
             for g in graphs.values() for t in range(10) for rep in range(2)]
    s=summarize_probe(records,[],graphs,list(map(str,range(10))),repeats=2)
    assert s['power']['repeat_projection']['by_repeats']['1']['val_tasks_needed']=={}
