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


def test_graph_and_single_lengths_are_not_mixed_in_projections(tmp_path):
    usage=lambda n: {'error':None,'status_code':200,'latency_s':1,'raw_usage':{'completion_tokens':n}}
    gp=tmp_path/'graph.json';gp.write_text(json.dumps({'nodes':{'n4':{
        'effective_role':'solver','attempts':[usage(100)]}}}))
    sp=tmp_path/'single.json';sp.write_text(json.dumps({'attempts':[usage(10)]}))
    graph=replace(_record('g','t1',1),node_outputs_path=str(gp))
    single=replace(_record('s','t2',1),split_role='pi_single',node_outputs_path=str(sp))
    cfg={'splits':{'train':256,'audit':300,'val':1350,'test':1000},
         'tensor':{'repeats':{'train':8,'audit':2,'val':2,'test':1}}}
    cost=cost_summary([graph,single],cfg,fresh_run_ids={graph.run_id,single.run_id},wall_s=10,concurrency=2)
    assert cost['completion_tokens_by_kind']['graph']['mean']==100
    assert cost['completion_tokens_by_kind']['pi_single']['mean']==10
    days=cost['projection']['days_at_measured_concurrency'];calls=cost['projection']['calls']
    assert days['tensor']==pytest.approx(571320*100/11/86400)
    assert days['p2_single']==pytest.approx(3300*10/11/86400)
    assert days['p2_total']==pytest.approx(days['p2_graph']+days['p2_single'])
    assert calls['tensor_val_r1']==449820 and calls['tensor_val_r2']==571320
    assert cost['by_task']['t1']['share_of_known_completion_tokens']==pytest.approx(100/110)
    assert cost['per_task_completion_tokens_known']['mean']==55
    graph_only=cost_summary([graph],cfg,fresh_run_ids={graph.run_id},wall_s=10,concurrency=2)
    assert graph_only['projection']['days_at_measured_concurrency']['p2_total'] is None
    assert graph_only['projection']['days_at_measured_concurrency']['val_single'] is None
