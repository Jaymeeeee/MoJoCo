"""Passive target tracing must leave physical trajectories untouched."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.insert_plug.scene import build_scene, load_config
from examples.insert_plug.simulation import Simulation


spec = importlib.util.spec_from_file_location('lift_trace_script',
    Path(__file__).resolve().parents[2]/'scripts/diagnose_insert_plug_policy_lift.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_trace_does_not_change_live_solver_or_actual_motion(tmp_path):
    config = load_config()
    scene = build_scene(config, tmp_path/'scene.xml')
    sim, reference = Simulation(scene, config, render=False), Simulation(scene, config, render=False)
    policy = SimpleNamespace(policy=SimpleNamespace(predict_action_chunk=lambda x: x))
    try:
        trace = module.LiftTrace(sim, policy)
        action = np.r_[sim.data.qpos[sim.arm_qadr], sim.gripper_width()]
        before = {key: getattr(sim.data, key).copy() for key in ('qpos','qvel','qacc_warmstart','ctrl')}
        state = trace.snapshot_state()
        target = trace.fk(action, state['qpos'])
        np.testing.assert_allclose(target, state['tcp'], atol=1e-15)
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(sim.data,key), value)
        trace.install()
        for _ in range(3):
            sim.step(action)
            reference.step(action)
        trace.uninstall()
        for key in before:
            np.testing.assert_array_equal(getattr(sim.data,key), getattr(reference.data,key))
        assert len(trace.rows)==3
        assert trace.rows[-1]['after']['time'] == reference.data.time
    finally:
        sim.close()
        reference.close()


def test_prediction_hook_only_copies_real_call_result():
    values=np.arange(24,dtype=np.float32).reshape(1,3,8)
    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return values
    result=Tensor()
    calls=[]
    trace=module.LiftTrace.__new__(module.LiftTrace)
    trace.original_predict=lambda batch: calls.append(batch) or result
    trace.sim=SimpleNamespace(data=SimpleNamespace(time=1.,qpos=np.ones(8)))
    trace.chunks=[]
    batch=object()
    assert trace.predict(batch) is result
    assert calls==[batch]
    trace.chunks[0]['normalized'][0,0,0]=999
    assert values[0,0,0]==0
