"""Grid layout reproducibility and full rendered-mesh containment checks."""

import numpy as np
import pytest

from examples.insert_plug.environment.layouts import candidate_layouts, head_visibility, mesh_world_vertices
from examples.insert_plug.environment.scene import build_scene, load_config
from examples.insert_plug.environment.simulation import Simulation
from mujoco_tools.cameras import camera_metadata


def test_candidates_are_repeatable_nonzero_two_centimetre_grid_offsets():
    config = load_config()
    first, again = list(candidate_layouts(config)), list(candidate_layouts(config))
    assert first == again and len(first) == 24 * 24
    assert len({(tuple(row['source_steps']), tuple(row['target_steps'])) for row in first}) == len(first)
    for row in first:
        for name in ('source', 'target'):
            steps = row[f'{name}_steps']
            assert steps != [0, 0]
            assert all(isinstance(x, int) and abs(x) <= 2 for x in steps)
            np.testing.assert_allclose(np.asarray(row[f'{name}_xy']) - config['board'][f'{name}_xy'],
                                       .02 * np.asarray(steps), atol=1e-10)


def test_visibility_uses_actual_visual_stl_vertices_not_proxy_socket_size(tmp_path):
    config = load_config()
    sim = Simulation(build_scene(config, tmp_path / 'scene.xml'), config, render=False)
    try:
        vertices = mesh_world_vertices(sim.model, sim.data, 'source_holder_visual')
        assert len(vertices) > 1000
        np.testing.assert_allclose(np.ptp(vertices, axis=0), [.025, .025, .040], atol=1e-7)
        assert np.ptp(vertices[:, 0]) != pytest.approx(2 * config['board']['socket_half_width'])
        result = head_visibility(sim)
        assert result['all_inside']
        assert all(obj['edge_margin_px'] >= 5 for obj in result['objects'].values())
    finally:
        sim.close()


def test_mesh_is_rejected_when_centre_visible_but_outer_surface_leaves_frame(tmp_path):
    config = load_config()
    config['board']['source_xy'] = [.45, -.20]
    sim = Simulation(build_scene(config, tmp_path / 'scene.xml'), config, render=False)
    try:
        camera = camera_metadata(sim.model, sim.data, 256, 256)['head']
        pose, matrix = np.asarray(camera['world_from_camera_optical_at_capture']), np.asarray(camera['intrinsics'])
        centre = np.array([.45, -.20, .02])
        optical = (centre - pose[:3, 3]) @ pose[:3, :3]
        pixel = matrix @ optical
        pixel = pixel[:2] / pixel[2]
        assert np.all(pixel > 0) and np.all(pixel < 255)
        result = head_visibility(sim)
        assert not result['all_inside']
        assert not result['objects']['source_holder_visual']['inside']
    finally:
        sim.close()


@pytest.mark.parametrize('spacing', [0, -.02, float('nan')])
def test_grid_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        next(candidate_layouts(load_config(), grid_spacing_m=spacing))
