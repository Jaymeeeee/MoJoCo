"""The deployment entry point resolves model/layout overrides without importing ML."""
import sys
from types import SimpleNamespace

import pytest

from examples.insert_plug import cli


@pytest.fixture
def runner(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'examples.insert_plug.rollout',
                        SimpleNamespace(run_rollouts=lambda **kw: calls.append(kw) or {'success': False}))
    monkeypatch.setattr(cli, '_configure_gl', lambda backend: backend)
    return calls


def test_cli_overrides_model_layout_and_action_horizon(runner):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(['rollout', '--checkpoint', 'outputs/custom_model', '--position', '3',
                  '--device', 'cpu', '--n-action-steps', '5', '--duration', '2',
                  '--no-video', '--continue-on-contact'])
    assert exit_info.value.code == 2  # Unsuccessful policy execution is never reported as success.
    call = runner[0]
    assert call['checkpoint'] == cli.ROOT / 'outputs/custom_model'
    assert call['position'] == '3'
    assert call['device'] == 'cpu'
    assert call['n_action_steps'] == 5
    assert call['duration'] == 2
    assert call['video'] is False
    assert call['stop_on_contact'] is False
    assert call['contact_diagnostics'] is False


def test_config_positions_are_available_without_manifest_override(runner):
    with pytest.raises(SystemExit):
        cli.main(['rollout', '--position', 'config'])
    assert runner[0]['position'] is None
    assert runner[0]['config'] == cli.ROOT / 'config/insert_plug.yaml'
    assert runner[0]['stop_on_contact'] is False
    assert runner[0]['contact_diagnostics'] is False


def test_headless_viewer_is_rejected_before_loading_model(runner, monkeypatch):
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.setattr(sys, 'platform', 'linux')
    with pytest.raises(SystemExit) as exit_info:
        cli.main(['rollout', '--viewer'])
    assert exit_info.value.code == 2
    assert runner == []


def test_policy_config_typo_is_rejected_before_execution(runner, tmp_path):
    path = tmp_path / 'policy.yaml'
    path.write_text('checkpoint: local_model\nn_action_step: 100\n')
    with pytest.raises(ValueError, match='Invalid policy configuration'):
        cli.main(['rollout', '--policy-config', str(path)])
    assert runner == []


def test_contact_diagnostics_are_explicit_and_do_not_imply_stopping(runner):
    with pytest.raises(SystemExit):
        cli.main(['rollout', '--contact-diagnostics'])
    assert runner[0]['contact_diagnostics'] is True
    assert runner[0]['stop_on_contact'] is False


def test_strict_contact_stopping_requires_explicit_option(runner):
    with pytest.raises(SystemExit):
        cli.main(['rollout', '--stop-on-contact'])
    assert runner[0]['stop_on_contact'] is True


def test_legacy_continue_flag_overrides_saved_strict_mode(runner, tmp_path):
    path = tmp_path / 'policy.yaml'
    path.write_text('checkpoint: local_model\nstop_on_contact: true\ncontact_diagnostics: true\n')
    with pytest.raises(SystemExit):
        cli.main(['rollout', '--policy-config', str(path), '--continue-on-contact'])
    assert runner[0]['stop_on_contact'] is False
    assert runner[0]['contact_diagnostics'] is True


def test_cli_can_disable_optional_contact_diagnostics_from_yaml(runner, tmp_path):
    path = tmp_path / 'policy.yaml'
    path.write_text('checkpoint: local_model\nstop_on_contact: true\ncontact_diagnostics: true\n')
    with pytest.raises(SystemExit):
        cli.main(['rollout', '--policy-config', str(path),
                  '--no-contact-diagnostics', '--no-stop-on-contact'])
    assert runner[0]['stop_on_contact'] is False
    assert runner[0]['contact_diagnostics'] is False


def test_conflicting_contact_stop_cli_flags_are_rejected(runner):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(['rollout', '--stop-on-contact', '--continue-on-contact'])
    assert exit_info.value.code == 2
    assert runner == []
