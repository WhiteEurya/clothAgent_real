from types import SimpleNamespace

from scripts.plan_zed_fold import planning_assumptions


def test_offline_assumptions_do_not_promote_configured_values_to_measurements():
    config = SimpleNamespace(
        raw={'safety': {'status': 'measured', 'arms': {
            'left': {'stop_excursion_deg': [1.25] * 7},
            'right': {'stop_excursion_deg': [1.25] * 7},
        }}},
        limits={'tracking_error_deg': 0.0},
    )
    report = planning_assumptions(config)
    assert set(report['physical_measurements_established_by_this_run'].values()) == {'UNKNOWN'}
    assert report['configured_model_inputs']['stop_excursion_deg']['right'] == [1.25] * 7
    assert report['configured_model_inputs']['tracking_error_deg'] == 0.0
    assert report['physical_acceptance_inferred'] is False
