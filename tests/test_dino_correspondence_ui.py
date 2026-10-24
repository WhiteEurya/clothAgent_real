"""Headless UI state tests; real desktop/model smoke is run separately."""
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture
def app():
    pytest.importorskip('tkinter')
    from cloth_agent.dino_correspondence_ui import App
    app = App.__new__(App)
    app.busy = app.closing = False
    app.record = None
    app.target_point = None
    app.point = [100, 100]
    app.vertices = []; app.regions = []
    app.mode = Value('blind'); app.visibility = Value('visible')
    app.annotator = Value('test')
    app.status = Value(''); app.save_status = Value('')
    app.messages = queue.Queue()
    app.root = SimpleNamespace(after=Mock(), destroy=Mock())
    app.render = Mock()
    app.writer = SimpleNamespace(pending={}, failed={}, poll=Mock(return_value=[]), shutdown=Mock())
    return app


def test_exact_point_reveal_and_label_lock(app):
    app.run_query = Mock()
    app.click_target([130, 140])
    assert app.target_point == [130, 140] and not app.vertices
    app.reveal()
    label = app.run_query.call_args.args[0]
    assert label['target_xy'] == [130, 140] and label['regions'] == []
    app.record = {'label': label}
    app.click_target([200, 200]); app.clear_target()
    assert app.target_point == [130, 140]
    app.record = None
    app.visibility.set('invisible'); app.visibility_changed()
    assert app.target_point is None
    app.mode.set('explore'); app.visibility.set('visible')
    app.click_target([200, 200])
    assert app.target_point is None


def test_close_waits_for_saves_and_keeps_failed_results_available(app, monkeypatch):
    from cloth_agent import dino_correspondence_ui as ui
    error = Mock(); monkeypatch.setattr(ui.messagebox, 'showerror', error)
    app.writer.pending = {'query': object()}
    app.close(); app.poll()
    app.root.destroy.assert_not_called()
    app.writer.shutdown.assert_not_called()
    app.writer.pending = {}; app.writer.failed = {'query': object()}
    app.poll()
    assert not app.closing
    error.assert_called_once()
    app.root.destroy.assert_not_called()
    app.writer.failed = {}
    app.close(); app.poll()
    app.writer.shutdown.assert_called_once()
    app.root.destroy.assert_called_once()


def test_explore_clicks_either_side_and_blind_b_stays_annotation(app):
    app.experiment = SimpleNamespace(pair=[object(), object()])
    app.run_query = Mock()
    app.mode.set('explore')
    app.click_b([210, 220])
    assert app.direction == 'B_to_A' and app.point == [210, 220]
    assert app.target_point is None and app.vertices == []
    app.click_a([110, 120])
    assert app.direction == 'A_to_B' and app.point == [110, 120]
    assert app.run_query.call_count == 2
    app.mode.set('blind')
    app.click_b([10, 20])
    assert app.vertices == [[10, 20]] and app.point == [110, 120]
    assert app.run_query.call_count == 2
    app.mode.set('explore'); app.busy = True
    app.click_b([310, 320])
    assert app.direction == 'A_to_B' and app.run_query.call_count == 2


def test_reverse_render_keeps_panels_fixed_and_heatmap_on_a(app, monkeypatch):
    import numpy as np
    from PIL import Image
    from cloth_agent import dino_correspondence_ui as ui
    from cloth_agent.dino_correspondence import image_metadata
    a, b = Image.new('RGB', (800, 400), 'white'), Image.new('RGB', (300, 600), 'black')
    ma, mb = image_metadata(a.size), image_metadata(b.size)
    app.experiment = SimpleNamespace(pair=[dict(image=a, meta=ma), dict(image=b, meta=mb)])
    app.a = SimpleNamespace(set=Mock()); app.b = SimpleNamespace(set=Mock())
    app.heat = Value(True); app.scores = np.zeros((32, 32))
    app.record = dict(direction='B_to_A', result={}, label={}, images=[dict(meta=mb), dict(meta=ma)])
    heat = Mock(side_effect=lambda image, scores, meta: image)
    draw = Mock(side_effect=lambda image, *args, **kwargs: image)
    monkeypatch.setattr(ui, 'heatmap_overlay', heat); monkeypatch.setattr(ui, 'draw_result', draw)
    ui.App.render(app)
    assert heat.call_args.args[0].size == a.size and heat.call_args.args[2] == ma
    assert draw.call_args_list[0].args[0].size == b.size
    assert draw.call_args_list[0].kwargs == dict(source=True)
    assert app.a.set.call_args.args[0].size == a.size
    assert app.b.set.call_args.args[0].size == b.size
