"""Keep manual hardware/network diagnostics out of automatic collection."""

collect_ignore = ["manual", "fixtures"]

import shutil
import pytest


@pytest.fixture
def fake_one_shot_view_hardware(monkeypatch):
    """Existing loop tests fake hardware only; retain actual concurrent routing."""
    from cloth_agent import one_shot_views as views
    class Tour:
        def __init__(self, runtime, before, folder, record):
            self.runtime, self.before, self.folder = runtime, before, folder
        def capture(self):
            (self.folder/'right_view').mkdir()
            shutil.copyfile(self.before.image, self.folder/'right_view/rgb.png')
        def return_to_perception(self):
            self.runtime.move_to_perception(self.folder/'view_return_perception')
    monkeypatch.setattr(views, 'view_target', lambda *a: None)
    monkeypatch.setattr(views, 'ViewTour', Tour)
