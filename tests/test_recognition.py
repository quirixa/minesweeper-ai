import numpy as np
import pytest

from solver.board_state import COVERED, FLAG, MINE
from vision.board_detection import detect_grid, geometry_from_manual
from vision.temporal import BoardChangeDetector, StabilityTracker, TemporalFilter, frame_change
from vision.tile_recognition import ProfileMissingError, RecognitionResult, TileRecognizer
from tests.simulator import Game


def played_game(cell=32, seed=3, rows=9, cols=12, mines=12):
    g = Game(rows, cols, mines, seed)
    fresh = g.render(cell=cell)
    g.reveal(rows // 2, cols // 2)
    g.flag(0, 0) if not g.revealed[0, 0] else None
    return g, fresh


def expected_grid(g):
    out = np.full((g.rows, g.cols), COVERED, int)
    for r in range(g.rows):
        for c in range(g.cols):
            if g.revealed[r, c]:
                out[r, c] = MINE if (r, c) in g.mines else g.number(r, c)
            elif g.flags[r, c]:
                out[r, c] = FLAG
    return out


@pytest.mark.parametrize("cell", [24, 32, 40, 56])
def test_grid_detection_at_several_zoom_levels(cell):
    g = Game(9, 12, 10)
    geo = detect_grid(g.render(cell=cell, canvas=(1100, 800)))
    assert geo is not None
    assert (geo.rows, geo.cols) == (9, 12)
    assert abs(geo.cell_w - cell) < 1.5 and abs(geo.x0 - 60) < 3 and abs(geo.y0 - 90) < 3


@pytest.mark.parametrize("cell", [24, 32, 48])
def test_recognition_matches_ground_truth(cell):
    g, fresh = played_game(cell)
    geo = detect_grid(fresh)
    rec = TileRecognizer()
    rec.learn_covered(fresh, geo)
    res = rec.recognize(g.render(cell=cell), geo)
    assert (res.grid == expected_grid(g)).all(), (res.grid, expected_grid(g))
    assert res.min_confidence > 0.5


def test_loss_is_visible_as_mine():
    g, fresh = played_game()
    geo = detect_grid(fresh)
    rec = TileRecognizer(); rec.learn_covered(fresh, geo)
    mine = next(iter(g.mines))
    g.lost = True
    res = rec.recognize(g.render(), geo)
    assert (res.grid == MINE).any()


def test_recognition_requires_calibration():
    g, fresh = played_game()
    with pytest.raises(ProfileMissingError):
        TileRecognizer().recognize(fresh, detect_grid(fresh))


def test_learning_rejects_non_fresh_board():
    g, fresh = played_game()
    geo = detect_grid(fresh)
    with pytest.raises(ProfileMissingError):
        TileRecognizer().learn_covered(g.render(), geo)


def test_noise_lowers_confidence_or_is_flagged():
    g, fresh = played_game()
    geo = detect_grid(fresh)
    rec = TileRecognizer(); rec.learn_covered(fresh, geo)
    img = g.render().copy()
    rng = np.random.default_rng(0)
    noisy = np.clip(img.astype(int) + rng.integers(-90, 90, img.shape), 0, 255).astype(np.uint8)
    res = rec.recognize(noisy, geo)
    assert res.min_confidence < 0.8  # an unsafe-looking board must not pass a 0.8 threshold


def test_non_board_frame_is_not_detected():
    assert detect_grid(np.full((400, 600, 3), 240, np.uint8)) is None


def test_taught_digit_overrides_palette():
    g, fresh = played_game()
    geo = detect_grid(fresh)
    rec = TileRecognizer(); rec.learn_covered(fresh, geo)
    cell = next((r, c) for r in range(g.rows) for c in range(g.cols) if g.revealed[r, c] and g.number(r, c) == 1)
    rec.teach_digit(g.render(), geo, cell, 1)
    assert rec.recognize(g.render(), geo).grid[cell] == 1


def test_temporal_filter_zeroes_unstable_cells():
    f = TemporalFilter(window=3)
    base = RecognitionResult(np.zeros((2, 2), int), np.ones((2, 2)))
    for _ in range(3):
        out = f.add(base)
    assert out.min_confidence == 1.0
    flicker = RecognitionResult(np.array([[1, 0], [0, 0]]), np.ones((2, 2)))
    out = f.add(flicker)
    assert out.confidence[0, 0] == 0.0 and out.confidence[1, 1] == 1.0


def test_board_change_and_stability_detection():
    g = Game(9, 9, 10)
    a = g.render()
    det = BoardChangeDetector()
    assert det.update(a)                 # first frame counts as a change
    assert not det.update(a.copy())      # identical frame: no change
    g.reveal(4, 4)
    b = g.render()
    assert frame_change(a, b) > 0.01 and det.update(b)
    st = StabilityTracker(required=2)
    assert not st.update(a)
    assert not st.update(b)              # changed
    assert st.update(b.copy())           # then still
