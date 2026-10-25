from facerec.detector import clamp_box


def test_box_inside_image_is_unchanged():
    assert clamp_box((10, 20, 50, 60), width=100, height=100) == (10, 20, 50, 60)


def test_box_is_clipped_to_image_bounds():
    assert clamp_box((-5, -10, 120, 130), width=100, height=80) == (0, 0, 100, 80)


def test_box_outside_image_is_dropped():
    assert clamp_box((110, 10, 150, 50), width=100, height=100) is None


def test_degenerate_box_is_dropped():
    assert clamp_box((30, 30, 30, 60), width=100, height=100) is None
