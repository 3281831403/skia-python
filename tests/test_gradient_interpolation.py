import numpy as np
import pytest

import skia


W = 32
H = 32

# Opaque colors for interpolation-space / hue-direction tests.
COLORS = [0xFFFF0000, 0xFF0000FF]
# Colors with different alpha so premultiplied interpolation is observable.
ALPHA_COLORS = [0xFFFF0000, 0x4000FF00]


def _render(shader):
    surface = skia.Surface(W, H)
    canvas = surface.getCanvas()
    paint = skia.Paint()
    paint.setAntiAlias(False)
    paint.setShader(shader)
    canvas.drawRect(skia.Rect.MakeWH(W, H), paint)
    return np.asarray(surface.toarray())


def _color4f(colors):
    return [skia.Color4f.FromColor(c) for c in colors]


def _make(kind, colors, flags=0, interpolation=None):
    """Build a gradient shader.

    ``flags`` selects the legacy integer-color overload; ``interpolation``
    selects the new ``Color4f`` overload.
    """
    if kind == 'linear':
        pts = [skia.Point(0, 0), skia.Point(W, H)]
        if interpolation is None:
            return skia.GradientShader.MakeLinear(pts, colors, flags=flags)
        return skia.GradientShader.MakeLinear(
            pts, colors, interpolation=interpolation)
    if kind == 'radial':
        center, radius = skia.Point(W / 2, H / 2), W / 2
        if interpolation is None:
            return skia.GradientShader.MakeRadial(center, radius, colors, flags=flags)
        return skia.GradientShader.MakeRadial(
            center, radius, colors, interpolation=interpolation)
    if kind == 'conical':
        if interpolation is None:
            return skia.GradientShader.MakeTwoPointConical(
                skia.Point(0, 0), 0, skia.Point(W, H), W, colors, flags=flags)
        return skia.GradientShader.MakeTwoPointConical(
            skia.Point(0, 0), 0, skia.Point(W, H), W, colors,
            interpolation=interpolation)
    if kind == 'sweep':
        if interpolation is None:
            return skia.GradientShader.MakeSweep(
                W / 2, H / 2, colors, flags=flags)
        return skia.GradientShader.MakeSweep(
            W / 2, H / 2, colors, interpolation=interpolation)
    raise ValueError(kind)


GRADIENT_KINDS = ['linear', 'radial', 'conical', 'sweep']


@pytest.mark.parametrize('kind', GRADIENT_KINDS)
def test_default_interpolation_matches_legacy(kind):
    legacy = _render(_make(kind, COLORS))
    modern = _render(_make(kind, _color4f(COLORS)))
    assert np.array_equal(legacy, modern)


@pytest.mark.parametrize('kind', GRADIENT_KINDS)
def test_premul_interpolation_matches_legacy_flag(kind):
    flag = skia.GradientShader.Flags.kInterpolateColorsInPremul_Flag
    legacy = _render(_make(kind, ALPHA_COLORS, flags=flag))

    interpolation = skia.GradientShader.Interpolation()
    interpolation.fInPremul = skia.GradientShader.Interpolation.InPremul.kYes
    modern = _render(_make(kind, _color4f(ALPHA_COLORS), interpolation=interpolation))

    assert np.array_equal(legacy, modern)


@pytest.mark.parametrize('kind', GRADIENT_KINDS)
def test_premul_interpolation_differs_from_unpremul(kind):
    unpremul = _render(_make(kind, ALPHA_COLORS))

    interpolation = skia.GradientShader.Interpolation()
    interpolation.fInPremul = skia.GradientShader.Interpolation.InPremul.kYes
    premul = _render(_make(kind, _color4f(ALPHA_COLORS), interpolation=interpolation))

    assert not np.array_equal(unpremul, premul)


@pytest.mark.parametrize('kind', GRADIENT_KINDS)
def test_interpolation_color_space_changes_pixels(kind):
    default = _render(_make(kind, _color4f(COLORS)))

    interpolation = skia.GradientShader.Interpolation()
    interpolation.fColorSpace = skia.GradientShader.Interpolation.ColorSpace.kOKLab
    oklab = _render(_make(kind, _color4f(COLORS), interpolation=interpolation))

    assert not np.array_equal(default, oklab)


@pytest.mark.parametrize('kind', GRADIENT_KINDS)
def test_hue_method_changes_pixels(kind):
    shorter = skia.GradientShader.Interpolation()
    shorter.fColorSpace = skia.GradientShader.Interpolation.ColorSpace.kOKLCH
    shorter.fHueMethod = skia.GradientShader.Interpolation.HueMethod.kShorter
    longer = skia.GradientShader.Interpolation()
    longer.fColorSpace = skia.GradientShader.Interpolation.ColorSpace.kOKLCH
    longer.fHueMethod = skia.GradientShader.Interpolation.HueMethod.kLonger

    shorter_img = _render(_make(kind, _color4f(COLORS), interpolation=shorter))
    longer_img = _render(_make(kind, _color4f(COLORS), interpolation=longer))

    assert not np.array_equal(shorter_img, longer_img)


def test_interpolation_defaults_are_destination_unpremul_shorter():
    interpolation = skia.GradientShader.Interpolation()
    assert interpolation.fColorSpace == (
        skia.GradientShader.Interpolation.ColorSpace.kDestination)
    assert interpolation.fInPremul == (
        skia.GradientShader.Interpolation.InPremul.kNo)
    assert interpolation.fHueMethod == (
        skia.GradientShader.Interpolation.HueMethod.kShorter)


def test_interpolation_from_flags():
    interpolation = skia.GradientShader.Interpolation.FromFlags(
        skia.GradientShader.Flags.kInterpolateColorsInPremul_Flag)
    assert interpolation.fInPremul == (
        skia.GradientShader.Interpolation.InPremul.kYes)
    assert interpolation.fColorSpace == (
        skia.GradientShader.Interpolation.ColorSpace.kDestination)


def test_positions_count_must_match_colors():
    with pytest.raises(ValueError):
        skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(W, H)],
            [0xFFFF0000, 0xFF00FF00, 0xFF0000FF], [0, 1])


def test_positions_must_be_strictly_increasing():
    with pytest.raises(ValueError):
        skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(W, H)],
            [0xFFFF0000, 0xFF00FF00], [0.5, 0.5])
    with pytest.raises(ValueError):
        skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(W, H)],
            [0xFFFF0000, 0xFF00FF00], [0.5, 0.2])


def test_positions_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(W, H)],
            [0xFFFF0000, 0xFF00FF00], [-0.1, 1.0])
    with pytest.raises(ValueError):
        skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(W, H)],
            [0xFFFF0000, 0xFF00FF00], [0.0, 1.1])


def test_empty_positions_equals_none():
    points = [skia.Point(0, 0), skia.Point(W, H)]
    colors = [0xFFFF0000, 0xFF00FF00]
    empty = _render(skia.GradientShader.MakeLinear(points, colors, []))
    none = _render(skia.GradientShader.MakeLinear(points, colors, None))
    assert np.array_equal(empty, none)
