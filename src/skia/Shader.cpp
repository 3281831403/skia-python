#include "common.h"
#include <include/effects/SkGradientShader.h>
#include <include/effects/SkPerlinNoiseShader.h>
#include <include/effects/SkBlenders.h>
#include <pybind11/stl.h>

namespace {

// Returns nullptr for a uniform distribution (None or an empty list), otherwise
// returns a pointer to validated color stops. The stops are moved into
// ``storage`` so the returned pointer stays valid for the enclosing factory
// call.
const SkScalar* get_gradient_positions(
    py::object pos, size_t colorCount, std::vector<SkScalar>& storage) {
    if (pos.is_none())
        return nullptr;
    auto stops = pos.cast<std::vector<SkScalar>>();
    if (stops.empty())
        return nullptr;
    if (stops.size() != colorCount)
        throw py::value_error(
            "positions must have the same number of elements as colors.");
    for (size_t i = 0; i < stops.size(); ++i) {
        if (!(stops[i] >= 0 && stops[i] <= 1))
            throw py::value_error("positions must be between 0 and 1.");
    }
    for (size_t i = 1; i < stops.size(); ++i) {
        if (!(stops[i] > stops[i - 1]))
            throw py::value_error("positions must be strictly increasing.");
    }
    storage = std::move(stops);
    return storage.data();
}

}  // namespace

void initShader(py::module &m) {
py::class_<SkShader, sk_sp<SkShader>, SkFlattenable> shader(
    m, "Shader", R"docstring(
    Shaders specify the source color(s) for what is being drawn.

    If a paint has no shader, then the paint's color is used. If the paint has a
    shader, then the shader's color(s) are use instead, but they are modulated
    by the paint's alpha. This makes it easy to create a shader once (e.g.
    bitmap tiling or gradient) and then change its transparency w/o having to
    modify the original shader... only the paint's alpha needs to be modified.

    .. rubric:: Subclasses

    .. autosummary::
        :nosignatures:

        ~skia.Shaders
        ~skia.GradientShader
        ~skia.PerlinNoiseShader
    )docstring");

/*
py::class_<SkShader::GradientInfo>(shader, "GradientInfo")
    .def(py::init<>())
    .def_readwrite("fColorCount", &SkShader::GradientInfo::fColorCount,
        R"docstring(
        In-out parameter, specifies passed size.
        )docstring")
    .def_property_readonly("fColors",
        [] (const SkShader::GradientInfo& info) {
            return std::vector<SkColor>(
                info.fColors, info.fColors + info.fColorCount);
        },
        R"docstring(
        The colors in the gradient.
        )docstring")
    .def_property_readonly("fColorOffsets",
        [] (const SkShader::GradientInfo& info) {
            return std::vector<SkColor>(
                info.fColorOffsets, info.fColorOffsets + info.fColorCount);
        },
        R"docstring(
        The unit offset for color transitions.
        )docstring")
    .def_property_readonly("fPoint",
        [] (const SkShader::GradientInfo& info) {
            return py::make_tuple(info.fPoint[0], info.fPoint[1]);
        },
        R"docstring(
        Type specific, see above.
        )docstring")
    .def_property_readonly("fRadius",
        [] (const SkShader::GradientInfo& info) {
            return py::make_tuple(info.fRadius[0], info.fRadius[1]);
        },
        R"docstring(
        Type specific, see above.
        )docstring")
    .def_readonly("fTileMode", &SkShader::GradientInfo::fTileMode)
    .def_readonly("fGradientFlags", &SkShader::GradientInfo::fGradientFlags,
        R"docstring(
        see :py:class:`GradientShader.Flags`
        )docstring")
    ;

py::enum_<SkShader::GradientType>(shader, "GradientType", R"docstring(
    If the shader subclass can be represented as a gradient, asAGradient returns
    the matching GradientType enum (or kNone_GradientType if it cannot).

    Also, if info is not null, asAGradient populates info with the relevant (see
    below) parameters for the gradient. fColorCount is both an input and output
    parameter. On input, it indicates how many entries in fColors and
    fColorOffsets can be used, if they are non-NULL. After asAGradient has run,
    fColorCount indicates how many color-offset pairs there are in the gradient.
    If there is insufficient space to store all of the color-offset pairs,
    fColors and fColorOffsets will not be altered. fColorOffsets specifies where
    on the range of 0 to 1 to transition to the given color. The meaning of
    fPoint and fRadius is dependant on the type of gradient.

    - None: info is ignored. Color: fColorOffsets[0] is meaningless.
    - Linear: fPoint[0] and fPoint[1] are the end-points of the gradient
    - Radial: fPoint[0] and fRadius[0] are the center and radius
    - Conical: fPoint[0] and fRadius[0] are the center and radius of the 1st
        circle fPoint[1] and fRadius[1] are the center and radius of the 2nd
        circle Sweep: fPoint[0] is the center of the sweep.
    )docstring")
    .value("kNone_GradientType",
        SkShader::GradientType::kNone_GradientType)
    .value("kColor_GradientType",
        SkShader::GradientType::kColor_GradientType)
    .value("kLinear_GradientType",
        SkShader::GradientType::kLinear_GradientType)
    .value("kRadial_GradientType",
        SkShader::GradientType::kRadial_GradientType)
    .value("kSweep_GradientType",
        SkShader::GradientType::kSweep_GradientType)
    .value("kConical_GradientType",
        SkShader::GradientType::kConical_GradientType)
    .value("kLast_GradientType",
        SkShader::GradientType::kLast_GradientType)
    .export_values();
*/

shader
    .def("isOpaque", &SkShader::isOpaque,
        R"docstring(
        Returns true if the shader is guaranteed to produce only opaque colors,
        subject to the :py:class:`Paint` using the shader to apply an opaque
        alpha value.

        Subclasses should override this to allow some optimizations.
        )docstring")
    .def("isAImage",
        [] (const SkShader& shader, SkMatrix* localMatrix,
            std::vector<SkTileMode>& xy) {
            if (xy.size() != 2)
                throw std::runtime_error("xy must have two elements.");
            return sk_sp<SkImage>(shader.isAImage(localMatrix, &xy[0]));
        },
        R"docstring(
        Iff this shader is backed by a single :py:class:`Image`, return its ptr
        (the caller must ref this if they want to keep it longer than the
        lifetime of the shader).

        If not, return nullptr.
        )docstring",
        py::arg("localMatrix"), py::arg("xy") = nullptr)
    .def("isAImage", py::overload_cast<>(&SkShader::isAImage, py::const_))
/*
    .def("asAGradient", &SkShader::asAGradient, py::arg("info"))
*/
    .def("makeWithLocalMatrix", &SkShader::makeWithLocalMatrix,
        R"docstring(
        Return a shader that will apply the specified localMatrix to this
        shader.

        The specified matrix will be applied before any matrix associated with
        this shader.
        )docstring",
        py::arg("matrix"))
    .def("makeWithColorFilter", &SkShader::makeWithColorFilter,
        R"docstring(
        Create a new shader that produces the same colors as invoking this
        shader and then applying the colorfilter.
        )docstring",
        py::arg("colorFilter"))
/*
    .def_static("Deserialize",
        [] (py::buffer b) {
            auto info = b.request();
            auto shader = SkShader::Deserialize(
                SkFlattenable::Type::kSkShaderBase_Type, info.ptr,
                info.shape[0] * info.strides[0]);
            return sk_sp<SkShader>(
                reinterpret_cast<SkShader*>(shader.release()));
        },
        py::arg("data"))
*/
    ;

py::class_<std::unique_ptr<int32_t>>(m, "Shaders")
    .def_static("Empty", &SkShaders::Empty)
    .def_static("Color", py::overload_cast<SkColor>(&SkShaders::Color),
        py::arg("color"))
    .def_static("Color",
        [] (const SkColor4f& color, const SkColorSpace& cs) {
            return SkShaders::Color(color, CloneFlattenable(cs));
        },
        py::arg("color"), py::arg("cs") = nullptr)
    .def_static("Blend",
        [] (SkBlendMode mode, const SkShader& dst,
            const SkShader& src) {
            return SkShaders::Blend(
                mode, CloneFlattenable(dst), CloneFlattenable(src));
        },
        py::arg("mode"), py::arg("dst"), py::arg("src"))
    .def_static("Blend",
        [] (sk_sp<SkBlender> blender, sk_sp<SkShader> dst,
            sk_sp<SkShader> src) {
            return SkShaders::Blend(
                blender, dst, src);
        },
        py::arg("blender"), py::arg("dst"), py::arg("src"))
    .def_static("Lerp",
        [] (SkScalar t, const SkShader& dst,
            const SkShader& src) {
            return SkShaders::Blend(
                SkBlenders::Arithmetic(0, t, 1-t, 0, false),
                CloneFlattenable(dst), CloneFlattenable(src));
        },
        py::arg("t"), py::arg("dst"), py::arg("src"))
    ;

py::class_<SkGradientShader> gradientshader(m, "GradientShader");

py::enum_<SkGradientShader::Flags>(gradientshader, "Flags", py::arithmetic())
    .value("kInterpolateColorsInPremul_Flag",
        SkGradientShader::Flags::kInterpolateColorsInPremul_Flag)
    .export_values();

py::class_<SkGradientShader::Interpolation> interpolation(
    gradientshader, "Interpolation", R"docstring(
    Controls how gradient colors are interpolated between color stops.

    The default values reproduce the legacy integer-color gradient behavior:
    interpolate in the destination color space, unpremultiplied, using the
    shorter hue arc.
    )docstring");

py::enum_<SkGradientShader::Interpolation::InPremul>(
    interpolation, "InPremul", R"docstring(
    Whether colors are premultiplied before interpolation.

    This mirrors
    :py:attr:`GradientShader.Flags.kInterpolateColorsInPremul_Flag`.
    )docstring")
    .value("kNo", SkGradientShader::Interpolation::InPremul::kNo)
    .value("kYes", SkGradientShader::Interpolation::InPremul::kYes)
    .export_values();

py::enum_<SkGradientShader::Interpolation::ColorSpace>(
    interpolation, "ColorSpace", R"docstring(
    The color space in which the gradient colors are interpolated.
    )docstring")
    .value("kDestination",
        SkGradientShader::Interpolation::ColorSpace::kDestination)
    .value("kSRGBLinear",
        SkGradientShader::Interpolation::ColorSpace::kSRGBLinear)
    .value("kLab", SkGradientShader::Interpolation::ColorSpace::kLab)
    .value("kOKLab", SkGradientShader::Interpolation::ColorSpace::kOKLab)
    .value("kOKLabGamutMap",
        SkGradientShader::Interpolation::ColorSpace::kOKLabGamutMap)
    .value("kLCH", SkGradientShader::Interpolation::ColorSpace::kLCH)
    .value("kOKLCH", SkGradientShader::Interpolation::ColorSpace::kOKLCH)
    .value("kOKLCHGamutMap",
        SkGradientShader::Interpolation::ColorSpace::kOKLCHGamutMap)
    .value("kSRGB", SkGradientShader::Interpolation::ColorSpace::kSRGB)
    .value("kHSL", SkGradientShader::Interpolation::ColorSpace::kHSL)
    .value("kHWB", SkGradientShader::Interpolation::ColorSpace::kHWB)
    .value("kDisplayP3",
        SkGradientShader::Interpolation::ColorSpace::kDisplayP3)
    .value("kRec2020",
        SkGradientShader::Interpolation::ColorSpace::kRec2020)
    .value("kProphotoRGB",
        SkGradientShader::Interpolation::ColorSpace::kProphotoRGB)
    .value("kA98RGB", SkGradientShader::Interpolation::ColorSpace::kA98RGB)
    .export_values();

py::enum_<SkGradientShader::Interpolation::HueMethod>(
    interpolation, "HueMethod", R"docstring(
    The direction of hue interpolation, only relevant for polar color spaces
    (LCH, OKLCH, HSL, or HWB).
    )docstring")
    .value("kShorter", SkGradientShader::Interpolation::HueMethod::kShorter)
    .value("kLonger", SkGradientShader::Interpolation::HueMethod::kLonger)
    .value("kIncreasing",
        SkGradientShader::Interpolation::HueMethod::kIncreasing)
    .value("kDecreasing",
        SkGradientShader::Interpolation::HueMethod::kDecreasing)
    .export_values();

interpolation
    .def(py::init<>())
    .def_readwrite("fInPremul",
        &SkGradientShader::Interpolation::fInPremul,
        R"docstring(
        Whether to premultiply the colors before interpolation.
        )docstring")
    .def_readwrite("fColorSpace",
        &SkGradientShader::Interpolation::fColorSpace,
        R"docstring(
        The color space in which to interpolate.
        )docstring")
    .def_readwrite("fHueMethod",
        &SkGradientShader::Interpolation::fHueMethod,
        R"docstring(
        The direction in which to interpolate hue.
        )docstring")
    .def_static("FromFlags", &SkGradientShader::Interpolation::FromFlags,
        R"docstring(
        Returns the interpolation corresponding to the legacy ``flags`` mask.
        )docstring", py::arg("flags"))
    ;

gradientshader
    .def_static("MakeLinear",
        [] (const std::vector<SkPoint>& pts,
            const std::vector<SkColor4f>& colors, py::object pos,
            SkTileMode mode, const SkColorSpace* colorSpace,
            const SkGradientShader::Interpolation& interpolation,
            const SkMatrix* localMatrix) {
            if (pts.size() != 2)
                throw std::runtime_error("pts must have two elements.");
            if (colors.size() < 2)
                throw std::runtime_error("length of colors must be 2 or more.");
            std::vector<SkScalar> positions;
            const SkScalar* posPtr = get_gradient_positions(
                pos, colors.size(), positions);
            return SkGradientShader::MakeLinear(
                &pts[0], &colors[0], CloneColorSpace(colorSpace), posPtr,
                colors.size(), mode, interpolation, localMatrix);
        },
        R"docstring(
        Returns a shader that generates a linear gradient between the two
        specified points, interpolating the colors in the given color space.

        :param List[skia.Point] points: The start and end points for the
            gradient.
        :param List[skia.Color4f] colors: The array of colors, to be distributed
            between the two points.
        :param List[float] positions: May be None or an empty list. The array of
            Scalars of the relative position of each corresponding color in the
            colors array. If this is None or empty, the colors are distributed
            evenly between the start and end point. Otherwise the values must
            lie between 0 and 1 and be strictly increasing, and must have the
            same number of elements as ``colors``.
        :param skia.TileMode mode: The tiling mode
        :param skia.ColorSpace colorSpace: The color space of ``colors``
        :param GradientShader.Interpolation interpolation: Interpolation options
        :param localMatrix: Local matrix
        )docstring",
        py::arg("points"), py::arg("colors"), py::arg("positions") = nullptr,
        py::arg_v("mode", SkTileMode::kClamp, "skia.TileMode.kClamp"),
        py::arg("colorSpace") = nullptr,
        py::arg("interpolation") = SkGradientShader::Interpolation{},
        py::arg("localMatrix") = nullptr)
    .def_static("MakeLinear",
        [] (const std::vector<SkPoint>& pts, const std::vector<SkColor>& colors,
            py::object pos, SkTileMode mode, uint32_t flags,
            const SkMatrix* localMatrix) {
            if (pts.size() != 2)
                throw std::runtime_error("pts must have two elements.");
            if (colors.size() < 2)
                throw std::runtime_error("length of colors must be 2 or more.");
            std::vector<SkScalar> positions;
            const SkScalar* posPtr = get_gradient_positions(
                pos, colors.size(), positions);
            return SkGradientShader::MakeLinear(
                &pts[0], &colors[0], posPtr, colors.size(), mode,
                flags, localMatrix);
        },
        R"docstring(
        Returns a shader that generates a linear gradient between the two
        specified points.

        :param List[skia.Point] points: The start and end points for the
            gradient.
        :param List[int] colors: The array of colors, to be distributed
            between the two points
        :param List[float] positions: May be None or an empty list. The array of
            Scalars of the relative position of each corresponding color in the
            colors array. If this is None or empty, the colors are distributed
            evenly between the start and end point. Otherwise the values must
            lie between 0 and 1 and be strictly increasing, and must have the
            same number of elements as ``colors``.
        :param skia.TileMode mode: The tiling mode
        :param flags: Legacy gradient flags, e.g.
            :py:attr:`GradientShader.Flags.kInterpolateColorsInPremul_Flag`.
        :param localMatrix: Local matrix
        )docstring",
        py::arg("points"), py::arg("colors"), py::arg("positions") = nullptr,
        py::arg_v("mode", SkTileMode::kClamp, "skia.TileMode.kClamp"),
        py::arg("flags") = 0,
        py::arg("localMatrix") = nullptr)
    .def_static("MakeRadial",
        [] (const SkPoint& center, SkScalar radius,
            const std::vector<SkColor4f>& colors, py::object pos,
            SkTileMode mode, const SkColorSpace* colorSpace,
            const SkGradientShader::Interpolation& interpolation,
            const SkMatrix* localMatrix) {
            if (colors.size() < 2)
                throw std::runtime_error("length of colors must be 2 or more.");
            std::vector<SkScalar> positions;
            const SkScalar* posPtr = get_gradient_positions(
                pos, colors.size(), positions);
            return SkGradientShader::MakeRadial(
                center, radius, &colors[0], CloneColorSpace(colorSpace), posPtr,
                colors.size(), mode, interpolation, localMatrix);
        },
        R"docstring(
        Returns a shader that generates a radial gradient given the center and
        radius, interpolating the colors in the given color space.

        :param skia.Point center:  The center of the circle for this gradient
        :param float radius:  Must be positive. The radius of the circle for
            this gradient
        :param List[skia.Color4f] colors:  The array of colors, to be
            distributed between the center and edge of the circle
        :param List[float] positions: May be None or an empty list. The array of
            Scalars of the relative position of each corresponding color in the
            colors array. If this is None or empty, the colors are distributed
            evenly between the center and edge of the circle. Otherwise the
            values must lie between 0 and 1 and be strictly increasing, and must
            have the same number of elements as ``colors``.
        :param skia.TileMode mode: The tiling mode
        :param skia.ColorSpace colorSpace: The color space of ``colors``
        :param GradientShader.Interpolation interpolation: Interpolation options
        :param localMatrix: Local matrix
        )docstring",
        py::arg("center"), py::arg("radius"), py::arg("colors"),
        py::arg("positions") = nullptr,
        py::arg_v("mode", SkTileMode::kClamp, "skia.TileMode.kClamp"),
        py::arg("colorSpace") = nullptr,
        py::arg("interpolation") = SkGradientShader::Interpolation{},
        py::arg("localMatrix") = nullptr)
    .def_static("MakeRadial",
        [] (const SkPoint& center, SkScalar radius,
            const std::vector<SkColor>& colors,
            py::object pos, SkTileMode mode, uint32_t flags,
            const SkMatrix* localMatrix) {
            if (colors.size() < 2)
                throw std::runtime_error("length of colors must be 2 or more.");
            std::vector<SkScalar> positions;
            const SkScalar* posPtr = get_gradient_positions(
                pos, colors.size(), positions);
            return SkGradientShader::MakeRadial(
                center, radius, &colors[0], posPtr,
                colors.size(), mode, flags, localMatrix);
        },
        R"docstring(
        Returns a shader that generates a radial gradient given the center and
        radius.

        :param skia.Point center:  The center of the circle for this gradient
        :param float radius:  Must be positive. The radius of the circle for
            this gradient
        :param List[int] colors:  The array of colors, to be distributed between
            the center and edge of the circle
        :param List[float] positions: May be None or an empty list. The array of
            Scalars of the relative position of each corresponding color in the
            colors array. If this is None or empty, the colors are distributed
            evenly between the center and edge of the circle. Otherwise the
            values must lie between 0 and 1 and be strictly increasing, and must
            have the same number of elements as ``colors``.
        :param skia.TileMode mode: The tiling mode
        :param flags: Legacy gradient flags, e.g.
            :py:attr:`GradientShader.Flags.kInterpolateColorsInPremul_Flag`.
        :param localMatrix: Local matrix
        )docstring",
        py::arg("center"), py::arg("radius"), py::arg("colors"),
        py::arg("positions") = nullptr,
        py::arg_v("mode", SkTileMode::kClamp, "skia.TileMode.kClamp"),
        py::arg("flags") = 0, py::arg("localMatrix") = nullptr)
    .def_static("MakeTwoPointConical",
        [] (const SkPoint& start, SkScalar startRadius, const SkPoint& end,
            SkScalar endRadius, const std::vector<SkColor4f>& colors,
            py::object pos, SkTileMode mode, const SkColorSpace* colorSpace,
            const SkGradientShader::Interpolation& interpolation,
            const SkMatrix *localMatrix) {
            if (colors.size() < 2)
                throw std::runtime_error("length of colors must be 2 or more.");
            std::vector<SkScalar> positions;
            const SkScalar* posPtr = get_gradient_positions(
                pos, colors.size(), positions);
            return SkGradientShader::MakeTwoPointConical(
                start, startRadius, end, endRadius, &colors[0],
                CloneColorSpace(colorSpace), posPtr, colors.size(), mode,
                interpolation, localMatrix);
        },
        R"docstring(
        Returns a shader that generates a conical gradient given two circles, or
        returns NULL if the inputs are invalid, interpolating the colors in the
        given color space.

        The gradient interprets the two circles according to the following HTML
        spec.
        http://dev.w3.org/html5/2dcontext/#dom-context-2d-createradialgradient

        :param List[float] positions: May be None or an empty list. The array of
            Scalars of the relative position of each corresponding color in the
            colors array. If this is None or empty, the colors are distributed
            evenly. Otherwise the values must lie between 0 and 1 and be
            strictly increasing, and must have the same number of elements as
            ``colors``.
        :param skia.ColorSpace colorSpace: The color space of ``colors``
        :param GradientShader.Interpolation interpolation: Interpolation options
        )docstring",
        py::arg("start"), py::arg("startRadius"), py::arg("end"),
        py::arg("endRadius"), py::arg("colors"), py::arg("positions") = nullptr,
        py::arg_v("mode", SkTileMode::kClamp, "skia.TileMode.kClamp"),
        py::arg("colorSpace") = nullptr,
        py::arg("interpolation") = SkGradientShader::Interpolation{},
        py::arg("localMatrix") = nullptr)
    .def_static("MakeTwoPointConical",
        [] (const SkPoint& start, SkScalar startRadius, const SkPoint& end,
            SkScalar endRadius, const std::vector<SkColor>& colors,
            py::object pos, SkTileMode mode, uint32_t flags,
            const SkMatrix *localMatrix) {
            if (colors.size() < 2)
                throw std::runtime_error("length of colors must be 2 or more.");
            std::vector<SkScalar> positions;
            const SkScalar* posPtr = get_gradient_positions(
                pos, colors.size(), positions);
            return SkGradientShader::MakeTwoPointConical(
                start, startRadius, end, endRadius, &colors[0],
                posPtr, colors.size(), mode, flags,
                localMatrix);
        },
        R"docstring(
        Returns a shader that generates a conical gradient given two circles, or
        returns NULL if the inputs are invalid.

        The gradient interprets the two circles according to the following HTML
        spec.
        http://dev.w3.org/html5/2dcontext/#dom-context-2d-createradialgradient
        )docstring",
        py::arg("start"), py::arg("startRadius"), py::arg("end"),
        py::arg("endRadius"), py::arg("colors"), py::arg("positions") = nullptr,
        py::arg_v("mode", SkTileMode::kClamp, "skia.TileMode.kClamp"),
        py::arg("flags") = 0,
        py::arg("localMatrix") = nullptr)
    .def_static("MakeSweep",
        [] (SkScalar cx, SkScalar cy, const std::vector<SkColor4f>& colors,
            py::object pos, SkTileMode mode, SkScalar startAngle,
            SkScalar endAngle, const SkColorSpace* colorSpace,
            const SkGradientShader::Interpolation& interpolation,
            const SkMatrix *localMatrix) {
            if (colors.size() < 2)
                throw std::runtime_error("length of colors must be 2 or more.");
            std::vector<SkScalar> positions;
            const SkScalar* posPtr = get_gradient_positions(
                pos, colors.size(), positions);
            return SkGradientShader::MakeSweep(
                cx, cy, &colors[0], CloneColorSpace(colorSpace), posPtr,
                colors.size(), mode, startAngle, endAngle, interpolation,
                localMatrix);
        },
        R"docstring(
        Returns a shader that generates a sweep gradient given a center,
        interpolating the colors in the given color space.

        The shader accepts negative angles and angles larger than 360, draws
        between 0 and 360 degrees, similar to the CSS conic-gradient semantics.

        :param List[skia.Color4f] colors: The array of colors, to be distributed
            around the center, within the gradient angle range.
        :param List[float] positions: May be None or an empty list. The array of
            Scalars of the relative position of each corresponding color in the
            colors array. If this is None or empty, the colors are distributed
            evenly within the angular range. Otherwise the values must lie
            between 0 and 1 and be strictly increasing, and must have the same
            number of elements as ``colors``.
        :param skia.ColorSpace colorSpace: The color space of ``colors``
        :param GradientShader.Interpolation interpolation: Interpolation options
        :param localMatrix: Local matrix
        )docstring",
        py::arg("cx"), py::arg("cy"), py::arg("colors"),
        py::arg("positions") = nullptr,
        py::arg_v("mode", SkTileMode::kClamp, "skia.TileMode.kClamp"),
        py::arg("startAngle") = 0, py::arg("endAngle") = 360,
        py::arg("colorSpace") = nullptr,
        py::arg("interpolation") = SkGradientShader::Interpolation{},
        py::arg("localMatrix") = nullptr)
    .def_static("MakeSweep",
        [] (SkScalar cx, SkScalar cy, const std::vector<SkColor>& colors,
            py::object pos, SkTileMode mode, SkScalar startAngle,
            SkScalar endAngle, uint32_t flags,
            const SkMatrix *localMatrix) {
            if (colors.size() < 2)
                throw std::runtime_error("length of colors must be 2 or more.");
            std::vector<SkScalar> positions;
            const SkScalar* posPtr = get_gradient_positions(
                pos, colors.size(), positions);
            return SkGradientShader::MakeSweep(
                cx, cy, &colors[0], posPtr,
                colors.size(), mode, startAngle, endAngle, flags, localMatrix);
        },
        R"docstring(
        Returns a shader that generates a sweep gradient given a center.

        The shader accepts negative angles and angles larger than 360, draws
        between 0 and 360 degrees, similar to the CSS conic-gradient semantics.
        )docstring",
        py::arg("cx"), py::arg("cy"), py::arg("colors"),
        py::arg("positions") = nullptr,
        py::arg_v("mode", SkTileMode::kClamp, "skia.TileMode.kClamp"),
        py::arg("startAngle") = 0, py::arg("endAngle") = 360,
        py::arg("flags") = 0, py::arg("localMatrix") = nullptr)
    ;

py::class_<std::unique_ptr<uint32_t>>(m, "PerlinNoiseShader",
    R"docstring(
    :py:class:`PerlinNoiseShader` creates an image using the Perlin turbulence
    function.

    It can produce tileable noise if asked to stitch tiles and provided a tile
    size. In order to fill a large area with repeating noise, set the
    stitchTiles flag to true, and render exactly a single tile of noise. Without
    this flag, the result will contain visible seams between tiles.

    The algorithm used is described here:
    http://www.w3.org/TR/SVG/filters.html#feTurbulenceElement
    )docstring")
    .def_static("MakeFractalNoise", &SkShaders::MakeFractalNoise,
        R"docstring(
        This will construct Perlin noise of the given type (Fractal Noise or
        Turbulence).

        Both base frequencies (X and Y) have a usual range of (0..1) and must be
        non-negative.

        The number of octaves provided should be fairly small, with a limit of
        255 enforced. Each octave doubles the frequency, so 10 octaves would
        produce noise from baseFrequency * 1, * 2, * 4, ..., * 512, which
        quickly yields insignificantly small periods and resembles regular
        unstructured noise rather than Perlin noise.

        If tileSize isn't NULL or an empty size, the tileSize parameter will be
        used to modify the frequencies so that the noise will be tileable for
        the given tile size. If tileSize is NULL or an empty size, the
        frequencies will be used as is without modification.
        )docstring",
        py::arg("baseFrequencyX"), py::arg("baseFrequencyY"),
        py::arg("numOctaves"), py::arg("seed"), py::arg("tileSize") = nullptr)
    .def_static("MakeTurbulence", &SkShaders::MakeTurbulence,
        py::arg("baseFrequencyX"), py::arg("baseFrequencyY"),
        py::arg("numOctaves"), py::arg("seed"), py::arg("tileSize") = nullptr)
/*
    .def_static("MakeImprovedNoise", &SkShaders::MakeImprovedNoise,
        R"docstring(
        Creates an Improved Perlin Noise shader.

        The z value is roughly equivalent to the seed of the other two types,
        but minor variations to z will only slightly change the noise.
        )docstring",
        py::arg("baseFrequencyX"), py::arg("baseFrequencyY"),
        py::arg("numOctaves"), py::arg("z"))
*/
    ;
}
