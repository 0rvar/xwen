# Z-Image control-map preprocessors

2026-09-08. Phase 5 of [the image-control PRD](../zimage-control-prd.md), plus
phase 4's Canny preprocessor. The pipeline can derive a Canny, depth or whole-body
pose map from an input image. `none` preserves the supplied map exactly.

[`Preprocessor`](../../src/zimage/preprocess.rs) owns lazy CPU model instances.
Images enter and leave as RGB8 tensors at the original dimensions. Missing models
produce an error with the fetch command; rendering never downloads them. Dropping
the preprocessor releases the model state.

## Canny, depth and pose

Canny uses grayscale conversion, Gaussian blur with sigma 1, Sobel gradients,
nonmaximum suppression and eight-neighbor hysteresis at thresholds 100 and 200.
The rectangle fixture checks that the perimeter produces edges and the interior
stays empty. This is a native Canny implementation; it has no pixel-equivalence
claim against OpenCV.

Depth uses the complete trained Depth Anything V2 Small checkpoint, including its
fine-tuned DINO backbone. The image is resized to 518x518 and normalized with the
author's ImageNet RGB mean and standard deviation. The prediction is resized back
with bilinear interpolation, then mapped to grayscale with a guarded min/max
normalization. Non-square inputs are distorted at the model boundary. This square
policy keeps the trained 37x37 position grid; it does not reproduce the author's
aspect-preserving image preprocessor.

The pinned Candle implementations needed four math corrections before they passed
the author reference. The vendored depth head uses bilinear interpolation with
`align_corners=true`, and its final 1x1 convolution has zero padding. The vendored
DINO backbone uses erf GELU and LayerNorm epsilon 1e-6. Candle had used nearest
interpolation, padding one on the final convolution, tanh GELU and epsilon 1e-5.
The unused ImageNet classifier and larger model factories are absent from this
port. Inputs outside the trained square grid are refused.

Pose uses the author's YOLOX-L detector and DWPose 133-point whole-body ONNX model.
The native RGB image is converted to BGR for both networks. Detection uses a
640x640 letterbox; pose inference uses a 288x384 crop with 1.25 bbox padding and
aspect correction. SimCC maxima are divided by two and projected back to image
coordinates. The neck requires both shoulders above 0.3 confidence, then the
points are reordered into OpenPose's body, face and hand layout. As in the author
code, an empty detector result falls back to the full image, and the renderer
omits foot keypoints.

The body colors, hand colors and face dots follow the author renderer. Native
bilinear rounding, ellipse fills and line rasterization approximate OpenCV, so
the rendered map is not bitwise identical. The fixture checks both coordinates
and foreground overlap.

## Reference checks

All eight preprocessor tests passed, including the three ignored tests that load
the real cached models. The committed reference uses the existing 512x512
fisherman image from the transformer fixture.

| Comparison | Result | Gate |
| --- | --- | --- |
| Raw depth, shared normalized 518x518 input | relative L2 7.9962e-7 | < 0.001 |
| Visible pose keypoints | maximum distance 0.9143 px | < 3 px |
| Pose confidence | maximum difference 0.006846 | < 0.1 |
| Rendered pose foreground | intersection over union 0.8493 | > 0.7 |

The depth comparison runs the author's f32 model on the same normalized tensor,
so it tests model math independently of the cubic image resize. Pose runs the
author's detector, crop and decoder on the same photograph. The native model
count matches the reference's one person. Synthetic NMS tests cover overlapping
and separate detections; this run does not establish parity on a photograph
containing several people.

[`tests/fixtures/zimage-preprocess`](../../tests/fixtures/zimage-preprocess/meta.json)
holds the shared depth input, raw depth output, pose coordinates and reference
PNG. Its metadata records model, source and output SHA256s. The author source
revisions are `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` for Depth Anything V2 and
`3dca5db79d9f9ffdd378753ddf6ec66535aace88` for DWPose's ONNX branch. Reference
versions were torch 2.14.0, ONNX Runtime 1.29.0 and OpenCV 5.0.0.93.

The cold CPU smoke produced valid maps and exercised loading. It was not a
controlled benchmark and establishes no performance figure. Reproduction
commands are in [the parity runbook](../parity.md#image-preprocessor-reference).

## Cache and runtime

The three files total 450.3 MB and were downloaded with SHA256 verification.
Cache lookup follows `HF_HUB_CACHE`, then `HF_HOME/hub`, then the default HF cache.

| Repository and revision | File |
| --- | --- |
| `jeroenvlek/depth-anything-v2-safetensors` at `a8cd5eb93485537f612b31b78864b41f659b245d` | `depth_anything_v2_vits.safetensors` |
| `yzd-v/DWPose` at `1a7144101628d69ee7a3768d1ee3a094070dc388` | `yolox_l.onnx` |
| `yzd-v/DWPose` at `1a7144101628d69ee7a3768d1ee3a094070dc388` | `dw-ll_ucoco_384.onnx` |

The Rust dependency is pinned to `ort` 2.0.0-rc.13. Its downloaded macOS ARM64
bundle contains ONNX Runtime 1.28.0 and links statically. The `coreml` feature
selects the available macOS bundle; sessions use the CPU execution provider.
`otool -L` on the linked test executable lists only system libraries and
frameworks, with no `libonnxruntime.dylib`. Installing the binary requires no
separate ONNX runtime file. Python and its reference packages remain confined
to the manual reference script.

The implementations follow [Depth Anything V2's author source](https://github.com/DepthAnything/Depth-Anything-V2)
and [DWPose's ONNX source](https://github.com/IDEA-Research/DWPose/tree/onnx/ControlNet-v1-1-nightly/annotator/dwpose).
The Rust depth modules retain their pinned Candle source credits and document
the corrections above.
