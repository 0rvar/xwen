//! DWPose ONNX detector, SimCC decoding, and OpenPose whole-body rendering.
//! Adapted from IDEA-Research/DWPose onnx branch annotator/dwpose (Apache-2.0),
//! with detector postprocessing from YOLOX (Apache-2.0).
use super::*;
use ort::{session::Session, value::Tensor as OrtTensor};
pub(super) struct Pose {
    detector: Session,
    estimator: Session,
}
#[derive(Clone, Copy, Debug)]
struct BoxScore {
    bbox: [f32; 4],
    score: f32,
}
#[derive(Clone, Copy, Debug, Default)]
struct Point {
    x: f32,
    y: f32,
    score: f32,
}
impl Pose {
    pub fn load() -> Result<Self> {
        let detector = Session::builder()?
            .with_intra_threads(8)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(cached("yzd-v/DWPose", "yolox_l.onnx")?)?;
        let estimator = Session::builder()?
            .with_intra_threads(8)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(cached("yzd-v/DWPose", "dw-ll_ucoco_384.onnx")?)?;
        Ok(Self {
            detector,
            estimator,
        })
    }
    pub fn run(&mut self, image: &RgbImage) -> Result<RgbImage> {
        let people = self.infer(image)?;
        Ok(render(&people, image.width(), image.height()))
    }
    fn infer(&mut self, image: &RgbImage) -> Result<Vec<Vec<Point>>> {
        let (w, h) = image.dimensions();
        let ratio = (640. / w as f32).min(640. / h as f32);
        let rw = (w as f32 * ratio) as usize;
        let rh = (h as f32 * ratio) as usize;
        let mut input = vec![114f32; 3 * 640 * 640];
        // DWPose's author accepts OpenCV BGR, with no channel reversal in either
        // network preprocessor. The native surface supplies RGB, so swap here.
        for y in 0..rh {
            for x in 0..rw {
                let p = sample(
                    image,
                    (x as f32 + 0.5) * w as f32 / rw as f32 - 0.5,
                    (y as f32 + 0.5) * h as f32 / rh as f32 - 0.5,
                    false,
                );
                for c in 0..3 {
                    input[c * 640 * 640 + y * 640 + x] = p[2 - c];
                }
            }
        }
        let outputs = self.detector.run(ort::inputs![OrtTensor::from_array((
            [1usize, 3, 640, 640],
            input
        ))?])?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        ensure!(
            shape.len() == 3 && shape[0] == 1 && shape[2] == 85,
            "unexpected DWPose detector output {shape:?}"
        );
        let mut boxes = decode_detector(values, ratio)?;
        drop(outputs);
        if boxes.is_empty() {
            boxes.push(BoxScore {
                bbox: [0., 0., w as f32, h as f32],
                score: 1.,
            });
        }
        let mut people = Vec::new();
        for detected in boxes {
            let (center, scale) = crop_geometry(detected.bbox, 288., 384.);
            let mut input = vec![0f32; 3 * 288 * 384];
            for y in 0..384 {
                for x in 0..288 {
                    let sx = x as f32 / 288. * scale[0] + center[0] - scale[0] / 2.;
                    let sy = y as f32 / 384. * scale[1] + center[1] - scale[1] / 2.;
                    let p = sample(image, sx, sy, true);
                    for c in 0..3 {
                        input[c * 288 * 384 + y * 288 + x] =
                            (p[2 - c] - [123.675, 116.28, 103.53][c]) / [58.395, 57.12, 57.375][c];
                    }
                }
            }
            let outputs = self.estimator.run(ort::inputs![OrtTensor::from_array((
                [1usize, 3, 384, 288],
                input
            ))?])?;
            let (xs, x) = outputs[0].try_extract_tensor::<f32>()?;
            let (ys, y) = outputs[1].try_extract_tensor::<f32>()?;
            ensure!(
                xs.len() == 3
                    && ys.len() == 3
                    && xs[0] == 1
                    && ys[0] == 1
                    && xs[1] == 133
                    && ys[1] == 133
                    && xs[2] == 576
                    && ys[2] == 768,
                "unexpected DWPose SimCC shapes {xs:?} {ys:?}"
            );
            people.push(openpose(decode_simcc(x, y, center, scale)));
        }
        Ok(people)
    }
}
// OpenCV INTER_LINEAR uses a 1/32 interpolation table in affine warps. Round
// the result to u8 before normalization, because the source photograph is u8.
fn sample(image: &RgbImage, mut x: f32, mut y: f32, black_border: bool) -> [f32; 3] {
    let (w, h) = image.dimensions();
    if !black_border {
        x = x.clamp(0., (w - 1) as f32);
        y = y.clamp(0., (h - 1) as f32)
    }
    if black_border {
        x = (x * 32.).round() / 32.;
        y = (y * 32.).round() / 32.;
    }
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let get = |xx: i32, yy: i32, c: usize| {
        if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
            0.
        } else {
            image.get_pixel(xx as u32, yy as u32)[c] as f32
        }
    };
    std::array::from_fn(|c| {
        ((get(x0, y0, c) * (1. - dx) + get(x0 + 1, y0, c) * dx) * (1. - dy)
            + (get(x0, y0 + 1, c) * (1. - dx) + get(x0 + 1, y0 + 1, c) * dx) * dy)
            .round()
            .clamp(0., 255.)
    })
}
fn decode_detector(values: &[f32], ratio: f32) -> Result<Vec<BoxScore>> {
    ensure!(values.len() == 8400 * 85, "unexpected YOLOX output length");
    let mut boxes = Vec::new();
    let mut i = 0;
    for stride in [8, 16, 32] {
        let side = 640 / stride;
        for gy in 0..side {
            for gx in 0..side {
                let v = &values[i * 85..(i + 1) * 85];
                i += 1;
                let score = v[4] * v[5];
                if score <= 0.3 || !score.is_finite() {
                    continue;
                }
                let cx = (v[0] + gx as f32) * stride as f32;
                let cy = (v[1] + gy as f32) * stride as f32;
                let bw = v[2].exp() * stride as f32;
                let bh = v[3].exp() * stride as f32;
                let bbox = [
                    (cx - bw / 2.) / ratio,
                    (cy - bh / 2.) / ratio,
                    (cx + bw / 2.) / ratio,
                    (cy + bh / 2.) / ratio,
                ];
                if bbox.iter().all(|x| x.is_finite()) && bbox[2] > bbox[0] && bbox[3] > bbox[1] {
                    boxes.push(BoxScore { bbox, score });
                }
            }
        }
    }
    Ok(nms(boxes, 0.45))
}
fn nms(mut boxes: Vec<BoxScore>, threshold: f32) -> Vec<BoxScore> {
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<BoxScore> = Vec::new();
    for b in boxes {
        if keep.iter().all(|k| iou(k.bbox, b.bbox) <= threshold) {
            keep.push(b)
        }
    }
    keep
}
fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let area = |v: [f32; 4]| (v[2] - v[0] + 1.).max(0.) * (v[3] - v[1] + 1.).max(0.);
    let overlap = (a[2].min(b[2]) - a[0].max(b[0]) + 1.).max(0.)
        * (a[3].min(b[3]) - a[1].max(b[1]) + 1.).max(0.);
    overlap / (area(a) + area(b) - overlap).max(f32::EPSILON)
}
fn crop_geometry(b: [f32; 4], w: f32, h: f32) -> ([f32; 2], [f32; 2]) {
    let center = [(b[0] + b[2]) / 2., (b[1] + b[3]) / 2.];
    let mut scale = [(b[2] - b[0]) * 1.25, (b[3] - b[1]) * 1.25];
    if scale[0] / scale[1] > w / h {
        scale[1] = scale[0] * h / w
    } else {
        scale[0] = scale[1] * w / h
    }
    (center, scale)
}
fn argmax(v: &[f32]) -> (usize, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > best.1 {
            best = (i, x)
        }
    }
    best
}
fn decode_simcc(x: &[f32], y: &[f32], center: [f32; 2], scale: [f32; 2]) -> Vec<Point> {
    (0..133)
        .map(|i| {
            let (ix, sx) = argmax(&x[i * 576..(i + 1) * 576]);
            let (iy, sy) = argmax(&y[i * 768..(i + 1) * 768]);
            let score = sx.min(sy);
            let (px, py) = if score > 0. {
                (ix as f32 / 2., iy as f32 / 2.)
            } else {
                (-0.5, -0.5)
            };
            Point {
                x: px / 288. * scale[0] + center[0] - scale[0] / 2.,
                y: py / 384. * scale[1] + center[1] - scale[1] / 2.,
                score,
            }
        })
        .collect()
}
fn openpose(mut points: Vec<Point>) -> Vec<Point> {
    let a = points[5];
    let b = points[6];
    points.insert(
        17,
        Point {
            x: (a.x + b.x) / 2.,
            y: (a.y + b.y) / 2.,
            score: if a.score > 0.3 && b.score > 0.3 {
                1.
            } else {
                0.
            },
        },
    );
    let original = points.clone();
    for (src, dst) in [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        .into_iter()
        .zip([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])
    {
        points[dst] = original[src]
    }
    points
}
const COLORS: [[u8; 3]; 18] = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
];
const LIMBS: [[usize; 2]; 17] = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
];
fn dot(canvas: &mut RgbImage, p: Point, r: i32, color: [u8; 3]) {
    if p.score < 0.3
        || !p.x.is_finite()
        || !p.y.is_finite()
        || p.x < -(r as f32)
        || p.y < -(r as f32)
        || p.x > canvas.width() as f32 + r as f32
        || p.y > canvas.height() as f32 + r as f32
    {
        return;
    }
    let cx = p.x as i32;
    let cy = p.y as i32;
    for y in cy - r..=cy + r {
        for x in cx - r..=cx + r {
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r
                && x >= 0
                && y >= 0
                && x < canvas.width() as i32
                && y < canvas.height() as i32
            {
                canvas.put_pixel(x as u32, y as u32, Rgb(color))
            }
        }
    }
}
fn limb(canvas: &mut RgbImage, a: Point, b: Point, r: f32, color: [u8; 3], ellipse: bool) {
    if a.score <= 0.3 || b.score <= 0.3 {
        return;
    }
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = dx.hypot(dy);
    if len < 1. {
        return;
    }
    let ux = dx / len;
    let uy = dy / len;
    let cx = (a.x + b.x) / 2.;
    let cy = (a.y + b.y) / 2.;
    for y in
        (a.y.min(b.y) - r).max(0.) as u32..=((a.y.max(b.y) + r) as u32).min(canvas.height() - 1)
    {
        for x in
            (a.x.min(b.x) - r).max(0.) as u32..=((a.x.max(b.x) + r) as u32).min(canvas.width() - 1)
        {
            let px = x as f32 - cx;
            let py = y as f32 - cy;
            let along = px * ux + py * uy;
            let across = -px * uy + py * ux;
            let inside = if ellipse {
                along * along / (len * len / 4.) + across * across / (r * r) <= 1.
            } else {
                along.abs() <= len / 2. && across.abs() <= r
            };
            if inside {
                canvas.put_pixel(x, y, Rgb(color))
            }
        }
    }
}
fn hsv(h: f32) -> [u8; 3] {
    let h = h * 6.;
    let i = h.floor() as usize;
    let f = h - i as f32;
    let q = ((1. - f) * 255.) as u8;
    let t = (f * 255.) as u8;
    match i % 6 {
        0 => [255, t, 0],
        1 => [q, 255, 0],
        2 => [0, 255, t],
        3 => [0, q, 255],
        4 => [t, 0, 255],
        _ => [255, 0, q],
    }
}
fn render(people: &[Vec<Point>], w: u32, h: u32) -> RgbImage {
    let mut canvas = RgbImage::new(w, h);
    for (i, [a, b]) in LIMBS.iter().enumerate() {
        for p in people {
            limb(&mut canvas, p[*a], p[*b], 4., COLORS[i], true)
        }
    }
    for pixel in canvas.pixels_mut() {
        for c in 0..3 {
            pixel[c] = (pixel[c] as f32 * 0.6) as u8
        }
    }
    for (i, color) in COLORS.iter().enumerate() {
        for p in people {
            dot(&mut canvas, p[i], 4, *color)
        }
    }
    for p in people {
        for start in [92, 113] {
            let hand = &p[start..start + 21];
            for finger in 0..5 {
                for bone in 0..4 {
                    let a = if bone == 0 { 0 } else { finger * 4 + bone };
                    let b = finger * 4 + bone + 1;
                    limb(
                        &mut canvas,
                        hand[a],
                        hand[b],
                        1.,
                        hsv((finger * 4 + bone) as f32 / 20.),
                        false,
                    )
                }
            }
            for point in hand {
                if point.x > 0.01 && point.y > 0.01 {
                    dot(&mut canvas, *point, 4, [0, 0, 255])
                }
            }
        }
        for point in &p[24..92] {
            if point.x > 0.01 && point.y > 0.01 {
                dot(&mut canvas, *point, 3, [255, 255, 255])
            }
        }
    }
    canvas
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires cached DWPose and author fixture"]
    fn author_pose_reference() {
        let file = std::env::var("XWEN_PREPROCESS_TEST_IMAGE").unwrap_or_else(|_| {
            "tests/fixtures/zimage-transformer/512x512-p1-s0/image-fp32.png".into()
        });
        let root = std::env::var("XWEN_PREPROCESS_REF_DIR")
            .unwrap_or_else(|_| "tests/fixtures/zimage-preprocess".into());
        let rgb = image::open(file).unwrap().to_rgb8();
        let points = Pose::load().unwrap().infer(&rgb).unwrap();
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(PathBuf::from(&root).join("pose-reference.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(points.len(), reference["points"].as_array().unwrap().len());
        let mut max_distance = 0f32;
        let mut max_score_gap = 0f32;
        for (i, person) in points.iter().enumerate() {
            for (j, point) in person.iter().enumerate() {
                let x = reference["points"][i][j][0].as_f64().unwrap() as f32;
                let y = reference["points"][i][j][1].as_f64().unwrap() as f32;
                let score = reference["scores"][i][j].as_f64().unwrap() as f32;
                if score > 0.3 && point.score > 0.3 {
                    max_distance = max_distance.max((point.x - x).hypot(point.y - y));
                }
                max_score_gap = max_score_gap.max((point.score - score).abs());
            }
        }
        eprintln!(
            "DWPose author: max visible point distance {max_distance}, score gap {max_score_gap}"
        );
        assert!(max_distance < 3., "visible point gap {max_distance}");
        assert!(max_score_gap < 0.1, "confidence gap {max_score_gap}");
        let expected = image::open(PathBuf::from(root).join("pose-reference.png"))
            .unwrap()
            .to_rgb8();
        let actual = render(&points, rgb.width(), rgb.height());
        assert_eq!(actual.dimensions(), expected.dimensions());
        let (mut intersection, mut union) = (0usize, 0usize);
        for (a, b) in actual.pixels().zip(expected.pixels()) {
            let foreground_a = a.0.iter().any(|v| *v != 0);
            let foreground_b = b.0.iter().any(|v| *v != 0);
            intersection += usize::from(foreground_a && foreground_b);
            union += usize::from(foreground_a || foreground_b);
        }
        let overlap = intersection as f64 / union as f64;
        eprintln!("DWPose rendered foreground IoU {overlap}");
        assert!(overlap > 0.7, "pose foreground IoU {overlap}");
    }
    #[test]
    fn nms_keeps_distinct_people() {
        let a = BoxScore {
            bbox: [0., 0., 10., 10.],
            score: 0.9,
        };
        let b = BoxScore {
            bbox: [1., 1., 11., 11.],
            score: 0.8,
        };
        let c = BoxScore {
            bbox: [30., 30., 40., 40.],
            score: 0.7,
        };
        assert_eq!(nms(vec![b, c, a], 0.45).len(), 2)
    }
    #[test]
    fn simcc_maps_crop_coordinates_and_requires_both_shoulders() {
        let mut x = vec![0.; 133 * 576];
        let mut y = vec![0.; 133 * 768];
        for i in 0..133 {
            x[i * 576 + 288] = 1.;
            y[i * 768 + 384] = 0.8;
        }
        let p = decode_simcc(&x, &y, [100., 200.], [288., 384.]);
        assert_eq!(p[0].x, 100.);
        assert_eq!(p[0].y, 200.);
        let mut p = openpose(p);
        assert_eq!(p[1].score, 1.);
        p[2].score = 0.;
        assert!(
            render(&[p], 300, 400)
                .pixels()
                .any(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
        );
    }
}
