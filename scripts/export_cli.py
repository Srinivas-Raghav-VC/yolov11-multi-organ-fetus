"""
Unified export CLI (ONNX / TensorRT / OpenVINO / CoreML / TFLite).
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def do_export(model: YOLO, fmt: str, out: Path, name: str, **kwargs):
    print(f"Exporting {fmt} -> {name} ...")
    result = model.export(format=fmt, project=str(out), name=name, exist_ok=True, **kwargs)
    print(f"  -> {result}")


def main():
    ap = argparse.ArgumentParser(description='Unified exporter')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--img', type=int, default=640)
    ap.add_argument('--out', default='exports')
    ap.add_argument('--formats', nargs='+', default=['onnx', 'engine', 'openvino', 'coreml'])
    ap.add_argument('--half', action='store_true', help='use FP16 where applicable')
    ap.add_argument('--trt-int8', action='store_true', help='export TensorRT INT8 (requires calibration)')
    ap.add_argument('--calib', help='calibration images dir (e.g., data_yolo/images/val)')
    args = ap.parse_args()

    w = Path(args.weights)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(w))

    # ONNX (no built-in NMS for cross-runtime stability)
    if 'onnx' in args.formats:
        do_export(model, 'onnx', out, name='onnx_fp16' if args.half else 'onnx_fp32', imgsz=args.img, half=args.half, dynamic=False, simplify=True)
    if 'engine' in args.formats:
        do_export(model, 'engine', out, name='tensorrt_fp16' if args.half else 'tensorrt_fp32', imgsz=args.img, half=args.half, dynamic=False)
    if 'openvino' in args.formats:
        do_export(model, 'openvino', out, name='openvino_fp16' if args.half else 'openvino_fp32', imgsz=args.img, half=args.half)
    if 'coreml' in args.formats:
        do_export(model, 'coreml', out, name='coreml_fp16' if args.half else 'coreml_fp32', imgsz=args.img, half=args.half)

    # Optional TRT INT8
    if args.trt_int8:
        try:
            kw = dict(imgsz=args.img, dynamic=False, int8=True)
            if args.calib:
                kw['data'] = str(Path(args.calib))
            do_export(model, 'engine', out, name='tensorrt_int8', **kw)
        except Exception as e:
            print(f"TensorRT INT8 export skipped: {e}")

    # TFLite INT8 (best effort)
    try:
        do_export(model, 'tflite', out, name='tflite_int8', imgsz=args.img, int8=True)
    except Exception as e:
        print(f"TFLite INT8 export skipped: {e}")

    print('Done.')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
