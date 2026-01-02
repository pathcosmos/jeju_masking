#!/usr/bin/env python3
"""
얼굴 및 번호판 마스킹 스크립트
- YOLOv8 기반 객체 감지 및 트래킹
- 얼굴: person 감지 후 상단 영역 블러
- 번호판: car/truck/bus 감지 후 번호판 영역 블러
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def get_face_region(person_box, expand_ratio=0.3):
    """person bbox에서 얼굴 영역 추정 (상단 1/5 영역)"""
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    width = x2 - x1

    # 얼굴은 상단 약 1/5 영역
    face_height = height * 0.25
    face_width = width * 0.6

    # 중앙 정렬
    center_x = (x1 + x2) / 2

    face_x1 = center_x - face_width / 2
    face_x2 = center_x + face_width / 2
    face_y1 = y1
    face_y2 = y1 + face_height

    # 확장
    expand_w = (face_x2 - face_x1) * expand_ratio
    expand_h = (face_y2 - face_y1) * expand_ratio

    face_x1 = max(0, face_x1 - expand_w)
    face_x2 = face_x2 + expand_w
    face_y1 = max(0, face_y1 - expand_h)
    face_y2 = face_y2 + expand_h

    return int(face_x1), int(face_y1), int(face_x2), int(face_y2)


def get_plate_region(vehicle_box, expand_ratio=0.2):
    """차량 bbox에서 번호판 영역 추정 (하단 중앙 영역)"""
    x1, y1, x2, y2 = vehicle_box
    height = y2 - y1
    width = x2 - x1

    # 번호판은 하단 1/4, 중앙 1/2 영역
    plate_height = height * 0.15
    plate_width = width * 0.4

    # 중앙 하단
    center_x = (x1 + x2) / 2

    plate_x1 = center_x - plate_width / 2
    plate_x2 = center_x + plate_width / 2
    plate_y1 = y2 - height * 0.35
    plate_y2 = y2 - height * 0.1

    # 확장
    expand_w = (plate_x2 - plate_x1) * expand_ratio
    expand_h = (plate_y2 - plate_y1) * expand_ratio

    plate_x1 = max(0, plate_x1 - expand_w)
    plate_x2 = plate_x2 + expand_w
    plate_y1 = max(0, plate_y1 - expand_h)
    plate_y2 = plate_y2 + expand_h

    return int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2)


def apply_blur(frame, x1, y1, x2, y2, blur_strength=51):
    """지정 영역에 가우시안 블러 적용"""
    h, w = frame.shape[:2]

    # 경계 체크
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return frame

    # 블러 강도는 홀수여야 함
    blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

    # 영역 추출 및 블러
    roi = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
    frame[y1:y2, x1:x2] = blurred

    return frame


def apply_mosaic(frame, x1, y1, x2, y2, block_size=15):
    """지정 영역에 모자이크 적용"""
    h, w = frame.shape[:2]

    # 경계 체크
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]

    if roi_h < block_size or roi_w < block_size:
        return frame

    # 축소 후 확대
    small = cv2.resize(roi, (roi_w // block_size, roi_h // block_size), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic

    return frame


def process_video(
    input_path: str,
    output_path: str = None,
    mask_faces: bool = True,
    mask_plates: bool = True,
    mask_type: str = "blur",  # "blur" or "mosaic"
    blur_strength: int = 51,
    mosaic_size: int = 15,
    confidence: float = 0.3,
    skip_frames: int = 0,
    preview: bool = False,
    max_frames: int = None,
):
    """비디오 처리 메인 함수"""

    # 모델 로드 (YOLOv8 nano - 빠르고 가벼움)
    print("YOLO 모델 로딩 중...")
    model = YOLO("yolov8n.pt")

    # COCO 클래스 ID
    PERSON_CLASS = 0
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # 비디오 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {input_path}")

    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"입력 비디오: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

    # 출력 경로 설정
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{input_stem}_masked.mp4")

    # 비디오 작성기
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"출력 비디오: {output_path}")
    print(f"마스킹 대상: {'얼굴' if mask_faces else ''} {'번호판' if mask_plates else ''}")
    print(f"마스킹 방식: {mask_type}")
    print()

    frame_count = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 최대 프레임 제한
            if max_frames and frame_count > max_frames:
                break

            # 프레임 스킵
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                out.write(frame)
                continue

            processed_count += 1

            # YOLO 감지 (트래킹 포함)
            results = model.track(frame, persist=True, conf=confidence, verbose=False)

            if results[0].boxes is not None:
                boxes = results[0].boxes

                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)

                    # 사람 감지 -> 얼굴 마스킹
                    if mask_faces and cls == PERSON_CLASS:
                        fx1, fy1, fx2, fy2 = get_face_region((x1, y1, x2, y2))
                        if mask_type == "blur":
                            frame = apply_blur(frame, fx1, fy1, fx2, fy2, blur_strength)
                        else:
                            frame = apply_mosaic(frame, fx1, fy1, fx2, fy2, mosaic_size)

                    # 차량 감지 -> 번호판 마스킹
                    if mask_plates and cls in VEHICLE_CLASSES:
                        px1, py1, px2, py2 = get_plate_region((x1, y1, x2, y2))
                        if mask_type == "blur":
                            frame = apply_blur(frame, px1, py1, px2, py2, blur_strength)
                        else:
                            frame = apply_mosaic(frame, px1, py1, px2, py2, mosaic_size)

            # 프레임 저장
            out.write(frame)

            # 진행 상황
            if processed_count % 100 == 0:
                progress = frame_count / total_frames * 100
                print(f"진행: {frame_count}/{total_frames} ({progress:.1f}%)")

            # 미리보기 (선택적)
            if preview:
                display = cv2.resize(frame, (1280, 720))
                cv2.imshow("Preview", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("사용자에 의해 중단됨")
                    break

    finally:
        cap.release()
        out.release()
        if preview:
            cv2.destroyAllWindows()

    print(f"\n완료! 총 {frame_count} 프레임 처리됨")
    print(f"출력: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="얼굴 및 번호판 마스킹 처리")
    parser.add_argument("input", help="입력 비디오 파일 경로")
    parser.add_argument("-o", "--output", help="출력 비디오 파일 경로")
    parser.add_argument("--no-faces", action="store_true", help="얼굴 마스킹 비활성화")
    parser.add_argument("--no-plates", action="store_true", help="번호판 마스킹 비활성화")
    parser.add_argument("--mask-type", choices=["blur", "mosaic"], default="blur", help="마스킹 방식")
    parser.add_argument("--blur-strength", type=int, default=51, help="블러 강도 (홀수)")
    parser.add_argument("--mosaic-size", type=int, default=15, help="모자이크 블록 크기")
    parser.add_argument("--confidence", type=float, default=0.3, help="감지 신뢰도 임계값")
    parser.add_argument("--skip-frames", type=int, default=0, help="스킵할 프레임 수 (속도 향상)")
    parser.add_argument("--preview", action="store_true", help="처리 중 미리보기 표시")
    parser.add_argument("--max-frames", type=int, help="처리할 최대 프레임 수 (테스트용)")

    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        mask_faces=not args.no_faces,
        mask_plates=not args.no_plates,
        mask_type=args.mask_type,
        blur_strength=args.blur_strength,
        mosaic_size=args.mosaic_size,
        confidence=args.confidence,
        skip_frames=args.skip_frames,
        preview=args.preview,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
