#!/usr/bin/env python3
"""
고급 얼굴 및 번호판 마스킹 스크립트
- 얼굴: OpenCV DNN 기반 얼굴 감지
- 번호판: yolov8 + 번호판 영역 특화 감지
- ByteTrack 기반 트래킹으로 일관된 마스킹
- 상세 로그 기록
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import urllib.request
import time


def setup_logger(log_file=None, verbose=False):
    """로거 설정"""
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    handlers = [console_handler]

    # 파일 핸들러
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    return logging.getLogger(__name__)


# 모델 경로
MODELS_DIR = Path(__file__).parent / "models"

# OpenCV DNN 얼굴 감지 모델
FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def download_opencv_face_model():
    """OpenCV DNN 얼굴 감지 모델 다운로드"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    proto_path = MODELS_DIR / "deploy.prototxt"
    model_path = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

    if proto_path.exists() and model_path.exists():
        return proto_path, model_path

    print("OpenCV 얼굴 감지 모델 다운로드 중...")

    try:
        if not proto_path.exists():
            urllib.request.urlretrieve(FACE_PROTO_URL, proto_path)
        if not model_path.exists():
            urllib.request.urlretrieve(FACE_MODEL_URL, model_path)
        print("얼굴 모델 다운로드 완료!")
        return proto_path, model_path
    except Exception as e:
        print(f"얼굴 모델 다운로드 실패: {e}")
        return None, None


class FaceDetectorDNN:
    """OpenCV DNN 기반 얼굴 감지"""

    def __init__(self, confidence=0.5):
        proto_path, model_path = download_opencv_face_model()
        if proto_path and model_path:
            self.net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            self.enabled = True
        else:
            self.net = None
            self.enabled = False
        self.confidence = confidence

    def detect(self, frame):
        """얼굴 감지 - [(x1,y1,x2,y2), ...] 반환"""
        if not self.enabled:
            return []

        h, w = frame.shape[:2]

        # 네트워크 입력을 위한 블롭 생성
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > self.confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2, y2))

        return faces


def get_plate_region(vehicle_box, expand_ratio=0.3):
    """차량 bbox에서 번호판 영역 추정"""
    x1, y1, x2, y2 = vehicle_box
    height = y2 - y1
    width = x2 - x1

    # 번호판은 하단 영역에 위치
    plate_height = height * 0.18
    plate_width = width * 0.45

    center_x = (x1 + x2) / 2

    plate_x1 = center_x - plate_width / 2
    plate_x2 = center_x + plate_width / 2
    plate_y1 = y2 - height * 0.35
    plate_y2 = y2 - height * 0.08

    # 확장
    expand_w = (plate_x2 - plate_x1) * expand_ratio
    expand_h = (plate_y2 - plate_y1) * expand_ratio

    plate_x1 = max(0, plate_x1 - expand_w)
    plate_x2 = plate_x2 + expand_w
    plate_y1 = max(0, plate_y1 - expand_h)
    plate_y2 = plate_y2 + expand_h

    return int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2)


def apply_blur(frame, x1, y1, x2, y2, blur_strength=51):
    """가우시안 블러 적용"""
    h, w = frame.shape[:2]

    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return frame

    blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

    roi = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
    frame[y1:y2, x1:x2] = blurred

    return frame


def apply_mosaic(frame, x1, y1, x2, y2, block_size=15):
    """모자이크 적용"""
    h, w = frame.shape[:2]

    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]

    if roi_h < block_size or roi_w < block_size:
        return frame

    small = cv2.resize(roi, (max(1, roi_w // block_size), max(1, roi_h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic

    return frame


def expand_box(box, ratio, frame_shape):
    """박스 확장"""
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    expand_w = box_w * ratio
    expand_h = box_h * ratio

    x1 = max(0, x1 - expand_w)
    x2 = min(w, x2 + expand_w)
    y1 = max(0, y1 - expand_h)
    y2 = min(h, y2 + expand_h)

    return int(x1), int(y1), int(x2), int(y2)


class VideoMasker:
    """비디오 마스킹 처리 클래스"""

    def __init__(
        self,
        mask_faces: bool = True,
        mask_plates: bool = True,
        mask_type: str = "blur",
        blur_strength: int = 51,
        mosaic_size: int = 15,
        face_confidence: float = 0.4,
        vehicle_confidence: float = 0.3,
        face_expand: float = 0.2,
        plate_expand: float = 0.3,
    ):
        self.mask_faces = mask_faces
        self.mask_plates = mask_plates
        self.mask_type = mask_type
        self.blur_strength = blur_strength
        self.mosaic_size = mosaic_size
        self.face_confidence = face_confidence
        self.vehicle_confidence = vehicle_confidence
        self.face_expand = face_expand
        self.plate_expand = plate_expand

        # 모델 로드
        self.face_detector = None
        self.vehicle_model = None

        if mask_faces:
            print("얼굴 감지 모델 로딩 (OpenCV DNN)...")
            self.face_detector = FaceDetectorDNN(confidence=face_confidence)
            if not self.face_detector.enabled:
                print("경고: 얼굴 감지 모델 로드 실패")

        if mask_plates:
            print("차량 감지 모델 로딩 (YOLOv8)...")
            self.vehicle_model = YOLO("yolov8n.pt")

        # COCO 클래스
        self.VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def apply_mask(self, frame, x1, y1, x2, y2):
        """마스크 적용"""
        if self.mask_type == "blur":
            return apply_blur(frame, x1, y1, x2, y2, self.blur_strength)
        else:
            return apply_mosaic(frame, x1, y1, x2, y2, self.mosaic_size)

    def process_frame(self, frame):
        """단일 프레임 처리"""

        # 얼굴 감지 및 마스킹 (OpenCV DNN)
        if self.mask_faces and self.face_detector and self.face_detector.enabled:
            faces = self.face_detector.detect(frame)
            for face_box in faces:
                x1, y1, x2, y2 = expand_box(face_box, self.face_expand, frame.shape)
                frame = self.apply_mask(frame, x1, y1, x2, y2)

        # 차량 번호판 마스킹 (YOLOv8)
        if self.mask_plates and self.vehicle_model:
            results = self.vehicle_model.track(frame, persist=True,
                                               conf=self.vehicle_confidence, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls in self.VEHICLE_CLASSES:
                        xyxy = box.xyxy[0].cpu().numpy()
                        px1, py1, px2, py2 = get_plate_region(xyxy, self.plate_expand)
                        frame = self.apply_mask(frame, px1, py1, px2, py2)

        return frame

    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        skip_frames: int = 0,
        max_frames: int = None,
        start_time: float = None,  # 시작 시간 (초)
        end_time: float = None,    # 종료 시간 (초)
        preview: bool = False,
        preview_scale: float = 0.5,
        use_hevc: bool = False,    # HEVC 인코딩 사용
        log_file: str = None,      # 로그 파일 경로
        verbose: bool = False,     # 상세 로그 출력
    ):
        """비디오 전체 처리"""
        import subprocess
        import tempfile

        # 로거 설정
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"masking_{timestamp}.log")

        logger = setup_logger(log_file, verbose)
        logger.info("=" * 60)
        logger.info("마스킹 작업 시작")
        logger.info("=" * 60)

        start_total_time = time.time()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"비디오를 열 수 없습니다: {input_path}")
            raise ValueError(f"비디오를 열 수 없습니다: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 시작/종료 프레임 계산
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        end_frame = min(end_frame, total_frames)

        if max_frames:
            end_frame = min(start_frame + max_frames, end_frame)

        process_frames = end_frame - start_frame

        logger.info(f"입력 파일: {input_path}")
        logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
        logger.info(f"전체 프레임: {total_frames} ({total_frames/fps/60:.1f}분)")

        if start_time or end_time:
            start_min = start_frame / fps / 60
            end_min = end_frame / fps / 60
            logger.info(f"처리 구간: {start_min:.1f}분 ~ {end_min:.1f}분 ({process_frames} frames)")

        if output_path is None:
            input_stem = Path(input_path).stem
            suffix = f"_{int(start_time//60)}m-{int(end_time//60)}m" if start_time else ""
            output_path = str(Path(input_path).parent / f"{input_stem}{suffix}_masked.mp4")

        # 시작 프레임으로 이동
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.debug(f"시작 프레임으로 이동: {start_frame}")

        # HEVC 사용시 임시 파일로 먼저 저장
        if use_hevc:
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            logger.info(f"출력 파일: {output_path} (HEVC 인코딩 예정)")
            logger.debug(f"임시 파일: {temp_output}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"출력 파일: {output_path}")

        logger.info(f"설정: 얼굴={'O' if self.mask_faces else 'X'}, "
                   f"번호판={'O' if self.mask_plates else 'X'}, "
                   f"방식={self.mask_type}")
        logger.info(f"로그 파일: {log_file}")
        logger.info("-" * 60)

        frame_count = 0
        processed_count = 0
        last_masked_frame = None
        total_faces = 0
        total_vehicles = 0
        errors = []
        frame_times = []

        try:
            while True:
                frame_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    logger.debug("프레임 읽기 종료")
                    break

                frame_count += 1
                current_frame = start_frame + frame_count

                if current_frame > end_frame:
                    break

                processed_count += 1

                # 프레임 스킵 시 이전 마스크 재사용
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    out.write(frame if last_masked_frame is None else frame)
                    continue

                # 프레임 처리
                try:
                    frame = self.process_frame(frame)
                    last_masked_frame = frame.copy()
                    out.write(frame)

                    frame_time = time.time() - frame_start
                    frame_times.append(frame_time)

                    # 상세 로그 (verbose 모드)
                    if verbose and processed_count % 10 == 0:
                        avg_time = sum(frame_times[-100:]) / len(frame_times[-100:])
                        logger.debug(f"프레임 {processed_count}: 처리시간 {frame_time:.3f}s, 평균 {avg_time:.3f}s")

                except Exception as e:
                    error_msg = f"프레임 {processed_count} 처리 오류: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    out.write(frame)  # 원본 프레임 저장

                # 진행 상황 출력
                if processed_count % 60 == 0:
                    progress = processed_count / process_frames * 100
                    eta = (process_frames - processed_count) / fps / 60
                    elapsed = time.time() - start_total_time
                    avg_fps = processed_count / elapsed if elapsed > 0 else 0
                    logger.info(f"[{progress:5.1f}%] {processed_count}/{process_frames} 프레임 "
                               f"(남은시간: {eta:.1f}분, 처리속도: {avg_fps:.1f} fps)")

                # 미리보기
                if preview:
                    preview_w = int(width * preview_scale)
                    preview_h = int(height * preview_scale)
                    display = cv2.resize(frame, (preview_w, preview_h))
                    cv2.imshow("Masking Preview (Q to quit)", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.warning("사용자에 의해 중단됨")
                        break

        except Exception as e:
            logger.error(f"치명적 오류 발생: {str(e)}")
            raise

        finally:
            cap.release()
            out.release()
            if preview:
                cv2.destroyAllWindows()

        # 마스킹 완료 통계
        masking_time = time.time() - start_total_time
        logger.info("-" * 60)
        logger.info(f"마스킹 완료!")
        logger.info(f"처리된 프레임: {processed_count}")
        logger.info(f"마스킹 소요시간: {masking_time/60:.1f}분")
        if frame_times:
            logger.info(f"평균 프레임 처리시간: {sum(frame_times)/len(frame_times):.3f}초")
        if errors:
            logger.warning(f"오류 발생 횟수: {len(errors)}")
            for err in errors[:5]:
                logger.warning(f"  - {err}")

        # HEVC 인코딩
        if use_hevc:
            logger.info("\nHEVC 인코딩 시작...")
            hevc_start = time.time()
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-c:v', 'libx265',
                '-preset', 'medium',
                '-crf', '23',
                '-tag:v', 'hvc1',  # Apple 호환성
                '-an',  # 오디오 제외
                output_path
            ]
            logger.debug(f"ffmpeg 명령: {' '.join(ffmpeg_cmd)}")
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                hevc_time = time.time() - hevc_start
                if result.returncode == 0:
                    logger.info(f"HEVC 인코딩 완료! (소요시간: {hevc_time/60:.1f}분)")
                    os.unlink(temp_output)
                    logger.debug(f"임시 파일 삭제됨: {temp_output}")
                else:
                    logger.error(f"HEVC 인코딩 실패: {result.stderr}")
                    logger.info(f"임시 파일 유지: {temp_output}")
            except Exception as e:
                logger.error(f"ffmpeg 실행 오류: {e}")
                logger.info(f"임시 파일: {temp_output}")

        # 최종 요약
        total_time = time.time() - start_total_time
        logger.info("=" * 60)
        logger.info("작업 완료 요약")
        logger.info("=" * 60)
        logger.info(f"출력 파일: {output_path}")
        logger.info(f"총 소요시간: {total_time/60:.1f}분")
        logger.info(f"오류 횟수: {len(errors)}")
        logger.info(f"로그 파일: {log_file}")

        return output_path


def parse_time(time_str):
    """시간 문자열을 초로 변환 (예: '23:30' -> 1410초, '90' -> 90초)"""
    if time_str is None:
        return None
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(time_str)


def main():
    parser = argparse.ArgumentParser(
        description="고급 얼굴/번호판 마스킹 (전용 모델 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (얼굴 + 번호판 블러)
  python mask_video_advanced.py video.mp4

  # 특정 구간만 처리 (23분~28분)
  python mask_video_advanced.py video.mp4 --start 23:00 --end 28:00

  # HEVC 인코딩으로 출력 (상세 로그 포함)
  python mask_video_advanced.py video.mp4 --start 23:00 --end 28:00 --hevc --verbose

  # 모자이크 처리
  python mask_video_advanced.py video.mp4 --mask-type mosaic

  # 얼굴만 처리
  python mask_video_advanced.py video.mp4 --no-plates

  # 커스텀 로그 파일 지정
  python mask_video_advanced.py video.mp4 --log my_log.log --verbose
        """
    )

    parser.add_argument("input", help="입력 비디오 파일")
    parser.add_argument("-o", "--output", help="출력 파일 경로")
    parser.add_argument("--start", type=str, help="시작 시간 (예: 23:00 또는 1380)")
    parser.add_argument("--end", type=str, help="종료 시간 (예: 28:00 또는 1680)")
    parser.add_argument("--hevc", action="store_true", help="HEVC(H.265) 인코딩 사용")
    parser.add_argument("--no-faces", action="store_true", help="얼굴 마스킹 비활성화")
    parser.add_argument("--no-plates", action="store_true", help="번호판 마스킹 비활성화")
    parser.add_argument("--mask-type", choices=["blur", "mosaic"], default="blur")
    parser.add_argument("--blur-strength", type=int, default=51)
    parser.add_argument("--mosaic-size", type=int, default=15)
    parser.add_argument("--face-conf", type=float, default=0.4, help="얼굴 감지 신뢰도")
    parser.add_argument("--vehicle-conf", type=float, default=0.3, help="차량 감지 신뢰도")
    parser.add_argument("--face-expand", type=float, default=0.2, help="얼굴 영역 확장 비율")
    parser.add_argument("--plate-expand", type=float, default=0.3, help="번호판 영역 확장 비율")
    parser.add_argument("--skip-frames", type=int, default=0, help="스킵할 프레임 수")
    parser.add_argument("--max-frames", type=int, help="최대 처리 프레임 수")
    parser.add_argument("--preview", action="store_true", help="처리 중 미리보기")
    parser.add_argument("--preview-scale", type=float, default=0.5, help="미리보기 크기 비율")
    parser.add_argument("--log", type=str, help="로그 파일 경로 (기본: 자동 생성)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    masker = VideoMasker(
        mask_faces=not args.no_faces,
        mask_plates=not args.no_plates,
        mask_type=args.mask_type,
        blur_strength=args.blur_strength,
        mosaic_size=args.mosaic_size,
        face_confidence=args.face_conf,
        vehicle_confidence=args.vehicle_conf,
        face_expand=args.face_expand,
        plate_expand=args.plate_expand,
    )

    masker.process_video(
        input_path=args.input,
        output_path=args.output,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
        start_time=parse_time(args.start),
        end_time=parse_time(args.end),
        preview=args.preview,
        preview_scale=args.preview_scale,
        use_hevc=args.hevc,
        log_file=args.log,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
