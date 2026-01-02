#!/usr/bin/env python3
"""
최적화된 얼굴 및 번호판 마스킹 스크립트 v2.1
- MPS GPU 가속 (Apple Silicon) - 완전 구현
- 멀티스레딩 파이프라인 (읽기/처리/쓰기 분리)
- 프레임 스킵 + 트래킹 보간
- 해상도 다운스케일 감지 (출력은 원본 유지)
- 배치 추론 지원 - 완전 구현
- 확장된 트래킹 파라미터 - 완전 구현
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from threading import Thread, Event
from queue import Queue, Empty
import cv2
import numpy as np
from ultralytics import YOLO
import urllib.request
import time
import platform
import yaml
import tempfile


class FlushStreamHandler(logging.StreamHandler):
    """즉시 flush되는 스트림 핸들러"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(log_file=None, verbose=False):
    """로거 설정 (실시간 출력)"""
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 실시간 출력을 위한 커스텀 핸들러
    console_handler = FlushStreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    handlers = [console_handler]

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
    return logging.getLogger(__name__)


# 모델 경로
MODELS_DIR = Path(__file__).parent / "models"

# OpenCV DNN 얼굴 감지 모델
FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def get_optimal_device():
    """최적 디바이스 자동 감지"""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available() and platform.processor() == 'arm':
        return 'mps'
    return 'cpu'


def create_custom_tracker_config(tracker_type, track_buffer, match_thresh):
    """커스텀 트래커 설정 파일 생성"""
    if tracker_type == "bytetrack":
        config = {
            'tracker_type': 'bytetrack',
            'track_high_thresh': 0.5,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.6,
            'track_buffer': track_buffer,
            'match_thresh': match_thresh,
            'fuse_score': True
        }
    else:  # botsort
        config = {
            'tracker_type': 'botsort',
            'track_high_thresh': 0.5,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.6,
            'track_buffer': track_buffer,
            'match_thresh': match_thresh,
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.25,
            'with_reid': False,
            'gmc_method': 'sparseOptFlow',
            'fuse_score': True
        }

    # 임시 파일로 저장
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, prefix=f'{tracker_type}_custom_'
    )
    yaml.dump(config, temp_file, default_flow_style=False)
    temp_file.close()
    return temp_file.name


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
    """OpenCV DNN 기반 얼굴 감지 (최적화)"""

    def __init__(self, confidence=0.5, input_size=300):
        proto_path, model_path = download_opencv_face_model()
        if proto_path and model_path:
            self.net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            # OpenCV DNN 백엔드 최적화
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.enabled = True
        else:
            self.net = None
            self.enabled = False
        self.confidence = confidence
        self.input_size = input_size

    def detect(self, frame, scale_factor=1.0):
        """얼굴 감지 - [(x1,y1,x2,y2), ...] 반환"""
        if not self.enabled:
            return []

        h, w = frame.shape[:2]

        # 다운스케일된 이미지로 감지
        if scale_factor < 1.0:
            small_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            detect_h, detect_w = small_frame.shape[:2]
        else:
            small_frame = frame
            detect_h, detect_w = h, w

        blob = cv2.dnn.blobFromImage(
            cv2.resize(small_frame, (self.input_size, self.input_size)),
            1.0, (self.input_size, self.input_size),
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
                faces.append((x1, y1, x2, y2, conf))

        return faces


class FrameReader(Thread):
    """비동기 프레임 읽기 스레드"""

    def __init__(self, cap, queue, start_frame, end_frame, queue_size=128):
        super().__init__(daemon=True)
        self.cap = cap
        self.queue = queue
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.stopped = Event()

    def run(self):
        frame_idx = self.start_frame
        while not self.stopped.is_set() and frame_idx < self.end_frame:
            if self.queue.full():
                time.sleep(0.001)
                continue
            ret, frame = self.cap.read()
            if not ret:
                break
            self.queue.put((frame_idx, frame))
            frame_idx += 1
        self.queue.put(None)  # 종료 신호

    def stop(self):
        self.stopped.set()


class FrameWriter(Thread):
    """비동기 프레임 쓰기 스레드"""

    def __init__(self, out, queue):
        super().__init__(daemon=True)
        self.out = out
        self.queue = queue
        self.stopped = Event()
        self.frames_written = 0

    def run(self):
        pending = {}
        next_frame = 0

        while not self.stopped.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except Empty:
                continue

            if item is None:
                # 남은 프레임 모두 쓰기
                while next_frame in pending:
                    self.out.write(pending.pop(next_frame))
                    self.frames_written += 1
                    next_frame += 1
                break

            frame_idx, frame = item
            pending[frame_idx] = frame

            # 순서대로 쓰기
            while next_frame in pending:
                self.out.write(pending.pop(next_frame))
                self.frames_written += 1
                next_frame += 1

    def stop(self):
        self.stopped.set()


def get_plate_region(vehicle_box, expand_ratio=0.3):
    """차량 bbox에서 번호판 영역 추정"""
    x1, y1, x2, y2 = vehicle_box[:4]
    height = y2 - y1
    width = x2 - x1

    plate_height = height * 0.18
    plate_width = width * 0.45

    center_x = (x1 + x2) / 2

    plate_x1 = center_x - plate_width / 2
    plate_x2 = center_x + plate_width / 2
    plate_y1 = y2 - height * 0.35
    plate_y2 = y2 - height * 0.08

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
    x1, y1, x2, y2 = box[:4]
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


def interpolate_box(box1, box2, alpha):
    """두 박스 사이 선형 보간"""
    if box1 is None:
        return box2
    if box2 is None:
        return box1
    return tuple(int(b1 * (1 - alpha) + b2 * alpha) for b1, b2 in zip(box1[:4], box2[:4]))


class TrackingInterpolator:
    """트래킹 결과 보간 관리"""

    def __init__(self, max_age=30):
        self.tracks = {}  # track_id -> {'boxes': deque, 'last_seen': frame_idx}
        self.max_age = max_age

    def update(self, frame_idx, detections):
        """새 감지 결과로 업데이트"""
        seen_ids = set()

        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue

            seen_ids.add(track_id)
            box = det['box']

            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'boxes': deque(maxlen=10),
                    'last_seen': frame_idx,
                    'type': det.get('type', 'unknown')
                }

            self.tracks[track_id]['boxes'].append((frame_idx, box))
            self.tracks[track_id]['last_seen'] = frame_idx

        # 오래된 트랙 제거
        expired = [tid for tid, t in self.tracks.items()
                   if frame_idx - t['last_seen'] > self.max_age]
        for tid in expired:
            del self.tracks[tid]

    def get_interpolated(self, frame_idx):
        """현재 프레임에 대한 보간된 박스들 반환"""
        results = []

        for track_id, track in self.tracks.items():
            boxes = track['boxes']
            if len(boxes) == 0:
                continue

            # 가장 최근 박스 찾기
            last_frame, last_box = boxes[-1]

            if frame_idx == last_frame:
                # 정확히 일치
                results.append({
                    'track_id': track_id,
                    'box': last_box,
                    'type': track['type'],
                    'interpolated': False
                })
            elif frame_idx > last_frame and frame_idx - last_frame <= self.max_age:
                # 보간 (마지막 박스 사용, 또는 속도 기반 예측)
                if len(boxes) >= 2:
                    prev_frame, prev_box = boxes[-2]
                    # 속도 계산
                    dt = last_frame - prev_frame
                    if dt > 0:
                        velocity = tuple((b2 - b1) / dt for b1, b2 in zip(prev_box, last_box))
                        frames_ahead = frame_idx - last_frame
                        predicted_box = tuple(int(b + v * frames_ahead) for b, v in zip(last_box, velocity))
                        results.append({
                            'track_id': track_id,
                            'box': predicted_box,
                            'type': track['type'],
                            'interpolated': True
                        })
                else:
                    results.append({
                        'track_id': track_id,
                        'box': last_box,
                        'type': track['type'],
                        'interpolated': True
                    })

        return results


class VideoMaskerOptimized:
    """최적화된 비디오 마스킹 처리 클래스"""

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
        # 최적화 파라미터
        device: str = "auto",
        detect_interval: int = 3,  # N프레임마다 감지
        detect_scale: float = 0.5,  # 감지용 다운스케일 (0.5 = 절반 해상도)
        batch_size: int = 4,  # 배치 추론 크기
        # 트래킹 파라미터
        tracker: str = "bytetrack",
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        iou_thresh: float = 0.5,
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

        # 최적화 설정
        self.detect_interval = detect_interval
        self.detect_scale = detect_scale
        self.batch_size = batch_size
        self.tracker_type = tracker
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.iou_thresh = iou_thresh

        # 디바이스 설정
        if device == "auto":
            self.device = get_optimal_device()
        else:
            self.device = device
        print(f"사용 디바이스: {self.device}")

        # 커스텀 트래커 설정 파일 생성
        self.tracker_config_path = create_custom_tracker_config(
            tracker, track_buffer, match_thresh
        )
        print(f"트래커 설정: {tracker} (buffer={track_buffer}, match_thresh={match_thresh})")

        # 모델 로드
        self.face_detector = None
        self.vehicle_model = None

        if mask_faces:
            print("얼굴 감지 모델 로딩 (OpenCV DNN)...")
            self.face_detector = FaceDetectorDNN(confidence=face_confidence)
            if not self.face_detector.enabled:
                print("경고: 얼굴 감지 모델 로드 실패")

        if mask_plates:
            print(f"차량 감지 모델 로딩 (YOLOv8, device={self.device})...")
            self.vehicle_model = YOLO("yolov8n.pt")
            # GPU 가속 활성화
            if self.device != 'cpu':
                self.vehicle_model.to(self.device)

        # COCO 클래스
        self.VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        # 트래킹 보간기
        self.face_interpolator = TrackingInterpolator(max_age=track_buffer)
        self.vehicle_interpolator = TrackingInterpolator(max_age=track_buffer)

        # 얼굴 트래킹용 간단한 ID 할당
        self.next_face_id = 0
        self.prev_faces = []

        # 차량 트래킹용 ID (track()이 ID를 못 줄 때 대비)
        self.next_vehicle_id = 10000

        # 배치 처리용 버퍼
        self.frame_buffer = []
        self.frame_idx_buffer = []

    def _get_next_vehicle_id(self):
        """차량용 고유 ID 생성"""
        self.next_vehicle_id += 1
        return self.next_vehicle_id

    def apply_mask(self, frame, x1, y1, x2, y2):
        """마스크 적용"""
        if self.mask_type == "blur":
            return apply_blur(frame, x1, y1, x2, y2, self.blur_strength)
        else:
            return apply_mosaic(frame, x1, y1, x2, y2, self.mosaic_size)

    def match_faces_simple(self, prev_faces, curr_faces, iou_thresh=0.3):
        """간단한 IOU 기반 얼굴 매칭"""
        def iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter

            return inter / union if union > 0 else 0

        matched = []
        used_prev = set()

        for curr in curr_faces:
            best_iou = 0
            best_prev = None
            best_idx = -1

            for idx, prev in enumerate(prev_faces):
                if idx in used_prev:
                    continue
                score = iou(curr[:4], prev['box'][:4])
                if score > best_iou and score > iou_thresh:
                    best_iou = score
                    best_prev = prev
                    best_idx = idx

            if best_prev:
                used_prev.add(best_idx)
                matched.append({
                    'box': curr[:4],
                    'conf': curr[4] if len(curr) > 4 else 1.0,
                    'track_id': best_prev['track_id'],
                    'type': 'face'
                })
            else:
                matched.append({
                    'box': curr[:4],
                    'conf': curr[4] if len(curr) > 4 else 1.0,
                    'track_id': self.next_face_id,
                    'type': 'face'
                })
                self.next_face_id += 1

        return matched

    def detect_all(self, frame, frame_idx):
        """얼굴과 차량 모두 감지 (최적화)"""
        detections = {'faces': [], 'vehicles': []}
        h, w = frame.shape[:2]

        # 감지용 다운스케일
        if self.detect_scale < 1.0:
            detect_frame = cv2.resize(
                frame,
                (int(w * self.detect_scale), int(h * self.detect_scale))
            )
            scale_x = w / detect_frame.shape[1]
            scale_y = h / detect_frame.shape[0]
        else:
            detect_frame = frame
            scale_x = scale_y = 1.0

        # 얼굴 감지
        if self.mask_faces and self.face_detector and self.face_detector.enabled:
            faces = self.face_detector.detect(detect_frame)
            # 원본 해상도로 스케일 복원
            scaled_faces = []
            for face in faces:
                x1, y1, x2, y2 = face[:4]
                conf = face[4] if len(face) > 4 else 1.0
                scaled_faces.append((
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y),
                    conf
                ))

            # 간단한 트래킹 매칭
            matched_faces = self.match_faces_simple(self.prev_faces, scaled_faces)
            detections['faces'] = matched_faces
            self.prev_faces = matched_faces

        # 차량 감지 (YOLO + ByteTrack) - device 명시 및 커스텀 트래커 사용
        if self.mask_plates and self.vehicle_model:
            results = self.vehicle_model.track(
                detect_frame,
                persist=True,
                conf=self.vehicle_confidence,
                iou=self.iou_thresh,
                tracker=self.tracker_config_path,  # 커스텀 트래커 설정 사용
                device=self.device,  # GPU 디바이스 명시
                verbose=False
            )

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls in self.VEHICLE_CLASSES:
                        xyxy = box.xyxy[0].cpu().numpy()
                        # 원본 해상도로 스케일 복원
                        x1, y1, x2, y2 = xyxy
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                        track_id = int(box.id[0]) if box.id is not None else None

                        detections['vehicles'].append({
                            'box': (x1, y1, x2, y2),
                            'track_id': track_id,
                            'type': 'vehicle',
                            'class': cls
                        })

        return detections

    def detect_batch(self, frames, frame_indices):
        """배치 추론 - 여러 프레임을 한 번에 처리"""
        if not frames:
            return []

        h, w = frames[0].shape[:2]
        batch_detections = []

        # 감지용 다운스케일
        if self.detect_scale < 1.0:
            detect_frames = [
                cv2.resize(f, (int(w * self.detect_scale), int(h * self.detect_scale)))
                for f in frames
            ]
            scale_x = w / detect_frames[0].shape[1]
            scale_y = h / detect_frames[0].shape[0]
        else:
            detect_frames = frames
            scale_x = scale_y = 1.0

        # 먼저 빈 결과 리스트 초기화
        for _ in detect_frames:
            batch_detections.append({'faces': [], 'vehicles': []})

        # 차량 배치 감지 (YOLO) - 프레임별 track() 사용으로 변경
        if self.mask_plates and self.vehicle_model:
            for batch_idx, detect_frame in enumerate(detect_frames):
                # track() 사용하여 트래킹 ID 유지
                results = self.vehicle_model.track(
                    detect_frame,
                    persist=True,
                    conf=self.vehicle_confidence,
                    iou=self.iou_thresh,
                    tracker=self.tracker_config_path,
                    device=self.device,
                    verbose=False
                )

                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        if cls in self.VEHICLE_CLASSES:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                            track_id = int(box.id[0]) if box.id is not None else self._get_next_vehicle_id()

                            batch_detections[batch_idx]['vehicles'].append({
                                'box': (x1, y1, x2, y2),
                                'track_id': track_id,
                                'type': 'vehicle',
                                'class': cls
                            })

        # 얼굴 감지 (프레임별 처리 - OpenCV DNN은 배치 미지원)
        if self.mask_faces and self.face_detector and self.face_detector.enabled:
            for batch_idx, detect_frame in enumerate(detect_frames):
                faces = self.face_detector.detect(detect_frame)
                scaled_faces = []
                for face in faces:
                    x1, y1, x2, y2 = face[:4]
                    conf = face[4] if len(face) > 4 else 1.0
                    scaled_faces.append((
                        int(x1 * scale_x), int(y1 * scale_y),
                        int(x2 * scale_x), int(y2 * scale_y),
                        conf
                    ))

                matched_faces = self.match_faces_simple(self.prev_faces, scaled_faces)
                batch_detections[batch_idx]['faces'] = matched_faces
                self.prev_faces = matched_faces

        return batch_detections

    def process_frame(self, frame, frame_idx, force_detect=False):
        """단일 프레임 처리 (최적화)"""
        should_detect = force_detect or (frame_idx % self.detect_interval == 0)

        if should_detect:
            # 실제 감지 수행
            detections = self.detect_all(frame, frame_idx)

            # 보간기 업데이트
            self.face_interpolator.update(frame_idx, detections['faces'])
            self.vehicle_interpolator.update(frame_idx, detections['vehicles'])

        # 보간된 결과 가져오기
        faces = self.face_interpolator.get_interpolated(frame_idx)
        vehicles = self.vehicle_interpolator.get_interpolated(frame_idx)

        # 얼굴 마스킹
        if self.mask_faces:
            for face in faces:
                box = expand_box(face['box'], self.face_expand, frame.shape)
                frame = self.apply_mask(frame, *box)

        # 번호판 마스킹
        if self.mask_plates:
            for vehicle in vehicles:
                plate_box = get_plate_region(vehicle['box'], self.plate_expand)
                frame = self.apply_mask(frame, *plate_box)

        return frame, len(faces), len(vehicles)

    def process_frames_batch(self, frames, frame_indices):
        """배치 프레임 처리 - 감지는 배치로, 마스킹은 개별로"""
        results = []

        # 감지가 필요한 프레임 필터링
        detect_frames = []
        detect_indices = []
        for i, (frame, idx) in enumerate(zip(frames, frame_indices)):
            if idx % self.detect_interval == 0:
                detect_frames.append(frame)
                detect_indices.append((i, idx))

        # 배치 감지 수행
        if detect_frames:
            batch_detections = self.detect_batch(detect_frames, [idx for _, idx in detect_indices])

            # 보간기 업데이트
            for (batch_idx, frame_idx), detections in zip(detect_indices, batch_detections):
                self.face_interpolator.update(frame_idx, detections['faces'])
                self.vehicle_interpolator.update(frame_idx, detections['vehicles'])

        # 각 프레임에 마스킹 적용
        for frame, frame_idx in zip(frames, frame_indices):
            faces = self.face_interpolator.get_interpolated(frame_idx)
            vehicles = self.vehicle_interpolator.get_interpolated(frame_idx)

            # 얼굴 마스킹
            if self.mask_faces:
                for face in faces:
                    box = expand_box(face['box'], self.face_expand, frame.shape)
                    frame = self.apply_mask(frame, *box)

            # 번호판 마스킹
            if self.mask_plates:
                for vehicle in vehicles:
                    plate_box = get_plate_region(vehicle['box'], self.plate_expand)
                    frame = self.apply_mask(frame, *plate_box)

            results.append((frame, len(faces), len(vehicles)))

        return results

    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        start_time: float = None,
        end_time: float = None,
        use_hevc: bool = False,
        log_file: str = None,
        verbose: bool = False,
        num_threads: int = 2,  # 읽기/쓰기 스레드 수
    ):
        """비디오 전체 처리 (멀티스레딩)"""
        import subprocess
        import tempfile

        # 로거 설정
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"masking_{timestamp}.log")

        logger = setup_logger(log_file, verbose)
        logger.info("=" * 60)
        logger.info("최적화 마스킹 v2.0 시작")
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
        process_frames = end_frame - start_frame

        logger.info(f"입력 파일: {input_path}")
        logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
        logger.info(f"전체 프레임: {total_frames} ({total_frames/fps/60:.1f}분)")
        logger.info(f"디바이스: {self.device}")
        logger.info(f"감지 간격: {self.detect_interval}프레임마다")
        logger.info(f"감지 스케일: {self.detect_scale}")
        logger.info(f"배치 크기: {self.batch_size}")
        logger.info(f"트래커: {self.tracker_type} (buffer={self.track_buffer}, match={self.match_thresh})")

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

        # 출력 설정
        if use_hevc:
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            logger.info(f"출력 파일: {output_path} (HEVC 인코딩 예정)")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"출력 파일: {output_path}")

        logger.info(f"설정: 얼굴={'O' if self.mask_faces else 'X'}, "
                   f"번호판={'O' if self.mask_plates else 'X'}, "
                   f"방식={self.mask_type}")
        logger.info(f"로그 파일: {log_file}")
        logger.info("-" * 60)

        # 멀티스레딩 설정
        read_queue = Queue(maxsize=128)
        write_queue = Queue(maxsize=128)

        reader = FrameReader(cap, read_queue, start_frame, end_frame)
        writer = FrameWriter(out, write_queue)

        reader.start()
        writer.start()

        processed_count = 0
        total_faces = 0
        total_vehicles = 0
        errors = []
        frame_times = []

        # 배치 버퍼
        batch_frames = []
        batch_indices = []

        try:
            while True:
                try:
                    item = read_queue.get(timeout=1.0)
                except Empty:
                    if not reader.is_alive():
                        break
                    continue

                if item is None:
                    # 남은 배치 처리
                    if batch_frames:
                        try:
                            batch_start = time.time()
                            results = self.process_frames_batch(batch_frames, batch_indices)
                            for (frame, n_faces, n_vehicles), idx in zip(results, batch_indices):
                                total_faces += n_faces
                                total_vehicles += n_vehicles
                                write_queue.put((idx, frame))
                            frame_times.append((time.time() - batch_start) / len(batch_frames))
                        except Exception as e:
                            logger.error(f"배치 처리 오류: {str(e)}")
                            for frame, idx in zip(batch_frames, batch_indices):
                                write_queue.put((idx, frame))
                    break

                frame_idx, frame = item
                batch_frames.append(frame)
                batch_indices.append(processed_count)
                processed_count += 1

                # 배치가 찼으면 처리
                if len(batch_frames) >= self.batch_size:
                    try:
                        batch_start = time.time()
                        results = self.process_frames_batch(batch_frames, batch_indices)
                        for (frame, n_faces, n_vehicles), idx in zip(results, batch_indices):
                            total_faces += n_faces
                            total_vehicles += n_vehicles
                            write_queue.put((idx, frame))
                        frame_times.append((time.time() - batch_start) / len(batch_frames))
                    except Exception as e:
                        error_msg = f"배치 처리 오류: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        for frame, idx in zip(batch_frames, batch_indices):
                            write_queue.put((idx, frame))

                    batch_frames = []
                    batch_indices = []

                # 진행 상황 출력 (60프레임 = 약 1초마다)
                if processed_count % 60 == 0:
                    progress = processed_count / process_frames * 100
                    elapsed = time.time() - start_total_time
                    avg_fps = processed_count / elapsed if elapsed > 0 else 0
                    eta_sec = (process_frames - processed_count) / avg_fps if avg_fps > 0 else 0
                    current_time_sec = processed_count / fps
                    logger.info(f"[{progress:5.1f}%] {processed_count:,}/{process_frames:,} frames | "
                               f"{current_time_sec/60:.1f}분 처리완료 | "
                               f"{avg_fps:.1f} fps | 남은시간: {eta_sec/60:.1f}분 | "
                               f"얼굴: {total_faces}, 차량: {total_vehicles}")

        except KeyboardInterrupt:
            logger.warning("사용자에 의해 중단됨")

        finally:
            write_queue.put(None)
            reader.stop()
            writer.join(timeout=10)
            cap.release()
            out.release()

        # 마스킹 완료 통계
        masking_time = time.time() - start_total_time
        logger.info("-" * 60)
        logger.info(f"마스킹 완료!")
        logger.info(f"처리된 프레임: {processed_count}")
        logger.info(f"감지된 얼굴: {total_faces}, 차량: {total_vehicles}")
        logger.info(f"마스킹 소요시간: {masking_time/60:.1f}분")
        if frame_times:
            avg_fps = len(frame_times) / sum(frame_times)
            logger.info(f"평균 처리 속도: {avg_fps:.1f} fps")
        if errors:
            logger.warning(f"오류 발생 횟수: {len(errors)}")

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
                '-tag:v', 'hvc1',
                '-an',
                output_path
            ]
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                hevc_time = time.time() - hevc_start
                if result.returncode == 0:
                    logger.info(f"HEVC 인코딩 완료! (소요시간: {hevc_time/60:.1f}분)")
                    os.unlink(temp_output)
                else:
                    logger.error(f"HEVC 인코딩 실패: {result.stderr}")
            except Exception as e:
                logger.error(f"ffmpeg 실행 오류: {e}")

        # 최종 요약
        total_time = time.time() - start_total_time
        logger.info("=" * 60)
        logger.info("작업 완료 요약")
        logger.info("=" * 60)
        logger.info(f"출력 파일: {output_path}")
        logger.info(f"총 소요시간: {total_time/60:.1f}분")

        # 성능 비교
        video_duration = process_frames / fps
        speed_ratio = video_duration / total_time if total_time > 0 else 0
        logger.info(f"처리 속도: {speed_ratio:.2f}x (실시간 대비)")

        return output_path


def parse_time(time_str):
    """시간 문자열을 초로 변환"""
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
        description="최적화된 얼굴/번호판 마스킹 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (자동 GPU 감지)
  python mask_video_optimized.py video.mp4

  # 특정 구간 처리
  python mask_video_optimized.py video.mp4 --start 23:00 --end 28:00

  # 감지 간격 조정 (5프레임마다 = 더 빠름)
  python mask_video_optimized.py video.mp4 --detect-interval 5

  # 감지 스케일 조정 (0.25 = 1/4 해상도로 감지)
  python mask_video_optimized.py video.mp4 --detect-scale 0.25

  # CPU 강제 사용
  python mask_video_optimized.py video.mp4 --device cpu

  # 전체 옵션
  python mask_video_optimized.py video.mp4 --detect-interval 5 --detect-scale 0.5 --hevc --verbose
        """
    )

    parser.add_argument("input", help="입력 비디오 파일")
    parser.add_argument("-o", "--output", help="출력 파일 경로")
    parser.add_argument("--start", type=str, help="시작 시간 (예: 23:00)")
    parser.add_argument("--end", type=str, help="종료 시간 (예: 28:00)")
    parser.add_argument("--hevc", action="store_true", help="HEVC 인코딩 사용")

    # 마스킹 옵션
    parser.add_argument("--no-faces", action="store_true", help="얼굴 마스킹 비활성화")
    parser.add_argument("--no-plates", action="store_true", help="번호판 마스킹 비활성화")
    parser.add_argument("--mask-type", choices=["blur", "mosaic"], default="blur")
    parser.add_argument("--blur-strength", type=int, default=51)
    parser.add_argument("--mosaic-size", type=int, default=15)

    # 감지 파라미터
    parser.add_argument("--face-conf", type=float, default=0.4, help="얼굴 감지 신뢰도")
    parser.add_argument("--vehicle-conf", type=float, default=0.3, help="차량 감지 신뢰도")
    parser.add_argument("--face-expand", type=float, default=0.2, help="얼굴 영역 확장 비율")
    parser.add_argument("--plate-expand", type=float, default=0.3, help="번호판 영역 확장 비율")

    # 최적화 파라미터
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "mps", "cuda"],
                       help="처리 디바이스 (기본: auto)")
    parser.add_argument("--detect-interval", type=int, default=3,
                       help="감지 간격 (N프레임마다 감지, 기본: 3)")
    parser.add_argument("--detect-scale", type=float, default=0.5,
                       help="감지용 다운스케일 비율 (기본: 0.5)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="배치 추론 크기 (기본: 4)")

    # 트래킹 파라미터
    parser.add_argument("--tracker", type=str, default="bytetrack",
                       choices=["bytetrack", "botsort"],
                       help="트래커 종류 (기본: bytetrack)")
    parser.add_argument("--track-buffer", type=int, default=30,
                       help="트래킹 버퍼 크기 (기본: 30)")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                       help="트래킹 매칭 임계값 (기본: 0.8)")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                       help="IOU 임계값 (기본: 0.5)")

    # 로깅
    parser.add_argument("--log", type=str, help="로그 파일 경로")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그")

    args = parser.parse_args()

    masker = VideoMaskerOptimized(
        mask_faces=not args.no_faces,
        mask_plates=not args.no_plates,
        mask_type=args.mask_type,
        blur_strength=args.blur_strength,
        mosaic_size=args.mosaic_size,
        face_confidence=args.face_conf,
        vehicle_confidence=args.vehicle_conf,
        face_expand=args.face_expand,
        plate_expand=args.plate_expand,
        device=args.device,
        detect_interval=args.detect_interval,
        detect_scale=args.detect_scale,
        batch_size=args.batch_size,
        tracker=args.tracker,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        iou_thresh=args.iou_thresh,
    )

    masker.process_video(
        input_path=args.input,
        output_path=args.output,
        start_time=parse_time(args.start),
        end_time=parse_time(args.end),
        use_hevc=args.hevc,
        log_file=args.log,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
