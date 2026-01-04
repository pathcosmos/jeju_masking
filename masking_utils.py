#!/usr/bin/env python3
"""
마스킹 공통 유틸리티 함수 모듈
- 블러/모자이크 적용
- 박스 영역 계산
- 시간 파싱
- 로깅 설정
"""

import logging
import sys
import cv2
import numpy as np


class FlushStreamHandler(logging.StreamHandler):
    """즉시 flush되는 스트림 핸들러"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(log_file=None, verbose=False, name=__name__):
    """로거 설정 (실시간 출력)"""
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

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
    return logging.getLogger(name)


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
    """박스 영역 확장"""
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


def validate_box(box, frame_shape, max_area_ratio=0.5, min_size=10, max_aspect_ratio=10.0):
    """
    박스가 유효한지 검증 (너무 크거나 비정상적인 박스 필터링)
    
    Args:
        box: (x1, y1, x2, y2) 박스 좌표
        frame_shape: 프레임 shape (h, w, ...)
        max_area_ratio: 프레임 대비 최대 면적 비율 (기본: 50%)
        min_size: 최소 크기 (픽셀)
        max_aspect_ratio: 최대 가로세로 비율
    
    Returns:
        bool: 유효한 박스인지 여부
    """
    x1, y1, x2, y2 = box[:4]
    h, w = frame_shape[:2]
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    # 최소 크기 검사
    if box_w < min_size or box_h < min_size:
        return False
    
    # 음수 크기 검사
    if box_w <= 0 or box_h <= 0:
        return False
    
    # 경계 검사
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        # 경계를 벗어나면 클리핑해서 다시 계산
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        box_w = x2 - x1
        box_h = y2 - y1
    
    # 면적 비율 검사 (프레임 대비 너무 큰 박스 제외)
    frame_area = w * h
    box_area = box_w * box_h
    area_ratio = box_area / frame_area if frame_area > 0 else 0
    
    if area_ratio > max_area_ratio:
        return False
    
    # 가로세로 비율 검사 (너무 극단적인 비율 제외)
    aspect_ratio = max(box_w, box_h) / min(box_w, box_h) if min(box_w, box_h) > 0 else float('inf')
    if aspect_ratio > max_aspect_ratio:
        return False
    
    return True


def validate_and_clip_box(box, frame_shape, max_area_ratio=0.5, min_size=10):
    """
    박스 검증 및 클리핑. 유효하지 않으면 None 반환.
    
    Returns:
        (x1, y1, x2, y2) or None
    """
    x1, y1, x2, y2 = box[:4]
    h, w = frame_shape[:2]
    
    # 클리핑
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    # 최소 크기 검사
    if box_w < min_size or box_h < min_size:
        return None
    
    # 면적 비율 검사
    frame_area = w * h
    box_area = box_w * box_h
    area_ratio = box_area / frame_area if frame_area > 0 else 0
    
    if area_ratio > max_area_ratio:
        return None
    
    return (x1, y1, x2, y2)


def get_plate_region(vehicle_box, expand_ratio=0.3):
    """차량 bbox에서 번호판 영역 추정 (하단 범퍼용 - 레거시)"""
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


def detect_plates_in_vehicle(frame, vehicle_box, min_plate_ratio=2.0, max_plate_ratio=5.5):
    """
    차량 영역 내에서 번호판 형태(가로 직사각형)를 OpenCV로 감지
    
    한국 번호판 비율:
    - 구형: 520mm x 110mm (비율 4.7:1)
    - 신형: 520mm x 110mm 또는 335mm x 170mm (비율 2.0:1)
    - 이륜차: 220mm x 170mm (비율 1.3:1)
    
    Returns: [(x1, y1, x2, y2), ...] 번호판 후보 영역들
    """
    x1, y1, x2, y2 = map(int, vehicle_box[:4])
    h, w = frame.shape[:2]
    
    # 경계 체크
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return []
    
    # 차량 영역 추출
    roi = frame[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    
    if roi_h < 20 or roi_w < 40:
        return []
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거 및 대비 향상
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # 적응형 이진화 (다양한 조명 조건 대응)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
    )
    
    # 모폴로지 연산으로 노이즈 제거 및 번호판 영역 강조
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 엣지 검출
    edges = cv2.Canny(gray, 30, 200)
    
    # 컨투어 찾기 (이진화 + 엣지 결합)
    combined = cv2.bitwise_or(binary, edges)
    contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_candidates = []
    
    # 차량 영역 대비 번호판 최소/최대 크기
    min_plate_width = roi_w * 0.15
    max_plate_width = roi_w * 0.7
    min_plate_height = roi_h * 0.04
    max_plate_height = roi_h * 0.25
    
    for contour in contours:
        # 윤곽선 근사화
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 사각형 형태 (4~6개 꼭지점)
        if len(approx) >= 4 and len(approx) <= 6:
            px, py, pw, ph = cv2.boundingRect(contour)
            
            # 크기 필터링
            if pw < min_plate_width or pw > max_plate_width:
                continue
            if ph < min_plate_height or ph > max_plate_height:
                continue
            
            # 비율 필터링 (가로가 세로보다 긴 직사각형)
            aspect_ratio = pw / ph if ph > 0 else 0
            if aspect_ratio < min_plate_ratio or aspect_ratio > max_plate_ratio:
                continue
            
            # 면적 필터링
            area = pw * ph
            roi_area = roi_w * roi_h
            if area < roi_area * 0.005 or area > roi_area * 0.15:
                continue
            
            # 전역 좌표로 변환
            plate_candidates.append((
                x1 + px,
                y1 + py,
                x1 + px + pw,
                y1 + py + ph
            ))
    
    # 중복 제거 (IoU 기반)
    if len(plate_candidates) > 1:
        plate_candidates = _remove_overlapping_boxes(plate_candidates, iou_thresh=0.3)
    
    return plate_candidates


def _remove_overlapping_boxes(boxes, iou_thresh=0.3):
    """중복되는 박스 제거 (NMS)"""
    if len(boxes) == 0:
        return []
    
    # 면적 기준 정렬 (큰 것 우선)
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
    
    keep = []
    for box in boxes:
        should_keep = True
        for kept_box in keep:
            iou = _compute_iou(box, kept_box)
            if iou > iou_thresh:
                should_keep = False
                break
        if should_keep:
            keep.append(box)
    
    return keep


def _compute_iou(box1, box2):
    """IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def get_all_plate_regions(frame, vehicle_box, expand_ratio=0.2, use_detection=True):
    """
    차량에서 모든 가능한 번호판 영역 반환
    
    1. OpenCV 기반 번호판 형태 감지 시도
    2. 감지 실패 시 여러 위치 추정 (하단 + 중앙)
    
    Returns: [(x1, y1, x2, y2), ...] 마스킹할 영역들
    """
    regions = []
    
    # 1. OpenCV로 번호판 형태 감지 시도
    if use_detection:
        detected = detect_plates_in_vehicle(frame, vehicle_box)
        if detected:
            # 감지된 영역 확장
            for plate in detected:
                expanded = expand_box(plate, expand_ratio, frame.shape)
                regions.append(expanded)
            return regions
    
    # 2. 감지 실패 시 여러 위치 추정
    x1, y1, x2, y2 = vehicle_box[:4]
    height = y2 - y1
    width = x2 - x1
    center_x = (x1 + x2) / 2
    
    plate_width = width * 0.45
    plate_height = height * 0.12
    
    # 위치 1: 하단 범퍼 영역 (기존 방식)
    lower_y1 = y2 - height * 0.30
    lower_y2 = y2 - height * 0.05
    regions.append((
        int(center_x - plate_width / 2),
        int(lower_y1),
        int(center_x + plate_width / 2),
        int(lower_y2)
    ))
    
    # 위치 2: 중앙 트렁크 영역 (승용차 뒷면)
    mid_y1 = y1 + height * 0.35
    mid_y2 = y1 + height * 0.55
    regions.append((
        int(center_x - plate_width / 2),
        int(mid_y1),
        int(center_x + plate_width / 2),
        int(mid_y2)
    ))
    
    # 영역 확장 및 경계 체크
    h, w = frame.shape[:2]
    expanded_regions = []
    for region in regions:
        rx1, ry1, rx2, ry2 = region
        
        # 확장
        rw = rx2 - rx1
        rh = ry2 - ry1
        rx1 = max(0, int(rx1 - rw * expand_ratio))
        rx2 = min(w, int(rx2 + rw * expand_ratio))
        ry1 = max(0, int(ry1 - rh * expand_ratio))
        ry2 = min(h, int(ry2 + rh * expand_ratio))
        
        if rx2 > rx1 and ry2 > ry1:
            expanded_regions.append((rx1, ry1, rx2, ry2))
    
    return expanded_regions


def interpolate_box(box1, box2, alpha):
    """두 박스 사이 선형 보간"""
    if box1 is None:
        return box2
    if box2 is None:
        return box1
    return tuple(int(b1 * (1 - alpha) + b2 * alpha) for b1, b2 in zip(box1[:4], box2[:4]))

