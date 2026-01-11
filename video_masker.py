#!/usr/bin/env python3
"""
비디오 마스킹 클래스 모듈
- VideoMasker: 기본 마스킹 클래스
- VideoMaskerOptimized: 최적화된 마스킹 클래스 (GPU 가속, 멀티스레딩, 트래킹 보간)
"""

import os
import time
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from collections import deque
from threading import Thread, Event
from queue import Queue, Empty

import cv2
import yaml
import numpy as np
from ultralytics import YOLO

from masking_utils import (
    setup_logger, apply_blur, apply_mosaic,
    expand_box, get_plate_region, get_all_plate_regions,
    validate_box, validate_and_clip_box,
    is_front_or_rear_view, get_vehicle_lower_region, get_plate_region_by_view
)
from encoding_utils import (
    get_system_info, get_optimal_settings, build_nvenc_command, print_system_info
)
from high_performance import process_video_high_performance
from two_pass import analyze_video, encode_with_masks, process_video_2pass
from adaptive_color_grade import ColorGrader


# ============================================================
# 디바이스 감지
# ============================================================

def get_optimal_device():
    """최적 디바이스 자동 감지"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available() and platform.processor() == 'arm':
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def setup_cuda_optimization(device='cuda', gpu_id=0):
    """CUDA 최적화 설정"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        
        if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.empty_cache()
        
        return True
    except Exception:
        return False


def create_custom_tracker_config(tracker_type, track_buffer, match_thresh):
    """
    커스텀 트래커 설정 파일 생성
    - track_high_thresh: 높은 신뢰도 임계값 (확실한 감지)
    - track_low_thresh: 낮은 신뢰도 임계값 (기존 트랙 유지용)
    - new_track_thresh: 새 트랙 생성 임계값
    - track_buffer: 감지 누락 시 트랙 유지 프레임 수
    """
    if tracker_type == "bytetrack":
        config = {
            'tracker_type': 'bytetrack',
            'track_high_thresh': 0.4,   # 낮춤: 더 많은 감지 포함
            'track_low_thresh': 0.05,   # 낮춤: 기존 트랙 더 오래 유지
            'new_track_thresh': 0.5,    # 낮춤: 새 트랙 더 쉽게 생성
            'track_buffer': track_buffer,
            'match_thresh': match_thresh,
            'fuse_score': True
        }
    else:
        config = {
            'tracker_type': 'botsort',
            'track_high_thresh': 0.4,
            'track_low_thresh': 0.05,
            'new_track_thresh': 0.5,
            'track_buffer': track_buffer,
            'match_thresh': match_thresh,
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.25,
            'with_reid': False,
            'gmc_method': 'sparseOptFlow',
            'fuse_score': True
        }

    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, prefix=f'{tracker_type}_custom_'
    )
    yaml.dump(config, temp_file, default_flow_style=False)
    temp_file.close()
    return temp_file.name


# ============================================================
# 멀티스레딩 워커
# ============================================================

class FFmpegPipeline:
    """FFmpeg 하드웨어 가속 파이프라인 (NVDEC 디코딩 + NVENC 인코딩)"""
    
    def __init__(self, input_path, output_path, width, height, fps, 
                 use_hwaccel=True, use_hevc=False, start_time=None, end_time=None):
        self.input_path = input_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.use_hwaccel = use_hwaccel
        self.use_hevc = use_hevc
        self.frame_size = width * height * 3
        
        self.decoder = None
        self.encoder = None
        
        # 디코더 명령어 구성
        decode_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']
        
        # 하드웨어 가속 디코딩 (NVDEC) - cuvid 디코더 사용
        if use_hwaccel:
            # HEVC/H.264 cuvid 디코더로 직접 디코딩
            decode_cmd.extend(['-hwaccel', 'cuda', '-c:v', 'hevc_cuvid'])
        
        # 시작/종료 시간 (입력 전에 -ss로 빠른 시크)
        if start_time:
            decode_cmd.extend(['-ss', start_time])
        decode_cmd.extend(['-i', input_path])
        if end_time:
            decode_cmd.extend(['-t', end_time])  # -to 대신 -t (duration)
        
        # 출력 형식 (raw video) - GPU에서 CPU로 전송 후 bgr24 변환
        if use_hwaccel:
            # scale_cuda로 GPU에서 포맷 변환 후 다운로드
            decode_cmd.extend(['-vf', 'scale_cuda=format=bgr24,hwdownload,format=bgr24'])
        else:
            decode_cmd.extend(['-pix_fmt', 'bgr24'])
        
        decode_cmd.extend(['-f', 'rawvideo', '-'])
        
        # 인코더 명령어 구성
        encode_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
        encode_cmd.extend([
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(fps),
            '-i', '-'
        ])
        
        if use_hevc:
            if use_hwaccel:
                # NVENC HEVC
                encode_cmd.extend([
                    '-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                    '-rc', 'vbr', '-cq', '23', '-b:v', '0',
                    '-tag:v', 'hvc1'
                ])
            else:
                encode_cmd.extend(['-c:v', 'libx265', '-preset', 'medium', '-crf', '23'])
        else:
            if use_hwaccel:
                # NVENC H.264
                encode_cmd.extend([
                    '-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'hq',
                    '-rc', 'vbr', '-cq', '23', '-b:v', '0'
                ])
            else:
                encode_cmd.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '23'])
        
        encode_cmd.extend(['-an', output_path])
        
        self.decode_cmd = decode_cmd
        self.encode_cmd = encode_cmd
    
    def start(self):
        """파이프라인 시작"""
        import subprocess
        self.decoder = subprocess.Popen(
            self.decode_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            bufsize=self.frame_size * 10  # 10 프레임 버퍼
        )
        self.encoder = subprocess.Popen(
            self.encode_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_size * 10
        )
        return self
    
    def read_frame(self):
        """프레임 읽기"""
        if self.decoder is None:
            return None
        
        raw_frame = self.decoder.stdout.read(self.frame_size)
        if len(raw_frame) != self.frame_size:
            return None
        
        import numpy as np
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame.copy()  # 쓰기 가능한 복사본
    
    def write_frame(self, frame):
        """프레임 쓰기"""
        if self.encoder is not None:
            self.encoder.stdin.write(frame.tobytes())
    
    def close(self):
        """파이프라인 종료"""
        if self.decoder:
            self.decoder.stdout.close()
            self.decoder.wait()
        if self.encoder:
            self.encoder.stdin.close()
            self.encoder.wait()
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.close()


class NVDECDecoder:
    """
    FFmpeg NVDEC 하드웨어 가속 디코더 (GPU 디코딩)
    - cv2.VideoCapture 대비 2-3배 빠른 디코딩
    - GPU에서 직접 디코딩하여 CPU 부하 최소화
    """

    # 코덱별 cuvid 디코더 매핑
    CUVID_DECODERS = {
        'hevc': 'hevc_cuvid',
        'h264': 'h264_cuvid',
        'h265': 'hevc_cuvid',
        'av1': 'av1_cuvid',
        'vp9': 'vp9_cuvid',
        'vp8': 'vp8_cuvid',
        'mpeg4': 'mpeg4_cuvid',
        'mpeg2video': 'mpeg2_cuvid',
        'mpeg1video': 'mpeg1_cuvid',
        'mjpeg': 'mjpeg_cuvid',
        'vc1': 'vc1_cuvid',
    }

    def __init__(self, input_path, width, height, start_time=None, end_time=None):
        self.input_path = input_path
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.decoder = None

        # 입력 비디오 코덱 감지
        codec = self._detect_codec(input_path)
        cuvid_decoder = self.CUVID_DECODERS.get(codec)

        # 디코더 명령어 구성
        decode_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']

        # NVDEC 하드웨어 가속 (cuvid 디코더 명시적 지정)
        if cuvid_decoder:
            decode_cmd.extend(['-hwaccel', 'cuda', '-c:v', cuvid_decoder])
        else:
            # 지원되지 않는 코덱은 소프트웨어 디코딩
            decode_cmd.extend(['-hwaccel', 'cuda'])

        # 시작 시간 (입력 전에 -ss로 빠른 시크)
        if start_time:
            decode_cmd.extend(['-ss', str(start_time)])

        decode_cmd.extend(['-i', input_path])

        # 종료 시간
        if end_time:
            if start_time:
                duration = end_time - start_time
                decode_cmd.extend(['-t', str(duration)])
            else:
                decode_cmd.extend(['-t', str(end_time)])

        # 출력 포맷 (FFmpeg이 자동으로 GPU->CPU 전송)
        decode_cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ])

        self.decode_cmd = decode_cmd
        self._cuvid_decoder = cuvid_decoder

    def _detect_codec(self, input_path):
        """입력 비디오의 코덱 감지"""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=codec_name',
                 '-of', 'default=noprint_wrappers=1:nokey=1', input_path],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip().lower()
        except Exception:
            return None

    def start(self):
        """디코더 시작"""
        self.decoder = subprocess.Popen(
            self.decode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_size * 32  # 큰 버퍼로 성능 향상
        )
        return self

    def read_frame(self):
        """프레임 읽기"""
        if self.decoder is None:
            return None

        raw_frame = self.decoder.stdout.read(self.frame_size)
        if len(raw_frame) != self.frame_size:
            return None

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame.copy()

    def close(self):
        """디코더 종료"""
        if self.decoder:
            self.decoder.stdout.close()
            self.decoder.terminate()
            self.decoder.wait()
            self.decoder = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


class FrameReader(Thread):
    """비동기 프레임 읽기 스레드"""

    def __init__(self, cap, queue, start_frame, end_frame):
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
        self.queue.put(None)

    def stop(self):
        self.stopped.set()


class FrameWriter(Thread):
    """비동기 프레임 쓰기 스레드"""

    def __init__(self, out, queue):
        super().__init__(daemon=False)  # daemon=False로 변경하여 안전한 종료
        self.out = out
        self.queue = queue
        self.stopped = Event()
        self.finished = Event()  # 완료 이벤트 추가
        self.frames_written = 0

    def run(self):
        pending = {}
        next_frame = 0

        try:
            while not self.stopped.is_set():
                try:
                    item = self.queue.get(timeout=0.1)
                except Empty:
                    continue

                if item is None:
                    # 남은 pending 프레임 모두 쓰기
                    while next_frame in pending:
                        self.out.write(pending.pop(next_frame))
                        self.frames_written += 1
                        next_frame += 1
                    break

                frame_idx, frame = item
                pending[frame_idx] = frame

                while next_frame in pending:
                    self.out.write(pending.pop(next_frame))
                    self.frames_written += 1
                    next_frame += 1
        finally:
            self.finished.set()  # 완료 표시

    def stop(self):
        self.stopped.set()

    def wait_finished(self, timeout=60):
        """완료될 때까지 대기"""
        return self.finished.wait(timeout=timeout)


# ============================================================
# 트래킹 보간기
# ============================================================

class TrackingInterpolator:
    """
    트래킹 결과 보간 관리 (개선된 버전)
    - 감지가 누락된 프레임에서도 마지막 위치 유지
    - 부드러운 속도 기반 예측
    - 깜빡임 방지를 위한 적극적인 보간
    """

    def __init__(self, max_age=30, velocity_smoothing=0.5):
        self.tracks = {}
        self.max_age = max_age
        self.velocity_smoothing = velocity_smoothing  # 속도 예측 제한 비율
        self._next_temp_id = -1  # track_id가 None인 경우 임시 ID

    def update(self, frame_idx, detections):
        """새 감지 결과로 업데이트"""
        seen_ids = set()

        for det in detections:
            track_id = det.get('track_id')

            # track_id가 None인 경우에도 처리 (위치 기반 매칭 시도)
            if track_id is None:
                box = det['box']
                matched_id = self._find_matching_track(box, frame_idx)
                if matched_id is not None:
                    track_id = matched_id
                else:
                    # 새로운 임시 ID 할당
                    track_id = self._next_temp_id
                    self._next_temp_id -= 1

            seen_ids.add(track_id)
            box = det['box']

            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'boxes': deque(maxlen=30),  # 더 많은 이력 유지
                    'last_seen': frame_idx,
                    'type': det.get('type', 'unknown'),
                    'velocity': (0, 0, 0, 0),  # 속도 저장
                    'conf': det.get('conf', 0.5)
                }

            # 속도 계산 및 스무딩
            if len(self.tracks[track_id]['boxes']) > 0:
                prev_frame, prev_box = self.tracks[track_id]['boxes'][-1]
                dt = frame_idx - prev_frame
                if dt > 0 and dt <= 10:  # 합리적인 프레임 간격에서만
                    new_velocity = tuple((b2 - b1) / dt for b1, b2 in zip(prev_box, box))
                    old_velocity = self.tracks[track_id]['velocity']
                    # 스무딩된 속도
                    smoothed_velocity = tuple(
                        old_v * 0.7 + new_v * 0.3
                        for old_v, new_v in zip(old_velocity, new_velocity)
                    )
                    self.tracks[track_id]['velocity'] = smoothed_velocity

            self.tracks[track_id]['boxes'].append((frame_idx, box))
            self.tracks[track_id]['last_seen'] = frame_idx
            self.tracks[track_id]['conf'] = det.get('conf', self.tracks[track_id]['conf'])

        expired = [tid for tid, t in self.tracks.items()
                   if frame_idx - t['last_seen'] > self.max_age]
        for tid in expired:
            del self.tracks[tid]

    def _find_matching_track(self, box, frame_idx, iou_thresh=0.3):
        """위치 기반으로 기존 트랙과 매칭"""
        best_match = None
        best_iou = iou_thresh

        for track_id, track in self.tracks.items():
            if frame_idx - track['last_seen'] > 10:  # 너무 오래된 트랙은 제외
                continue

            if len(track['boxes']) == 0:
                continue

            _, last_box = track['boxes'][-1]
            iou = self._calculate_iou(box, last_box)

            if iou > best_iou:
                best_iou = iou
                best_match = track_id

        return best_match

    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_interpolated(self, frame_idx):
        """현재 프레임에 대한 보간된 박스들 반환 (개선된 버전)"""
        results = []

        for track_id, track in self.tracks.items():
            boxes = track['boxes']
            if len(boxes) == 0:
                continue

            last_frame, last_box = boxes[-1]
            frames_gap = frame_idx - last_frame

            # 현재 프레임에 감지된 경우
            if frames_gap == 0:
                results.append({
                    'track_id': track_id,
                    'box': last_box,
                    'type': track['type'],
                    'interpolated': False
                })
            # 감지가 누락되었지만 max_age 이내인 경우 - 보간
            elif frames_gap > 0 and frames_gap <= self.max_age:
                velocity = track.get('velocity', (0, 0, 0, 0))

                # 속도 기반 예측 (제한된 범위 내에서)
                max_shift = 50 * frames_gap  # 프레임당 최대 50픽셀 이동
                predicted_box = []
                for i, (coord, vel) in enumerate(zip(last_box, velocity)):
                    shift = vel * frames_gap
                    # 이동량 제한
                    shift = max(-max_shift, min(max_shift, shift))
                    predicted_box.append(int(coord + shift))

                # 박스 크기가 너무 변하지 않도록 보정
                orig_w = last_box[2] - last_box[0]
                orig_h = last_box[3] - last_box[1]
                new_w = predicted_box[2] - predicted_box[0]
                new_h = predicted_box[3] - predicted_box[1]

                # 크기가 20% 이상 변하면 원래 크기 유지
                if new_w < orig_w * 0.8 or new_w > orig_w * 1.2:
                    center_x = (predicted_box[0] + predicted_box[2]) // 2
                    predicted_box[0] = center_x - orig_w // 2
                    predicted_box[2] = center_x + orig_w // 2
                if new_h < orig_h * 0.8 or new_h > orig_h * 1.2:
                    center_y = (predicted_box[1] + predicted_box[3]) // 2
                    predicted_box[1] = center_y - orig_h // 2
                    predicted_box[3] = center_y + orig_h // 2

                results.append({
                    'track_id': track_id,
                    'box': tuple(predicted_box),
                    'type': track['type'],
                    'interpolated': True
                })

        return results

    def get_all_active_tracks(self, frame_idx):
        """활성화된 모든 트랙 수 반환 (디버깅용)"""
        return sum(1 for t in self.tracks.values()
                   if frame_idx - t['last_seen'] <= self.max_age)


class PlateTracker:
    """
    차량별 번호판 위치 트래킹 및 보간
    - 차량 track_id별로 번호판 위치를 캐싱
    - 감지 실패 시 이전 위치 사용
    - 부드러운 보간으로 깜빡임 방지
    """
    
    def __init__(self, max_age=60, smoothing=0.7):
        """
        Args:
            max_age: 번호판이 감지되지 않아도 유지할 최대 프레임 수
            smoothing: 위치 스무딩 비율 (0~1, 높을수록 이전 위치 유지)
        """
        self.plate_cache = {}  # vehicle_track_id -> {'plates': [...], 'last_seen': frame_idx}
        self.max_age = max_age
        self.smoothing = smoothing
    
    def update(self, frame_idx, vehicle_track_id, plate_regions):
        """
        차량의 번호판 위치 업데이트
        
        Args:
            frame_idx: 현재 프레임 인덱스
            vehicle_track_id: 차량 트래킹 ID
            plate_regions: 감지된 번호판 영역들 [(x1,y1,x2,y2), ...]
        """
        if vehicle_track_id is None:
            return
        
        if plate_regions and len(plate_regions) > 0:
            # 새로운 번호판 감지됨
            if vehicle_track_id in self.plate_cache:
                # 기존 위치와 스무딩
                old_plates = self.plate_cache[vehicle_track_id]['plates']
                smoothed_plates = []
                
                for i, new_plate in enumerate(plate_regions):
                    if i < len(old_plates):
                        # 이전 위치와 새 위치를 스무딩
                        old = old_plates[i]
                        smoothed = tuple(
                            int(old[j] * self.smoothing + new_plate[j] * (1 - self.smoothing))
                            for j in range(4)
                        )
                        smoothed_plates.append(smoothed)
                    else:
                        smoothed_plates.append(new_plate)
                
                self.plate_cache[vehicle_track_id] = {
                    'plates': smoothed_plates,
                    'last_seen': frame_idx,
                    'detected': True
                }
            else:
                # 새로운 차량
                self.plate_cache[vehicle_track_id] = {
                    'plates': list(plate_regions),
                    'last_seen': frame_idx,
                    'detected': True
                }
        else:
            # 번호판 감지 실패 - 이전 위치 유지
            if vehicle_track_id in self.plate_cache:
                self.plate_cache[vehicle_track_id]['detected'] = False
    
    def get_plates(self, frame_idx, vehicle_track_id, vehicle_box=None):
        """
        차량의 번호판 위치 반환 (캐시된 위치 또는 추정 위치)
        
        Args:
            frame_idx: 현재 프레임 인덱스
            vehicle_track_id: 차량 트래킹 ID
            vehicle_box: 현재 차량 박스 (캐시가 없을 때 위치 추정용)
        
        Returns:
            [(x1, y1, x2, y2), ...] 번호판 영역들
        """
        if vehicle_track_id is None:
            return []
        
        if vehicle_track_id in self.plate_cache:
            cache = self.plate_cache[vehicle_track_id]
            age = frame_idx - cache['last_seen']
            
            if age <= self.max_age:
                return cache['plates']
        
        return []
    
    def cleanup(self, frame_idx):
        """오래된 캐시 정리"""
        expired = [
            tid for tid, cache in self.plate_cache.items()
            if frame_idx - cache['last_seen'] > self.max_age
        ]
        for tid in expired:
            del self.plate_cache[tid]


# ============================================================
# 기본 마스킹 클래스
# ============================================================

class VideoMasker:
    """기본 비디오 마스킹 클래스 (간단하고 빠른 설정용)"""
    
    # COCO 클래스 ID
    PERSON_CLASS = 0
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def __init__(
        self,
        mask_persons: bool = True,
        mask_plates: bool = True,
        mask_type: str = "blur",
        blur_strength: int = 51,
        mosaic_size: int = 15,
        person_confidence: float = 0.4,
        vehicle_confidence: float = 0.3,
        person_expand: float = 0.1,
        plate_expand: float = 0.3,
        plate_detect_mode: str = "auto",  # "auto", "multi", "legacy"
        max_mask_ratio: float = 0.4,  # 프레임 대비 최대 마스킹 영역 비율
    ):
        self.mask_persons = mask_persons
        self.mask_plates = mask_plates
        self.mask_type = mask_type
        self.blur_strength = blur_strength
        self.mosaic_size = mosaic_size
        self.person_confidence = person_confidence
        self.vehicle_confidence = vehicle_confidence
        self.person_expand = person_expand
        self.plate_expand = plate_expand
        self.plate_detect_mode = plate_detect_mode
        self.max_mask_ratio = max_mask_ratio  # 비정상적으로 큰 마스킹 방지

        # 모델 로드
        self.yolo_model = None
        if mask_persons or mask_plates:
            print("YOLOv8 모델 로딩...")
            self.yolo_model = YOLO("yolov8n.pt")
            print(f"   ✅ 사람: {'O' if mask_persons else 'X'}, 번호판: {'O' if mask_plates else 'X'}")
            if mask_plates:
                print(f"   📍 번호판 감지 모드: {plate_detect_mode}")
            print(f"   🛡️ 최대 마스킹 비율: {max_mask_ratio*100:.0f}%")

    def apply_mask(self, frame, x1, y1, x2, y2):
        """마스크 적용"""
        if self.mask_type == "blur":
            return apply_blur(frame, x1, y1, x2, y2, self.blur_strength)
        else:
            return apply_mosaic(frame, x1, y1, x2, y2, self.mosaic_size)

    def safe_apply_mask(self, frame, x1, y1, x2, y2, max_ratio=None):
        """
        안전한 마스크 적용 (비정상적으로 큰 영역 필터링)
        
        Returns:
            frame: 마스크 적용된 프레임 (또는 원본)
            bool: 마스크가 적용되었는지 여부
        """
        if max_ratio is None:
            max_ratio = self.max_mask_ratio
        
        # 박스 검증 및 클리핑
        validated_box = validate_and_clip_box(
            (x1, y1, x2, y2), 
            frame.shape, 
            max_area_ratio=max_ratio,
            min_size=10
        )
        
        if validated_box is None:
            # 비정상적인 박스 - 마스킹 스킵
            return frame, False
        
        x1, y1, x2, y2 = validated_box
        return self.apply_mask(frame, x1, y1, x2, y2), True

    def process_frame(self, frame):
        """단일 프레임 처리"""
        if not self.yolo_model:
            return frame

        classes_to_detect = []
        if self.mask_persons:
            classes_to_detect.append(self.PERSON_CLASS)
        if self.mask_plates:
            classes_to_detect.extend(self.VEHICLE_CLASSES)

        results = self.yolo_model.track(
            frame,
            persist=True,
            conf=min(self.person_confidence, self.vehicle_confidence),
            classes=classes_to_detect,
            verbose=False
        )

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                # 사람 마스킹 (안전한 적용)
                if self.mask_persons and cls == self.PERSON_CLASS and conf >= self.person_confidence:
                    x1, y1, x2, y2 = expand_box(xyxy, self.person_expand, frame.shape)
                    frame, _ = self.safe_apply_mask(frame, x1, y1, x2, y2)

                # 번호판 마스킹 (개선된 방식 + 안전한 적용)
                if self.mask_plates and cls in self.VEHICLE_CLASSES and conf >= self.vehicle_confidence:
                    if self.plate_detect_mode == "legacy":
                        # 기존 방식: 하단 영역만
                        px1, py1, px2, py2 = get_plate_region(xyxy, self.plate_expand)
                        frame, _ = self.safe_apply_mask(frame, px1, py1, px2, py2, max_ratio=0.15)
                    elif self.plate_detect_mode == "lower_body":
                        # v3.4 신규: 앞/뒷면 차량만 하단 전체 마스킹
                        plate_regions = get_plate_region_by_view(
                            frame, xyxy, frame.shape,
                            aspect_threshold=1.4,
                            upper_ratio=0.35,
                            expand_ratio=self.plate_expand
                        )
                        for px1, py1, px2, py2 in plate_regions:
                            frame, _ = self.safe_apply_mask(frame, px1, py1, px2, py2, max_ratio=0.25)
                    else:
                        # 개선된 방식: OpenCV 감지 또는 다중 위치
                        use_detection = (self.plate_detect_mode == "auto")
                        plate_regions = get_all_plate_regions(
                            frame, xyxy,
                            expand_ratio=self.plate_expand,
                            use_detection=use_detection
                        )
                        for px1, py1, px2, py2 in plate_regions:
                            # 번호판은 프레임의 15% 이하여야 함
                            frame, _ = self.safe_apply_mask(frame, px1, py1, px2, py2, max_ratio=0.15)

        return frame

    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        start_time: float = None,
        end_time: float = None,
        max_frames: int = None,
        preview: bool = False,
        preview_scale: float = 0.5,
        use_hevc: bool = False,
        log_file: str = None,
        verbose: bool = False,
    ):
        """비디오 전체 처리"""
        # 로거 설정
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"masking_{timestamp}.log")

        # 입력 파일명에서 prefix 추출
        input_stem = Path(input_path).stem
        log_prefix = input_stem.split('_')[0] if '_' in input_stem else input_stem

        logger = setup_logger(log_file, verbose, prefix=log_prefix)
        logger.info("=" * 60)
        logger.info("마스킹 작업 시작 (기본 모드)")
        logger.info("=" * 60)

        start_total_time = time.time()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        end_frame = min(end_frame, total_frames)
        if max_frames:
            end_frame = min(start_frame + max_frames, end_frame)
        process_frames = end_frame - start_frame

        logger.info(f"입력 파일: {input_path}")
        logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
        logger.info(f"처리 프레임: {process_frames}")

        if output_path is None:
            input_stem = Path(input_path).stem
            suffix = f"_{int(start_time//60)}m-{int(end_time//60)}m" if start_time else ""
            output_path = str(Path(input_path).parent / f"{input_stem}{suffix}_masked.mp4")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # HEVC 사용시 임시 파일
        if use_hevc:
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            logger.info(f"출력 파일: {output_path} (HEVC 예정)")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"출력 파일: {output_path}")

        logger.info(f"설정: 사람={'O' if self.mask_persons else 'X'}, "
                   f"번호판={'O' if self.mask_plates else 'X'}, 방식={self.mask_type}")
        logger.info("-" * 60)

        frame_count = 0
        errors = []

        try:
            while frame_count < process_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                try:
                    frame = self.process_frame(frame)
                    out.write(frame)
                except Exception as e:
                    logger.error(f"프레임 {frame_count} 오류: {e}")
                    errors.append(str(e))
                    out.write(frame)

                if frame_count % 60 == 0:
                    progress = frame_count / process_frames * 100
                    elapsed = time.time() - start_total_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"[{progress:5.1f}%] {frame_count}/{process_frames} | {avg_fps:.1f} fps")

                if preview:
                    display = cv2.resize(frame, (int(width * preview_scale), int(height * preview_scale)))
                    cv2.imshow("Preview (Q to quit)", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.warning("사용자 중단")
                        break

        finally:
            cap.release()
            out.release()
            if preview:
                cv2.destroyAllWindows()

        # HEVC 인코딩
        if use_hevc:
            logger.info("HEVC 인코딩 시작...")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', temp_output,
                '-c:v', 'libx265', '-preset', 'medium', '-crf', '23',
                '-tag:v', 'hvc1', '-an', output_path
            ]
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("HEVC 인코딩 완료!")
                    os.unlink(temp_output)
                else:
                    logger.error(f"HEVC 실패: {result.stderr}")
            except Exception as e:
                logger.error(f"ffmpeg 오류: {e}")

        total_time = time.time() - start_total_time
        logger.info("=" * 60)
        logger.info(f"완료! 출력: {output_path}")
        logger.info(f"소요시간: {total_time/60:.1f}분, 오류: {len(errors)}")
        logger.info("=" * 60)

        return output_path


# ============================================================
# 최적화된 마스킹 클래스
# ============================================================

class VideoMaskerOptimized(VideoMasker):
    """최적화된 비디오 마스킹 클래스 (GPU 가속, 멀티스레딩, 트래킹 보간)"""

    def __init__(
        self,
        mask_persons: bool = True,
        mask_plates: bool = True,
        mask_type: str = "blur",
        blur_strength: int = 51,
        mosaic_size: int = 15,
        person_confidence: float = 0.4,
        vehicle_confidence: float = 0.3,
        person_expand: float = 0.1,
        plate_expand: float = 0.3,
        plate_detect_mode: str = "auto",  # "auto", "multi", "legacy"
        plate_smoothing: float = 0.6,  # 번호판 위치 스무딩 (0~1, 높을수록 안정적)
        max_mask_ratio: float = 0.4,  # 프레임 대비 최대 마스킹 영역 비율
        # 최적화 파라미터
        device: str = "auto",
        detect_interval: int = -1,
        detect_scale: float = -1,
        batch_size: int = -1,
        # NVIDIA GPU 최적화
        gpu_id: int = 0,
        use_fp16: bool = None,
        use_tensorrt: bool = False,
        yolo_model: str = "yolov8n",  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        # 시스템 최적화
        auto_optimize: bool = True,
        queue_size: int = -1,
        high_performance: bool = False,  # 고성능 모드 (FFmpeg 파이프라인)
        # 트래킹 파라미터 (track_buffer: 감지 누락 시 유지할 프레임 수)
        tracker: str = "bytetrack",
        track_buffer: int = 60,  # 기본값 증가: 깜빡임 방지
        match_thresh: float = 0.8,
        iou_thresh: float = 0.5,
        # 하드웨어 가속 디코딩
        use_nvdec: bool = True,  # NVDEC GPU 디코딩 (NVIDIA GPU 필수)
        # 적응형 색보정 옵션
        color_grade: bool = False,  # 적응형 색보정 적용 여부
        color_grade_interval: int = 1000,  # 색보정 분석 간격 (프레임)
        color_grade_smooth: int = 300,  # 색보정 스무딩 윈도우 (프레임)
    ):
        # 부모 클래스의 기본 속성만 설정 (모델 로드 제외)
        self.mask_persons = mask_persons
        self.mask_plates = mask_plates
        self.mask_type = mask_type
        self.blur_strength = blur_strength
        self.mosaic_size = mosaic_size
        self.person_confidence = person_confidence
        self.vehicle_confidence = vehicle_confidence
        self.person_expand = person_expand
        self.plate_expand = plate_expand
        self.plate_detect_mode = plate_detect_mode
        self.plate_smoothing = plate_smoothing
        self.max_mask_ratio = max_mask_ratio
        
        # 시스템 정보 수집 및 자동 최적화
        self.system_info = None
        self.optimal_settings = None
        
        if auto_optimize:
            self.system_info = get_system_info()
            self.optimal_settings = get_optimal_settings(self.system_info)
            print_system_info(self.system_info, self.optimal_settings)
        
        # 디바이스 설정
        if device == "auto":
            self.device = self.optimal_settings['device'] if self.optimal_settings else get_optimal_device()
        else:
            self.device = device

        # 최적화 설정
        if self.optimal_settings:
            self.detect_interval = detect_interval if detect_interval > 0 else self.optimal_settings['detect_interval']
            self.detect_scale = detect_scale if detect_scale > 0 else self.optimal_settings['detect_scale']
            self.batch_size = batch_size if batch_size > 0 else self.optimal_settings['batch_size']
            self.queue_size = queue_size if queue_size > 0 else self.optimal_settings['queue_size']
            self.use_fp16 = use_fp16 if use_fp16 is not None else self.optimal_settings['use_fp16']
            self.num_workers = self.optimal_settings['num_workers']
        else:
            self.detect_interval = detect_interval if detect_interval > 0 else 3
            self.detect_scale = detect_scale if detect_scale > 0 else 0.5
            self.batch_size = batch_size if batch_size > 0 else 4
            self.queue_size = queue_size if queue_size > 0 else 128
            self.use_fp16 = use_fp16 if use_fp16 is not None else False
            self.num_workers = 2
        
        # 고성능 모드 (FFmpeg 파이프라인 + 진정한 배치 추론)
        self.high_performance = high_performance and self.device == 'cuda'
        if self.high_performance:
            # 고성능 모드에서는 배치 크기와 큐 크기 증가
            self.batch_size = max(self.batch_size, 8)
            self.queue_size = max(self.queue_size, 256)
            print(f"   🚀 고성능 모드 활성화 (배치: {self.batch_size}, 큐: {self.queue_size})")
        
        self.tracker_type = tracker
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.iou_thresh = iou_thresh
        self.gpu_id = gpu_id
        self.use_tensorrt = use_tensorrt

        # NVDEC 설정 (CUDA 디바이스일 때만 사용 가능)
        self.use_nvdec = use_nvdec and self.device == 'cuda'
        if self.use_nvdec:
            print(f"   🎬 NVDEC GPU 디코딩 활성화")

        # CUDA 최적화
        if self.device == 'cuda':
            setup_cuda_optimization(self.device, gpu_id)
            if self.use_fp16:
                print("   ⚡ FP16 반정밀도 추론 활성화")

        # 트래커 설정
        self.tracker_config_path = create_custom_tracker_config(tracker, track_buffer, match_thresh)
        print(f"트래커: {tracker} (buffer={track_buffer})")

        # 모델 로드
        self.yolo_model = None
        self.yolo_half = False

        self.yolo_model_name = yolo_model

        if mask_persons or mask_plates:
            model_file = f"{yolo_model}.pt"
            print(f"YOLO 모델 로딩: {model_file} (device={self.device})...")

            if use_tensorrt and self.device == 'cuda':
                tensorrt_path = Path(f"{yolo_model}.engine")
                if tensorrt_path.exists():
                    print(f"   TensorRT 엔진 로딩: {tensorrt_path}")
                    self.yolo_model = YOLO(str(tensorrt_path), task='detect')
                else:
                    print(f"   ⚠️ TensorRT 엔진 없음, PyTorch 모델 사용")
                    self.yolo_model = YOLO(model_file)
            else:
                self.yolo_model = YOLO(model_file)

            # TensorRT 모델은 .to() 호출 불가 (이미 GPU에 컴파일됨)
            if self.device != 'cpu' and not self.use_tensorrt:
                self.yolo_model.to(self.device)

            self.yolo_half = self.use_fp16 and self.device == 'cuda' and not self.use_tensorrt
            print(f"   ✅ 사람: {'O' if mask_persons else 'X'}, 번호판: {'O' if mask_plates else 'X'}")

        # 트래킹 보간기
        self.person_interpolator = TrackingInterpolator(max_age=track_buffer)
        self.vehicle_interpolator = TrackingInterpolator(max_age=track_buffer)
        
        # 번호판 트래커 (깜빡임 방지)
        self.plate_tracker = PlateTracker(max_age=track_buffer * 2, smoothing=self.plate_smoothing)

        # 적응형 색보정 설정
        self.color_grade = color_grade
        self.color_grade_interval = color_grade_interval
        self.color_grade_smooth = color_grade_smooth
        self.color_grader = None  # 비디오 처리 시 초기화

    def init_color_grader(self, video_path: str):
        """적응형 색보정 초기화 (비디오 처리 전 호출)"""
        if self.color_grade:
            print(f"[ColorGrade] 적응형 색보정 활성화")
            print(f"   분석 간격: {self.color_grade_interval} 프레임, 스무딩: {self.color_grade_smooth} 프레임")
            self.color_grader = ColorGrader(
                video_path,
                analyze_interval=self.color_grade_interval,
                smooth_window=self.color_grade_smooth,
                verbose=True
            )

    def apply_color_grade(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """프레임에 적응형 색보정 적용"""
        if self.color_grader is not None:
            return self.color_grader.apply(frame, frame_idx)
        return frame

    def detect_all(self, frame, frame_idx):
        """사람과 차량 모두 감지"""
        detections = {'persons': [], 'vehicles': []}
        h, w = frame.shape[:2]

        if not self.yolo_model:
            return detections

        # 다운스케일
        if self.detect_scale < 1.0:
            detect_frame = cv2.resize(frame, (int(w * self.detect_scale), int(h * self.detect_scale)))
            scale_x = w / detect_frame.shape[1]
            scale_y = h / detect_frame.shape[0]
        else:
            detect_frame = frame
            scale_x = scale_y = 1.0

        classes_to_detect = []
        if self.mask_persons:
            classes_to_detect.append(self.PERSON_CLASS)
        if self.mask_plates:
            classes_to_detect.extend(self.VEHICLE_CLASSES)

        results = self.yolo_model.track(
            detect_frame,
            persist=True,
            conf=min(self.person_confidence, self.vehicle_confidence),
            iou=self.iou_thresh,
            classes=classes_to_detect,
            tracker=self.tracker_config_path,
            device=self.device,
            half=self.yolo_half,
            verbose=False
        )

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = xyxy
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                track_id = int(box.id[0]) if box.id is not None else None

                if self.mask_persons and cls == self.PERSON_CLASS and conf >= self.person_confidence:
                    detections['persons'].append({
                        'box': (x1, y1, x2, y2), 'track_id': track_id,
                        'type': 'person', 'conf': conf
                    })
                
                if self.mask_plates and cls in self.VEHICLE_CLASSES and conf >= self.vehicle_confidence:
                    detections['vehicles'].append({
                        'box': (x1, y1, x2, y2), 'track_id': track_id,
                        'type': 'vehicle', 'class': cls
                    })

        return detections

    def detect_batch(self, frames, frame_indices):
        """
        여러 프레임에 대해 진정한 배치 추론 수행
        GPU 활용률 극대화를 위한 핵심 메서드
        """
        if not self.yolo_model or not frames:
            return [{} for _ in frames]
        
        h, w = frames[0].shape[:2]
        
        # 다운스케일
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
        
        classes_to_detect = []
        if self.mask_persons:
            classes_to_detect.append(self.PERSON_CLASS)
        if self.mask_plates:
            classes_to_detect.extend(self.VEHICLE_CLASSES)
        
        # 진정한 배치 추론 - 모든 프레임을 한번에 GPU로 전송
        results_list = self.yolo_model.track(
            detect_frames,
            persist=True,
            conf=min(self.person_confidence, self.vehicle_confidence),
            iou=self.iou_thresh,
            classes=classes_to_detect,
            tracker=self.tracker_config_path,
            device=self.device,
            half=self.yolo_half,
            verbose=False
        )
        
        all_detections = []
        for i, (results, frame_idx) in enumerate(zip(results_list, frame_indices)):
            detections = {'persons': [], 'vehicles': []}
            
            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    x1, y1, x2, y2 = xyxy
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    track_id = int(box.id[0]) if box.id is not None else None
                    
                    if self.mask_persons and cls == self.PERSON_CLASS and conf >= self.person_confidence:
                        detections['persons'].append({
                            'box': (x1, y1, x2, y2), 'track_id': track_id,
                            'type': 'person', 'conf': conf
                        })
                    
                    if self.mask_plates and cls in self.VEHICLE_CLASSES and conf >= self.vehicle_confidence:
                        detections['vehicles'].append({
                            'box': (x1, y1, x2, y2), 'track_id': track_id,
                            'type': 'vehicle', 'class': cls
                        })
            
            all_detections.append(detections)
        
        return all_detections

    def process_batch_parallel(self, frames, frame_indices):
        """
        배치 프레임 처리 (진정한 배치 추론 + 마스킹)
        """
        # 감지가 필요한 프레임 찾기
        detect_needed = [
            (i, f, idx) for i, (f, idx) in enumerate(zip(frames, frame_indices))
            if idx % self.detect_interval == 0
        ]
        
        # 배치 감지 수행
        if detect_needed:
            detect_frames = [f for _, f, _ in detect_needed]
            detect_indices = [idx for _, _, idx in detect_needed]
            batch_detections = self.detect_batch(detect_frames, detect_indices)
            
            # 보간기 업데이트
            for (i, _, idx), detections in zip(detect_needed, batch_detections):
                self.person_interpolator.update(idx, detections['persons'])
                self.vehicle_interpolator.update(idx, detections['vehicles'])
        
        # 모든 프레임에 마스킹 적용
        results = []
        total_persons = 0
        total_vehicles = 0
        
        for frame, frame_idx in zip(frames, frame_indices):
            persons = self.person_interpolator.get_interpolated(frame_idx)
            vehicles = self.vehicle_interpolator.get_interpolated(frame_idx)
            
            # 사람 마스킹
            if self.mask_persons:
                for person in persons:
                    box = expand_box(person['box'], self.person_expand, frame.shape)
                    frame, _ = self.safe_apply_mask(frame, *box)
                total_persons += len(persons)
            
            # 번호판 마스킹
            if self.mask_plates:
                for vehicle in vehicles:
                    track_id = vehicle.get('track_id')
                    vehicle_box = vehicle['box']
                    
                    if self.plate_detect_mode == "legacy":
                        plate_box = get_plate_region(vehicle_box, self.plate_expand)
                        frame, _ = self.safe_apply_mask(frame, *plate_box, max_ratio=0.15)
                    else:
                        use_detection = (self.plate_detect_mode == "auto")
                        
                        if frame_idx % self.detect_interval == 0:
                            detected_plates = get_all_plate_regions(
                                frame, vehicle_box,
                                expand_ratio=self.plate_expand,
                                use_detection=use_detection
                            )
                            self.plate_tracker.update(frame_idx, track_id, detected_plates)
                        
                        plate_regions = self.plate_tracker.get_plates(frame_idx, track_id, vehicle_box)
                        
                        if not plate_regions:
                            plate_regions = get_all_plate_regions(
                                frame, vehicle_box,
                                expand_ratio=self.plate_expand,
                                use_detection=use_detection
                            )
                        
                        for plate_box in plate_regions:
                            frame, _ = self.safe_apply_mask(frame, *plate_box, max_ratio=0.15)
                
                total_vehicles += len(vehicles)
            
            results.append(frame)
        
        # 캐시 정리
        if frame_indices and frame_indices[-1] % 30 == 0:
            self.plate_tracker.cleanup(frame_indices[-1])
        
        return results, total_persons, total_vehicles

    def process_frame_optimized(self, frame, frame_idx, force_detect=False):
        """단일 프레임 처리 (보간 + 번호판 트래킹 포함)"""
        # 적응형 색보정 적용 (마스킹 전)
        frame = self.apply_color_grade(frame, frame_idx)

        should_detect = force_detect or (frame_idx % self.detect_interval == 0)

        if should_detect:
            detections = self.detect_all(frame, frame_idx)
            self.person_interpolator.update(frame_idx, detections['persons'])
            self.vehicle_interpolator.update(frame_idx, detections['vehicles'])

        persons = self.person_interpolator.get_interpolated(frame_idx)
        vehicles = self.vehicle_interpolator.get_interpolated(frame_idx)

        # 사람 마스킹 (안전한 적용)
        if self.mask_persons:
            for person in persons:
                box = expand_box(person['box'], self.person_expand, frame.shape)
                frame, _ = self.safe_apply_mask(frame, *box)

        # 번호판 마스킹 (트래킹 + 보간으로 깜빡임 방지)
        if self.mask_plates:
            for vehicle in vehicles:
                track_id = vehicle.get('track_id')
                vehicle_box = vehicle['box']

                if self.plate_detect_mode == "legacy":
                    # 기존 방식: 하단 영역만 (트래킹 없음)
                    plate_box = get_plate_region(vehicle_box, self.plate_expand)
                    frame, _ = self.safe_apply_mask(frame, *plate_box, max_ratio=0.15)
                elif self.plate_detect_mode == "lower_body":
                    # v3.4 신규: 앞/뒷면 차량만 하단 전체 마스킹 (안정적)
                    plate_regions = get_plate_region_by_view(
                        frame, vehicle_box, frame.shape,
                        aspect_threshold=1.4,
                        upper_ratio=0.35,
                        expand_ratio=self.plate_expand
                    )
                    for plate_box in plate_regions:
                        frame, _ = self.safe_apply_mask(frame, *plate_box, max_ratio=0.25)
                else:
                    # 개선된 방식: 번호판 트래킹 + 보간
                    plate_regions = []

                    # 감지가 필요한 프레임에서만 새로 감지
                    if should_detect or not vehicle.get('interpolated', False):
                        use_detection = (self.plate_detect_mode == "auto")
                        detected_plates = get_all_plate_regions(
                            frame, vehicle_box,
                            expand_ratio=self.plate_expand,
                            use_detection=use_detection
                        )

                        # 번호판 트래커 업데이트
                        self.plate_tracker.update(frame_idx, track_id, detected_plates)

                    # 캐시된 번호판 위치 사용 (부드러운 보간)
                    plate_regions = self.plate_tracker.get_plates(frame_idx, track_id, vehicle_box)

                    # 캐시가 없으면 현재 감지 결과 사용
                    if not plate_regions:
                        use_detection = (self.plate_detect_mode == "auto")
                        plate_regions = get_all_plate_regions(
                            frame, vehicle_box,
                            expand_ratio=self.plate_expand,
                            use_detection=use_detection
                        )

                    for plate_box in plate_regions:
                        # 번호판은 프레임의 15% 이하여야 함
                        frame, _ = self.safe_apply_mask(frame, *plate_box, max_ratio=0.15)
            
            # 오래된 캐시 정리 (주기적으로)
            if frame_idx % 30 == 0:
                self.plate_tracker.cleanup(frame_idx)

        return frame, len(persons), len(vehicles)

    def process_video_high_performance(
        self,
        input_path: str,
        output_path: str,
        start_time: float = None,
        end_time: float = None,
        use_hevc: bool = False,
        logger = None
    ):
        """
        고성능 모드: 멀티스레딩 파이프라인 + 진정한 배치 추론
        - 비동기 디코딩 (스레드)
        - 진정한 배치 YOLO 추론 (GPU)
        - 비동기 NVENC 인코딩 (스레드)
        
        RTX 4070 SUPER 기준 4K 60fps에서 ~43 fps 달성
        """
        import subprocess
        from threading import Thread
        from queue import Queue, Empty
        
        # 비디오 정보 가져오기
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        process_frames = end_frame - start_frame
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 최적화 설정 가져오기
        encode_settings = self.optimal_settings if self.optimal_settings else {
            'queue_size': self.queue_size,
            'ffmpeg_threads': 16,
            'nvenc_preset': 'p4',
            'nvenc_lookahead': 32,
            'nvenc_surfaces': 16,
            'nvenc_bframes': 4,
            'use_spatial_aq': True,
            'use_temporal_aq': True,
        }

        # 적응형 색보정 초기화
        self.init_color_grader(input_path)

        logger.info(f"고성능 모드 (멀티스레딩 파이프라인 + 배치 GPU 추론)")
        logger.info(f"   [추론] 배치: {self.batch_size}, 감지간격: {self.detect_interval}, FP16: {self.use_fp16}")
        if self.color_grade:
            logger.info(f"   [색보정] 적응형 색보정 활성화 (간격: {self.color_grade_interval}, 스무딩: {self.color_grade_smooth})")
        logger.info(f"   [인코딩] NVENC {encode_settings.get('nvenc_preset', 'p4')}, "
                   f"Lookahead: {encode_settings.get('nvenc_lookahead', 32)}, "
                   f"Surfaces: {encode_settings.get('nvenc_surfaces', 16)}")
        logger.info(f"   [시스템] 큐: {self.queue_size}, 스레드: {encode_settings.get('ffmpeg_threads', 16)}")
        logger.info(f"   처리 프레임: {process_frames:,}")

        # 큐 설정 (대용량 RAM 활용)
        decode_queue = Queue(maxsize=self.queue_size)
        encode_queue = Queue(maxsize=self.queue_size)
        done_decode = [False]
        done_process = [False]

        # 최적화된 NVENC 인코더 시작
        encode_cmd = build_nvenc_command(
            output_path, width, height, fps,
            encode_settings, use_hevc=use_hevc
        )
        logger.info(f"   인코더: {' '.join(encode_cmd[:8])}...")

        # 큰 버퍼로 인코더 시작 (4K 프레임 = ~24MB)
        frame_buffer_size = width * height * 3 * 32  # 32프레임 버퍼
        encoder = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            bufsize=frame_buffer_size
        )
        
        # 통계
        stats = {'processed': 0, 'persons': 0, 'vehicles': 0}
        start_total_time = time.time()
        
        def decoder_thread():
            """비동기 프레임 디코딩"""
            count = 0
            while count < process_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                decode_queue.put((count, frame))
                count += 1
            done_decode[0] = True
        
        def processor_thread():
            """배치 GPU 추론 + 마스킹"""
            batch_frames = []
            batch_indices = []
            
            while True:
                try:
                    item = decode_queue.get(timeout=0.3)
                except Empty:
                    if done_decode[0] and decode_queue.empty():
                        break
                    continue
                
                idx, frame = item
                batch_frames.append(frame)
                batch_indices.append(idx)
                
                if len(batch_frames) >= self.batch_size:
                    # 배치 처리
                    results, n_p, n_v = self.process_batch_parallel(batch_frames, batch_indices)
                    stats['persons'] += n_p
                    stats['vehicles'] += n_v
                    
                    for i, result_frame in enumerate(results):
                        encode_queue.put((batch_indices[i], result_frame))
                    
                    stats['processed'] += len(batch_frames)
                    batch_frames = []
                    batch_indices = []
            
            # 남은 배치 처리
            if batch_frames:
                results, n_p, n_v = self.process_batch_parallel(batch_frames, batch_indices)
                stats['persons'] += n_p
                stats['vehicles'] += n_v
                for i, result_frame in enumerate(results):
                    encode_queue.put((batch_indices[i], result_frame))
                stats['processed'] += len(batch_frames)
            
            done_process[0] = True
        
        def encoder_thread():
            """비동기 프레임 인코딩 (순서 보장)"""
            pending = {}
            next_idx = 0
            
            while True:
                try:
                    item = encode_queue.get(timeout=0.3)
                except Empty:
                    if done_process[0] and encode_queue.empty():
                        break
                    continue
                
                idx, frame = item
                pending[idx] = frame
                
                # 순서대로 쓰기
                while next_idx in pending:
                    encoder.stdin.write(pending.pop(next_idx).tobytes())
                    next_idx += 1
            
            # 남은 프레임 쓰기
            while next_idx in pending:
                encoder.stdin.write(pending.pop(next_idx).tobytes())
                next_idx += 1
        
        # 스레드 시작
        threads = [
            Thread(target=decoder_thread, name='Decoder'),
            Thread(target=processor_thread, name='Processor'),
            Thread(target=encoder_thread, name='Encoder')
        ]
        
        for t in threads:
            t.start()
        
        # 진행 상황 모니터링
        last_log = 0
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            if stats['processed'] - last_log >= 100:
                elapsed = time.time() - start_total_time
                avg_fps = stats['processed'] / elapsed if elapsed > 0 else 0
                progress = stats['processed'] / process_frames * 100
                logger.info(f"[{progress:5.1f}%] {stats['processed']:,}/{process_frames:,} | "
                           f"{avg_fps:.1f} fps | 사람: {stats['persons']}, 차량: {stats['vehicles']}")
                last_log = stats['processed']
        
        # 스레드 종료 대기
        for t in threads:
            t.join()
        
        # 인코더 종료
        encoder.stdin.close()
        encoder.wait()
        cap.release()
        
        total_time = time.time() - start_total_time
        avg_fps = stats['processed'] / total_time if total_time > 0 else 0
        
        logger.info("-" * 60)
        logger.info(f"✅ 완료! 프레임: {stats['processed']:,}, 시간: {total_time/60:.1f}분, FPS: {avg_fps:.1f}")
        logger.info(f"   총 마스킹: 사람 {stats['persons']:,}, 차량 {stats['vehicles']:,}")
        
        # GPU 메모리 정리
        if self.device == 'cuda':
            import torch
            torch.cuda.empty_cache()
        
        return output_path

    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        start_time: float = None,
        end_time: float = None,
        use_hevc: bool = False,
        log_file: str = None,
        verbose: bool = False,
        **kwargs
    ):
        """비디오 전체 처리 (멀티스레딩)"""
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"masking_{timestamp}.log")

        # 입력 파일명에서 prefix 추출 (예: 01-111_4k_420_10bit.mp4 -> 01-111)
        input_stem = Path(input_path).stem
        log_prefix = input_stem.split('_')[0] if '_' in input_stem else input_stem

        logger = setup_logger(log_file, verbose, prefix=log_prefix)
        logger.info("=" * 60)
        logger.info("최적화 마스킹 v3.0 시작")
        logger.info("=" * 60)

        # 출력 경로 결정
        if output_path is None:
            input_stem = Path(input_path).stem
            suffix = f"_{int(start_time//60)}m-{int(end_time//60)}m" if start_time else ""
            output_path = str(Path(input_path).parent / f"{input_stem}{suffix}_masked.mp4")

        # 고성능 모드일 때 FFmpeg 파이프라인 사용
        if self.high_performance:
            return self.process_video_high_performance(
                input_path, output_path, start_time, end_time, use_hevc, logger
            )

        start_total_time = time.time()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        end_frame = min(end_frame, total_frames)
        process_frames = end_frame - start_frame

        logger.info(f"입력: {input_path}")
        logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
        logger.info(f"디바이스: {self.device}, 배치: {self.batch_size}, 감지간격: {self.detect_interval}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if use_hevc:
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logger.info(f"출력: {output_path}")
        logger.info("-" * 60)

        # 멀티스레딩
        read_queue = Queue(maxsize=self.queue_size)
        write_queue = Queue(maxsize=self.queue_size)

        reader = FrameReader(cap, read_queue, start_frame, end_frame)
        writer = FrameWriter(out, write_queue)

        reader.start()
        writer.start()

        processed_count = 0
        total_persons = 0
        total_vehicles = 0
        errors = []
        frame_times = []
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
                    if batch_frames:
                        batch_start = time.time()
                        for frame, idx in zip(batch_frames, batch_indices):
                            frame, n_p, n_v = self.process_frame_optimized(frame, idx)
                            total_persons += n_p
                            total_vehicles += n_v
                            write_queue.put((idx, frame))
                        frame_times.append((time.time() - batch_start) / len(batch_frames))
                    break

                frame_idx, frame = item
                batch_frames.append(frame)
                batch_indices.append(processed_count)
                processed_count += 1

                if len(batch_frames) >= self.batch_size:
                    batch_start = time.time()
                    for frame, idx in zip(batch_frames, batch_indices):
                        try:
                            frame, n_p, n_v = self.process_frame_optimized(frame, idx)
                            total_persons += n_p
                            total_vehicles += n_v
                            write_queue.put((idx, frame))
                        except Exception as e:
                            errors.append(str(e))
                            write_queue.put((idx, frame))
                    frame_times.append((time.time() - batch_start) / len(batch_frames))
                    batch_frames = []
                    batch_indices = []

                if processed_count % 60 == 0:
                    progress = processed_count / process_frames * 100
                    elapsed = time.time() - start_total_time
                    avg_fps = processed_count / elapsed if elapsed > 0 else 0
                    logger.info(f"[{progress:5.1f}%] {processed_count:,}/{process_frames:,} | "
                               f"{avg_fps:.1f} fps | 사람: {total_persons}, 차량: {total_vehicles}")

        except KeyboardInterrupt:
            logger.warning("사용자 중단")

        finally:
            write_queue.put(None)
            reader.stop()
            # writer가 완전히 끝날 때까지 대기
            if not writer.wait_finished(timeout=60):
                logger.warning("Writer 타임아웃, 강제 종료")
                writer.stop()
            writer.join(timeout=5)
            # VideoCapture/Writer 안전하게 해제
            try:
                cap.release()
            except Exception:
                pass
            try:
                out.release()
            except Exception:
                pass

        masking_time = time.time() - start_total_time
        logger.info("-" * 60)
        logger.info(f"마스킹 완료! 프레임: {processed_count}, 시간: {masking_time/60:.1f}분")

        # HEVC 인코딩 (최적화된 설정)
        if use_hevc:
            logger.info("HEVC 인코딩 (NVENC 최적화)...")
            use_nvenc = self.device == 'cuda'

            if use_nvenc:
                # 최적화 설정 적용
                settings = self.optimal_settings if self.optimal_settings else {
                    'nvenc_preset': 'p4',
                    'nvenc_lookahead': 32,
                    'nvenc_surfaces': 16,
                    'nvenc_bframes': 4,
                    'use_spatial_aq': True,
                    'use_temporal_aq': True,
                    'ffmpeg_threads': 16,
                }

                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                    '-i', temp_output,
                    '-c:v', 'hevc_nvenc',
                    '-preset', settings.get('nvenc_preset', 'p4'),
                    '-tune', 'hq',
                    '-rc', 'vbr', '-cq', '23', '-b:v', '0',
                    '-rc-lookahead', str(settings.get('nvenc_lookahead', 32)),
                    '-surfaces', str(settings.get('nvenc_surfaces', 16)),
                    '-spatial-aq', '1', '-aq-strength', '8',
                    '-temporal-aq', '1',
                    '-bf', str(settings.get('nvenc_bframes', 4)),
                    '-b_ref_mode', 'middle',
                    '-tag:v', 'hvc1',
                    '-an', output_path
                ]
            else:
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', temp_output,
                    '-c:v', 'libx265', '-preset', 'medium', '-crf', '23',
                    '-tag:v', 'hvc1', '-an', output_path
                ]

            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0 and use_nvenc:
                    logger.warning("NVENC 실패, libx265로 재시도...")
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-i', temp_output,
                        '-c:v', 'libx265', '-preset', 'medium', '-crf', '23',
                        '-tag:v', 'hvc1', '-an', output_path
                    ]
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info("HEVC 인코딩 완료!")
                    os.unlink(temp_output)
                else:
                    logger.error(f"HEVC 인코딩 실패: {result.stderr}")
            except Exception as e:
                logger.error(f"ffmpeg 오류: {e}")

        # GPU 메모리 정리
        if self.device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        total_time = time.time() - start_total_time
        logger.info("=" * 60)
        logger.info(f"완료! 출력: {output_path}")
        logger.info(f"총 시간: {total_time/60:.1f}분")
        logger.info("=" * 60)

        return output_path

    # ========================================================
    # 2-Pass 모드: 분석과 인코딩 분리
    # ========================================================

    def analyze_video(
        self,
        input_path: str,
        output_json: str = None,
        start_time: float = None,
        end_time: float = None,
        log_file: str = None,
        verbose: bool = False,
    ):
        """
        Pass 1: 비디오 분석만 수행하고 마스킹 좌표를 JSON으로 저장 (멀티스레딩)

        GPU를 YOLO 추론에만 100% 활용
        2-스레드 파이프라인: 디코더 → 분석기

        Returns:
            str: 생성된 JSON 파일 경로
        """
        from threading import Thread
        from queue import Queue, Empty

        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"analyze_{timestamp}.log")

        # 입력 파일명에서 prefix 추출
        input_stem = Path(input_path).stem
        log_prefix = input_stem.split('_')[0] if '_' in input_stem else input_stem

        logger = setup_logger(log_file, verbose, prefix=log_prefix)
        logger.info("=" * 60)
        logger.info("2-Pass 모드: Pass 1 (분석) - 멀티스레딩")
        logger.info("GPU 100% YOLO 추론 전용")
        logger.info("=" * 60)

        start_total_time = time.time()

        # 비디오 정보 가져오기 (cv2로)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        end_frame = min(end_frame, total_frames)
        process_frames = end_frame - start_frame

        queue_size = self.queue_size

        # NVDEC 사용 여부 결정
        use_nvdec = self.use_nvdec

        logger.info(f"입력: {input_path}")
        logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
        logger.info(f"처리 프레임: {process_frames:,}")
        logger.info(f"디바이스: {self.device}, 배치: {self.batch_size}, FP16: {self.use_fp16}")
        logger.info(f"디코딩: {'NVDEC (GPU)' if use_nvdec else 'CPU'}")
        logger.info(f"멀티스레딩: 큐={queue_size}")

        # NVDEC 사용 시 cv2 닫고 NVDEC 디코더 준비
        if use_nvdec:
            cap.release()
            nvdec_decoder = NVDECDecoder(
                input_path, width, height,
                start_time=start_time, end_time=end_time
            )
        else:
            nvdec_decoder = None
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # JSON 출력 경로
        if output_json is None:
            input_stem = Path(input_path).stem
            suffix = f"_{int(start_time//60)}m-{int(end_time//60)}m" if start_time else ""
            output_json = str(Path(input_path).parent / f"{input_stem}{suffix}_masks.json")

        # 마스크 데이터 구조
        mask_data = {
            'version': '2.0',
            'source': str(input_path),
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'start_frame': start_frame,
                'end_frame': end_frame,
            },
            'settings': {
                'mask_persons': self.mask_persons,
                'mask_plates': self.mask_plates,
                'person_expand': self.person_expand,
                'plate_expand': self.plate_expand,
                'plate_detect_mode': self.plate_detect_mode,
            },
            'frames': {}  # frame_idx -> {'persons': [...], 'plates': [...]}
        }

        logger.info("-" * 60)

        # 큐 및 동기화 설정
        decode_queue = Queue(maxsize=queue_size)
        done_decode = [False]

        # 통계
        stats = {'processed': 0, 'persons': 0, 'plates': 0}

        # 참조 캡처
        mask_persons = self.mask_persons
        mask_plates = self.mask_plates
        person_expand = self.person_expand
        plate_expand = self.plate_expand
        plate_detect_mode = self.plate_detect_mode
        max_mask_ratio = self.max_mask_ratio
        detect_interval = self.detect_interval

        def decoder_thread():
            """비동기 프레임 디코딩 (NVDEC 또는 CPU)"""
            nonlocal nvdec_decoder
            count = 0

            if use_nvdec:
                # NVDEC GPU 디코딩
                nvdec_decoder.start()
                while count < process_frames:
                    frame = nvdec_decoder.read_frame()
                    if frame is None:
                        break
                    decode_queue.put((count, frame))
                    count += 1
                nvdec_decoder.close()
            else:
                # CPU 디코딩 (cv2)
                while count < process_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    decode_queue.put((count, frame))
                    count += 1

            done_decode[0] = True

        def analyzer_thread():
            """YOLO 분석 및 마스크 좌표 추출"""
            while True:
                try:
                    item = decode_queue.get(timeout=0.5)
                except Empty:
                    if done_decode[0] and decode_queue.empty():
                        break
                    continue

                frame_idx, frame = item

                # YOLO 감지 (간격에 따라)
                if frame_idx % detect_interval == 0:
                    detections = self.detect_all(frame, frame_idx)
                    self.person_interpolator.update(frame_idx, detections['persons'])
                    self.vehicle_interpolator.update(frame_idx, detections['vehicles'])

                # 보간된 결과 가져오기
                persons = self.person_interpolator.get_interpolated(frame_idx)
                vehicles = self.vehicle_interpolator.get_interpolated(frame_idx)

                frame_masks = {'persons': [], 'plates': []}

                # 사람 마스킹 좌표
                if mask_persons:
                    for person in persons:
                        box = expand_box(person['box'], person_expand, frame.shape)
                        validated = validate_and_clip_box(box, frame.shape, max_mask_ratio)
                        if validated:
                            frame_masks['persons'].append(list(validated))
                            stats['persons'] += 1

                # 번호판 마스킹 좌표
                if mask_plates:
                    for vehicle in vehicles:
                        track_id = vehicle.get('track_id')
                        vehicle_box = vehicle['box']

                        if plate_detect_mode == "legacy":
                            plate_box = get_plate_region(vehicle_box, plate_expand)
                            validated = validate_and_clip_box(plate_box, frame.shape, 0.15)
                            if validated:
                                frame_masks['plates'].append(list(validated))
                                stats['plates'] += 1
                        else:
                            use_detection = (plate_detect_mode == "auto")

                            if frame_idx % detect_interval == 0:
                                detected_plates = get_all_plate_regions(
                                    frame, vehicle_box,
                                    expand_ratio=plate_expand,
                                    use_detection=use_detection
                                )
                                self.plate_tracker.update(frame_idx, track_id, detected_plates)

                            plate_regions = self.plate_tracker.get_plates(frame_idx, track_id, vehicle_box)
                            if not plate_regions:
                                plate_regions = get_all_plate_regions(
                                    frame, vehicle_box,
                                    expand_ratio=plate_expand,
                                    use_detection=use_detection
                                )

                            for plate_box in plate_regions:
                                validated = validate_and_clip_box(plate_box, frame.shape, 0.15)
                                if validated:
                                    frame_masks['plates'].append(list(validated))
                                    stats['plates'] += 1

                # 마스크가 있는 프레임만 저장
                if frame_masks['persons'] or frame_masks['plates']:
                    mask_data['frames'][str(frame_idx)] = frame_masks

                stats['processed'] += 1

                # 캐시 정리
                if frame_idx % 30 == 0:
                    self.plate_tracker.cleanup(frame_idx)

        # 스레드 시작
        decoder = Thread(target=decoder_thread, name='Decoder')
        analyzer = Thread(target=analyzer_thread, name='Analyzer')

        decoder.start()
        analyzer.start()

        # 진행 상황 모니터링
        last_log = 0
        while decoder.is_alive() or analyzer.is_alive():
            time.sleep(1)
            if stats['processed'] - last_log >= 200:
                elapsed = time.time() - start_total_time
                avg_fps = stats['processed'] / elapsed if elapsed > 0 else 0
                progress = stats['processed'] / process_frames * 100
                logger.info(f"[{progress:5.1f}%] {stats['processed']:,}/{process_frames:,} | "
                           f"{avg_fps:.1f} fps | 사람: {stats['persons']:,}, 번호판: {stats['plates']:,}")
                last_log = stats['processed']

        # 스레드 종료 대기
        decoder.join()
        analyzer.join()

        # 리소스 정리 (NVDEC 미사용 시에만)
        if not use_nvdec:
            cap.release()

        # JSON 저장
        mask_data['stats'] = {
            'frames_with_masks': len(mask_data['frames']),
            'total_persons': stats['persons'],
            'total_plates': stats['plates'],
        }

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(mask_data, f, ensure_ascii=False)

        # GPU 메모리 정리
        if self.device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        total_time = time.time() - start_total_time
        avg_fps = stats['processed'] / total_time if total_time > 0 else 0

        logger.info("=" * 60)
        logger.info(f"Pass 1 완료!")
        logger.info(f"   분석 프레임: {stats['processed']:,}")
        logger.info(f"   마스크 프레임: {len(mask_data['frames']):,}")
        logger.info(f"   총 마스킹: 사람 {stats['persons']:,}, 번호판 {stats['plates']:,}")
        logger.info(f"   처리 속도: {avg_fps:.1f} fps")
        logger.info(f"   소요 시간: {total_time/60:.1f}분")
        logger.info(f"   JSON 저장: {output_json}")
        logger.info("=" * 60)

        return output_json

    def encode_with_masks(
        self,
        input_path: str,
        mask_json: str,
        output_path: str = None,
        use_hevc: bool = False,
        log_file: str = None,
        verbose: bool = False,
    ):
        """
        Pass 2: JSON 마스크 데이터를 적용하고 인코딩 (멀티스레딩)

        GPU를 NVENC 인코딩에만 100% 활용
        3-스레드 파이프라인: 디코더 → 마스커 → 인코더

        Returns:
            str: 생성된 비디오 파일 경로
        """
        from threading import Thread
        from queue import Queue, Empty

        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"encode_{timestamp}.log")

        # 입력 파일명에서 prefix 추출
        input_stem = Path(input_path).stem
        log_prefix = input_stem.split('_')[0] if '_' in input_stem else input_stem

        logger = setup_logger(log_file, verbose, prefix=log_prefix)
        logger.info("=" * 60)
        logger.info("2-Pass 모드: Pass 2 (인코딩) - 멀티스레딩")
        logger.info("GPU: NVDEC 디코딩 + NVENC 인코딩")
        logger.info("=" * 60)

        # JSON 마스크 데이터 로드
        with open(mask_json, 'r', encoding='utf-8') as f:
            mask_data = json.load(f)

        video_info = mask_data['video_info']
        frames_data = mask_data['frames']
        stats = mask_data.get('stats', {})

        logger.info(f"마스크 데이터: {mask_json}")
        logger.info(f"   마스크 프레임: {len(frames_data):,}")
        logger.info(f"   총 사람: {stats.get('total_persons', 0):,}")
        logger.info(f"   총 번호판: {stats.get('total_plates', 0):,}")

        start_total_time = time.time()

        # 비디오 정보 가져오기
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {input_path}")

        fps = video_info['fps']
        width = video_info['width']
        height = video_info['height']
        start_frame = video_info['start_frame']
        end_frame = video_info['end_frame']
        process_frames = end_frame - start_frame

        # NVDEC 사용 여부
        use_nvdec = self.use_nvdec
        start_time_sec = start_frame / fps if start_frame > 0 else None
        end_time_sec = end_frame / fps if end_frame < video_info.get('total_frames', end_frame) else None

        logger.info(f"입력: {input_path}")
        logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
        logger.info(f"처리 프레임: {process_frames:,}")
        logger.info(f"디코딩: {'NVDEC (GPU)' if use_nvdec else 'CPU'}")

        if output_path is None:
            input_stem = Path(input_path).stem
            output_path = str(Path(input_path).parent / f"{input_stem}_masked.mp4")

        # NVDEC 사용 시 cv2 닫고 NVDEC 디코더 준비
        if use_nvdec:
            cap.release()
            nvdec_decoder = NVDECDecoder(
                input_path, width, height,
                start_time=start_time_sec, end_time=end_time_sec
            )
        else:
            nvdec_decoder = None
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 최적화된 NVENC 인코더 설정
        encode_settings = self.optimal_settings if self.optimal_settings else {
            'queue_size': 512,
            'ffmpeg_threads': 16,
            'nvenc_preset': 'p4',
            'nvenc_lookahead': 32,
            'nvenc_surfaces': 16,
            'nvenc_bframes': 4,
            'use_spatial_aq': True,
            'use_temporal_aq': True,
        }

        queue_size = encode_settings.get('queue_size', 512)

        encode_cmd = build_nvenc_command(
            output_path, width, height, fps,
            encode_settings, use_hevc=use_hevc
        )

        logger.info(f"출력: {output_path}")
        logger.info(f"코덱: {'HEVC' if use_hevc else 'H.264'} (NVENC)")
        logger.info(f"NVENC: preset={encode_settings.get('nvenc_preset')}, "
                   f"lookahead={encode_settings.get('nvenc_lookahead')}, "
                   f"surfaces={encode_settings.get('nvenc_surfaces')}")
        logger.info(f"멀티스레딩: 큐={queue_size}")
        logger.info("-" * 60)

        # 큐 설정
        decode_queue = Queue(maxsize=queue_size)
        encode_queue = Queue(maxsize=queue_size)
        done_decode = [False]
        done_mask = [False]

        # 통계
        stats_counter = {'processed': 0, 'masks_applied': 0}

        # 큰 버퍼로 인코더 시작
        frame_buffer_size = width * height * 3 * 64
        encoder = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            bufsize=frame_buffer_size
        )

        # 마스킹 설정 캡처
        mask_type = self.mask_type
        blur_strength = self.blur_strength
        mosaic_size = self.mosaic_size

        def decoder_thread():
            """비동기 프레임 디코딩 (NVDEC 또는 CPU)"""
            nonlocal nvdec_decoder
            count = 0

            if use_nvdec:
                # NVDEC GPU 디코딩
                nvdec_decoder.start()
                while count < process_frames:
                    frame = nvdec_decoder.read_frame()
                    if frame is None:
                        break
                    decode_queue.put((count, frame))
                    count += 1
                nvdec_decoder.close()
            else:
                # CPU 디코딩 (cv2)
                while count < process_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    decode_queue.put((count, frame))
                    count += 1

            done_decode[0] = True

        def masker_thread():
            """비동기 마스크 적용"""
            while True:
                try:
                    item = decode_queue.get(timeout=0.5)
                except Empty:
                    if done_decode[0] and decode_queue.empty():
                        break
                    continue

                frame_idx, frame = item
                frame_key = str(frame_idx)

                if frame_key in frames_data:
                    frame_masks = frames_data[frame_key]

                    # 사람 마스킹
                    for box in frame_masks.get('persons', []):
                        x1, y1, x2, y2 = box
                        if mask_type == "blur":
                            frame = apply_blur(frame, x1, y1, x2, y2, blur_strength)
                        else:
                            frame = apply_mosaic(frame, x1, y1, x2, y2, mosaic_size)
                        stats_counter['masks_applied'] += 1

                    # 번호판 마스킹
                    for box in frame_masks.get('plates', []):
                        x1, y1, x2, y2 = box
                        if mask_type == "blur":
                            frame = apply_blur(frame, x1, y1, x2, y2, blur_strength)
                        else:
                            frame = apply_mosaic(frame, x1, y1, x2, y2, mosaic_size)
                        stats_counter['masks_applied'] += 1

                encode_queue.put((frame_idx, frame))
                stats_counter['processed'] += 1

            done_mask[0] = True

        def encoder_thread():
            """비동기 NVENC 인코딩 (순서 보장)"""
            pending = {}
            next_idx = 0

            while True:
                try:
                    item = encode_queue.get(timeout=0.5)
                except Empty:
                    if done_mask[0] and encode_queue.empty():
                        break
                    continue

                frame_idx, frame = item
                pending[frame_idx] = frame

                # 순서대로 인코딩
                while next_idx in pending:
                    encoder.stdin.write(pending.pop(next_idx).tobytes())
                    next_idx += 1

            # 남은 프레임 처리
            while next_idx in pending:
                encoder.stdin.write(pending.pop(next_idx).tobytes())
                next_idx += 1

        # 스레드 시작
        threads = [
            Thread(target=decoder_thread, name='Decoder'),
            Thread(target=masker_thread, name='Masker'),
            Thread(target=encoder_thread, name='Encoder')
        ]

        for t in threads:
            t.start()

        # 진행 상황 모니터링
        last_log = 0
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            if stats_counter['processed'] - last_log >= 200:
                elapsed = time.time() - start_total_time
                avg_fps = stats_counter['processed'] / elapsed if elapsed > 0 else 0
                progress = stats_counter['processed'] / process_frames * 100
                logger.info(f"[{progress:5.1f}%] {stats_counter['processed']:,}/{process_frames:,} | "
                           f"{avg_fps:.1f} fps | 마스크: {stats_counter['masks_applied']:,}")
                last_log = stats_counter['processed']

        # 스레드 종료 대기
        for t in threads:
            t.join()

        # 인코더 종료
        encoder.stdin.close()
        encoder.wait()

        # 리소스 정리 (NVDEC 미사용 시에만)
        if not use_nvdec:
            cap.release()

        total_time = time.time() - start_total_time
        avg_fps = stats_counter['processed'] / total_time if total_time > 0 else 0

        logger.info("=" * 60)
        logger.info(f"Pass 2 완료!")
        logger.info(f"   인코딩 프레임: {stats_counter['processed']:,}")
        logger.info(f"   적용된 마스크: {stats_counter['masks_applied']:,}")
        logger.info(f"   처리 속도: {avg_fps:.1f} fps")
        logger.info(f"   소요 시간: {total_time/60:.1f}분")
        logger.info(f"   출력: {output_path}")
        logger.info("=" * 60)

        return output_path

    def process_video_2pass(
        self,
        input_path: str,
        output_path: str = None,
        start_time: float = None,
        end_time: float = None,
        use_hevc: bool = False,
        keep_json: bool = False,
        log_file: str = None,
        verbose: bool = False,
    ):
        """
        2-Pass 모드 전체 실행

        Pass 1: 분석 (GPU → YOLO 100%)
        Pass 2: 인코딩 (GPU → NVENC 100%)

        각 Pass에서 GPU를 최대한 활용
        """
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"2pass_{timestamp}.log")

        logger = setup_logger(log_file, verbose)
        logger.info("=" * 60)
        logger.info("2-Pass 모드 시작")
        logger.info("=" * 60)

        start_total = time.time()

        # Pass 1: 분석
        logger.info("\n>>> Pass 1: 분석 시작")
        json_path = self.analyze_video(
            input_path,
            start_time=start_time,
            end_time=end_time,
            log_file=log_file,
            verbose=verbose
        )

        # Pass 2: 인코딩
        logger.info("\n>>> Pass 2: 인코딩 시작")
        result_path = self.encode_with_masks(
            input_path,
            json_path,
            output_path=output_path,
            use_hevc=use_hevc,
            log_file=log_file,
            verbose=verbose
        )

        # JSON 정리
        if not keep_json:
            os.unlink(json_path)
            logger.info(f"임시 JSON 삭제: {json_path}")

        total_time = time.time() - start_total
        logger.info("=" * 60)
        logger.info(f"2-Pass 완료! 총 시간: {total_time/60:.1f}분")
        logger.info(f"출력: {result_path}")
        logger.info("=" * 60)

        return result_path

