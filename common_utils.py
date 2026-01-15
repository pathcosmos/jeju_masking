#!/usr/bin/env python3
"""
공통 유틸리티 모듈
- 비디오 디코딩/인코딩 파이프라인
- 멀티스레딩 워커
- 시스템 정보 및 최적화 설정

마스킹과 필터 모듈에서 공통으로 사용
"""

import os
import time
import subprocess
from threading import Thread, Event
from queue import Queue, Empty

import cv2
import numpy as np


# ============================================================
# NVDEC 디코더 (GPU 가속 디코딩)
# ============================================================

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


# ============================================================
# 멀티스레딩 워커
# ============================================================

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
        super().__init__(daemon=False)
        self.out = out
        self.queue = queue
        self.stopped = Event()
        self.finished = Event()
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
            self.finished.set()

    def stop(self):
        self.stopped.set()

    def wait_finished(self, timeout=60):
        """완료될 때까지 대기"""
        return self.finished.wait(timeout=timeout)


# ============================================================
# 비디오 정보 유틸리티
# ============================================================

def get_video_info(video_path):
    """비디오 정보 가져오기"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
    
    cap.release()
    return info


def calculate_frame_range(video_info, start_time=None, end_time=None, max_frames=None):
    """프레임 범위 계산"""
    fps = video_info['fps']
    total_frames = video_info['total_frames']
    
    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames
    end_frame = min(end_frame, total_frames)
    
    if max_frames:
        end_frame = min(start_frame + max_frames, end_frame)
    
    return start_frame, end_frame


# ============================================================
# 인코더 파이프라인
# ============================================================

class NVENCEncoder:
    """NVENC 하드웨어 가속 인코더"""
    
    def __init__(self, output_path, width, height, fps, settings, use_hevc=False):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3
        self.encoder = None
        
        # 인코딩 명령어 구성
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(fps),
            '-thread_queue_size', str(settings.get('queue_size', 512)),
            '-i', '-'
        ]

        threads = settings.get('ffmpeg_threads', 0)
        if threads > 0:
            cmd.extend(['-threads', str(threads)])

        codec = 'hevc_nvenc' if use_hevc else 'h264_nvenc'
        cmd.extend(['-c:v', codec])

        preset = settings.get('nvenc_preset', 'p4')
        cmd.extend(['-preset', preset, '-tune', 'hq'])

        cmd.extend(['-rc', 'vbr', '-cq', '23', '-b:v', '0'])

        lookahead = settings.get('nvenc_lookahead', 32)
        if lookahead > 0:
            cmd.extend(['-rc-lookahead', str(lookahead)])

        surfaces = settings.get('nvenc_surfaces', 8)
        cmd.extend(['-surfaces', str(surfaces)])

        if settings.get('use_spatial_aq', True):
            cmd.extend(['-spatial-aq', '1', '-aq-strength', '8'])
        if settings.get('use_temporal_aq', True):
            cmd.extend(['-temporal-aq', '1'])

        bframes = settings.get('nvenc_bframes', 3)
        if bframes > 0:
            cmd.extend(['-bf', str(bframes), '-b_ref_mode', 'middle'])

        # 출력 픽셀 포맷 명시 (색상 왜곡 방지)
        cmd.extend(['-pix_fmt', 'yuv420p'])

        if use_hevc:
            cmd.extend(['-tag:v', 'hvc1'])

        cmd.extend(['-an', output_path])
        
        self.encode_cmd = cmd
    
    def start(self):
        """인코더 시작"""
        self.encoder = subprocess.Popen(
            self.encode_cmd,
            stdin=subprocess.PIPE,
            bufsize=self.frame_size * 32
        )
        return self
    
    def write_frame(self, frame):
        """프레임 쓰기"""
        if self.encoder is not None:
            self.encoder.stdin.write(frame.tobytes())
    
    def close(self):
        """인코더 종료"""
        if self.encoder:
            self.encoder.stdin.close()
            self.encoder.wait()
            self.encoder = None
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.close()


# ============================================================
# 기본 인코딩 설정
# ============================================================

def get_default_encode_settings():
    """기본 인코딩 설정"""
    return {
        'queue_size': 512,
        'ffmpeg_threads': 16,
        'nvenc_preset': 'p4',
        'nvenc_lookahead': 32,
        'nvenc_surfaces': 16,
        'nvenc_bframes': 4,
        'use_spatial_aq': True,
        'use_temporal_aq': True,
    }
