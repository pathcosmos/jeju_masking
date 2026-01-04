#!/usr/bin/env python3
"""
고성능 모드 모듈
- 멀티스레딩 파이프라인
- 진정한 배치 GPU 추론
- NVENC 실시간 인코딩

1-Pass로 분석+마스킹+인코딩 동시 처리
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime
from threading import Thread
from queue import Queue, Empty

import cv2

from masking_utils import setup_logger
from encoding_utils import build_nvenc_command


def process_video_high_performance(
    masker,
    input_path: str,
    output_path: str,
    start_time: float = None,
    end_time: float = None,
    use_hevc: bool = False,
    logger=None
):
    """
    고성능 모드: 멀티스레딩 파이프라인 + 진정한 배치 추론

    - 비동기 디코딩 (스레드)
    - 진정한 배치 YOLO 추론 (GPU)
    - 비동기 NVENC 인코딩 (스레드)

    RTX 4070 SUPER 기준 4K 60fps에서 ~48 fps 달성

    Args:
        masker: VideoMaskerOptimized 인스턴스
        input_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        use_hevc: HEVC 인코딩 사용
        logger: 로거 인스턴스

    Returns:
        str: 생성된 비디오 파일 경로
    """
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
    encode_settings = masker.optimal_settings if masker.optimal_settings else {
        'queue_size': masker.queue_size,
        'ffmpeg_threads': 16,
        'nvenc_preset': 'p4',
        'nvenc_lookahead': 32,
        'nvenc_surfaces': 16,
        'nvenc_bframes': 4,
        'use_spatial_aq': True,
        'use_temporal_aq': True,
    }

    logger.info(f"고성능 모드 (멀티스레딩 파이프라인 + 배치 GPU 추론)")
    logger.info(f"   [추론] 배치: {masker.batch_size}, 감지간격: {masker.detect_interval}, FP16: {masker.use_fp16}")
    logger.info(f"   [인코딩] NVENC {encode_settings.get('nvenc_preset', 'p4')}, "
               f"Lookahead: {encode_settings.get('nvenc_lookahead', 32)}, "
               f"Surfaces: {encode_settings.get('nvenc_surfaces', 16)}")
    logger.info(f"   [시스템] 큐: {masker.queue_size}, 스레드: {encode_settings.get('ffmpeg_threads', 16)}")
    logger.info(f"   처리 프레임: {process_frames:,}")

    # 큐 설정 (대용량 RAM 활용)
    decode_queue = Queue(maxsize=masker.queue_size)
    encode_queue = Queue(maxsize=masker.queue_size)
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

            if len(batch_frames) >= masker.batch_size:
                # 배치 처리
                results, n_p, n_v = masker.process_batch_parallel(batch_frames, batch_indices)
                stats['persons'] += n_p
                stats['vehicles'] += n_v

                for i, result_frame in enumerate(results):
                    encode_queue.put((batch_indices[i], result_frame))

                stats['processed'] += len(batch_frames)
                batch_frames = []
                batch_indices = []

        # 남은 프레임 처리
        if batch_frames:
            results, n_p, n_v = masker.process_batch_parallel(batch_frames, batch_indices)
            stats['persons'] += n_p
            stats['vehicles'] += n_v

            for i, result_frame in enumerate(results):
                encode_queue.put((batch_indices[i], result_frame))

            stats['processed'] += len(batch_frames)

        done_process[0] = True

    def encoder_thread():
        """비동기 NVENC 인코딩 (순서 보장)"""
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

            # 순서대로 인코딩
            while next_idx in pending:
                encoder.stdin.write(pending.pop(next_idx).tobytes())
                next_idx += 1

        # 남은 프레임 처리
        while pending:
            if next_idx in pending:
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
        if stats['processed'] - last_log >= 128:
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
    logger.info(f"완료! 프레임: {stats['processed']:,}, 시간: {total_time/60:.1f}분, FPS: {avg_fps:.1f}")
    logger.info(f"   총 마스킹: 사람 {stats['persons']}, 차량 {stats['vehicles']}")

    return output_path
