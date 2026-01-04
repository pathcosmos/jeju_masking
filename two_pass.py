#!/usr/bin/env python3
"""
2-Pass 모드 모듈
- Pass 1: 분석 (GPU 100% YOLO 추론)
- Pass 2: 인코딩 (GPU 100% NVENC 인코딩)

마스크 좌표를 JSON으로 저장하여 재작업 지원
"""

import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from threading import Thread
from queue import Queue, Empty

import cv2

from masking_utils import (
    setup_logger, apply_blur, apply_mosaic,
    expand_box, get_plate_region, get_all_plate_regions,
    validate_and_clip_box
)
from encoding_utils import build_nvenc_command


def analyze_video(
    masker,
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

    Args:
        masker: VideoMaskerOptimized 인스턴스
        input_path: 입력 비디오 경로
        output_json: 출력 JSON 경로
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        log_file: 로그 파일 경로
        verbose: 상세 로깅

    Returns:
        str: 생성된 JSON 파일 경로
    """
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = str(Path(input_path).parent / f"analyze_{timestamp}.log")

    logger = setup_logger(log_file, verbose)
    logger.info("=" * 60)
    logger.info("2-Pass 모드: Pass 1 (분석) - 멀티스레딩")
    logger.info("GPU 100% YOLO 추론 전용")
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
    process_frames = end_frame - start_frame

    queue_size = masker.queue_size

    logger.info(f"입력: {input_path}")
    logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
    logger.info(f"처리 프레임: {process_frames:,}")
    logger.info(f"디바이스: {masker.device}, 배치: {masker.batch_size}, FP16: {masker.use_fp16}")
    logger.info(f"멀티스레딩: 큐={queue_size}")

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
            'mask_persons': masker.mask_persons,
            'mask_plates': masker.mask_plates,
            'person_expand': masker.person_expand,
            'plate_expand': masker.plate_expand,
            'plate_detect_mode': masker.plate_detect_mode,
        },
        'frames': {}
    }

    logger.info("-" * 60)

    # 큐 및 동기화 설정
    decode_queue = Queue(maxsize=queue_size)
    done_decode = [False]

    # 통계
    stats = {'processed': 0, 'persons': 0, 'plates': 0}

    # 참조 캡처
    mask_persons = masker.mask_persons
    mask_plates = masker.mask_plates
    person_expand = masker.person_expand
    plate_expand = masker.plate_expand
    plate_detect_mode = masker.plate_detect_mode
    max_mask_ratio = masker.max_mask_ratio
    detect_interval = masker.detect_interval

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
                detections = masker.detect_all(frame, frame_idx)
                masker.person_interpolator.update(frame_idx, detections['persons'])
                masker.vehicle_interpolator.update(frame_idx, detections['vehicles'])

            # 보간된 결과 가져오기
            persons = masker.person_interpolator.get_interpolated(frame_idx)
            vehicles = masker.vehicle_interpolator.get_interpolated(frame_idx)

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
                            masker.plate_tracker.update(frame_idx, track_id, detected_plates)

                        plate_regions = masker.plate_tracker.get_plates(frame_idx, track_id, vehicle_box)
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
                masker.plate_tracker.cleanup(frame_idx)

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
    if masker.device == 'cuda':
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
    masker,
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

    Args:
        masker: VideoMaskerOptimized 인스턴스
        input_path: 입력 비디오 경로
        mask_json: 마스크 JSON 경로
        output_path: 출력 비디오 경로
        use_hevc: HEVC 인코딩 사용
        log_file: 로그 파일 경로
        verbose: 상세 로깅

    Returns:
        str: 생성된 비디오 파일 경로
    """
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = str(Path(input_path).parent / f"encode_{timestamp}.log")

    logger = setup_logger(log_file, verbose)
    logger.info("=" * 60)
    logger.info("2-Pass 모드: Pass 2 (인코딩) - 멀티스레딩")
    logger.info("GPU 100% NVENC 인코딩 전용")
    logger.info("=" * 60)

    # JSON 마스크 데이터 로드
    with open(mask_json, 'r', encoding='utf-8') as f:
        mask_data = json.load(f)

    video_info = mask_data['video_info']
    frames_data = mask_data['frames']
    json_stats = mask_data.get('stats', {})

    logger.info(f"마스크 데이터: {mask_json}")
    logger.info(f"   마스크 프레임: {len(frames_data):,}")
    logger.info(f"   총 사람: {json_stats.get('total_persons', 0):,}")
    logger.info(f"   총 번호판: {json_stats.get('total_plates', 0):,}")

    start_total_time = time.time()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {input_path}")

    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    start_frame = video_info['start_frame']
    end_frame = video_info['end_frame']
    process_frames = end_frame - start_frame

    logger.info(f"입력: {input_path}")
    logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
    logger.info(f"처리 프레임: {process_frames:,}")

    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{input_stem}_masked.mp4")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 최적화된 NVENC 인코더 설정
    encode_settings = masker.optimal_settings if masker.optimal_settings else {
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
    mask_type = masker.mask_type
    blur_strength = masker.blur_strength
    mosaic_size = masker.mosaic_size

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
    masker,
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

    Args:
        masker: VideoMaskerOptimized 인스턴스
        input_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        use_hevc: HEVC 인코딩 사용
        keep_json: JSON 파일 유지
        log_file: 로그 파일 경로
        verbose: 상세 로깅

    Returns:
        str: 생성된 비디오 파일 경로
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
    json_path = analyze_video(
        masker,
        input_path,
        start_time=start_time,
        end_time=end_time,
        log_file=log_file,
        verbose=verbose
    )

    # Pass 2: 인코딩
    logger.info("\n>>> Pass 2: 인코딩 시작")
    result_path = encode_with_masks(
        masker,
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
