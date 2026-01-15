#!/usr/bin/env python3
"""
비디오 필터 처리기 (독립 모듈)
- 적응형 색보정 (Adaptive Color Grading)
- 마스킹과 완전히 분리된 독립 모듈

사용 예시:
  # 기본 색보정 적용
  python filter_video.py video.mp4

  # 분석만 수행 (파라미터 JSON 저장)
  python filter_video.py video.mp4 --analyze-only

  # 기존 파라미터로 색보정 적용
  python filter_video.py video.mp4 --params video_params.json

  # HEVC 인코딩
  python filter_video.py video.mp4 --hevc

  # 고성능 모드 (GPU 가속)
  python filter_video.py video.mp4 --high-performance
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock
from queue import Queue, Empty

import cv2
import numpy as np

from masking_utils import setup_logger
from common_utils import (
    NVDECDecoder, NVENCEncoder, FrameReader, FrameWriter,
    get_video_info, calculate_frame_range, get_default_encode_settings
)
from adaptive_color_grade import (
    ColorGrader, CorrectionParams, analyze_video as analyze_color,
    interpolate_params, apply_correction, apply_correction_gpu
)
from encoding_utils import get_system_info, get_optimal_settings, print_system_info


class VideoFilter:
    """
    비디오 필터 처리 클래스
    - 적응형 색보정
    - 마스킹과 완전히 분리된 독립 모듈
    """

    def __init__(
        self,
        # 색보정 옵션
        color_grade: bool = True,
        cg_interval: int = 1000,
        cg_smooth: int = 300,
        # 시스템 옵션
        use_gpu: bool = True,
        use_nvdec: bool = True,
        high_performance: bool = False,
        auto_optimize: bool = True,
    ):
        self.color_grade = color_grade
        self.cg_interval = cg_interval
        self.cg_smooth = cg_smooth
        self.use_gpu = use_gpu
        self.use_nvdec = use_nvdec
        self.high_performance = high_performance

        # 시스템 정보 및 최적화 설정
        self.system_info = None
        self.optimal_settings = None

        if auto_optimize:
            self.system_info = get_system_info()
            self.optimal_settings = get_optimal_settings(self.system_info)
            print_system_info(self.system_info, self.optimal_settings)

        # CUDA 확인
        self.device = 'cpu'
        if self.optimal_settings:
            self.device = self.optimal_settings.get('device', 'cpu')
        
        # NVDEC는 CUDA 디바이스에서만 사용
        self.use_nvdec = use_nvdec and self.device == 'cuda'
        
        # 큐 크기
        self.queue_size = 512
        if self.optimal_settings:
            self.queue_size = self.optimal_settings.get('queue_size', 512)

        print(f"[VideoFilter] 초기화 완료")
        print(f"   디바이스: {self.device}")
        print(f"   NVDEC: {'활성화' if self.use_nvdec else '비활성화'}")
        print(f"   색보정: {'활성화' if self.color_grade else '비활성화'}")
        if self.color_grade:
            print(f"   분석 간격: {self.cg_interval} 프레임")
            print(f"   스무딩: {self.cg_smooth} 프레임")

    def analyze_video(
        self,
        input_path: str,
        output_json: str = None,
        start_time: float = None,
        end_time: float = None,
        log_file: str = None,
        verbose: bool = False,
    ) -> str:
        """
        비디오 분석 (색보정 파라미터 추출)
        
        Returns:
            str: 생성된 JSON 파일 경로
        """
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"filter_analyze_{timestamp}.log")

        input_stem = Path(input_path).stem
        log_prefix = input_stem.split('_')[0] if '_' in input_stem else input_stem

        logger = setup_logger(log_file, verbose, prefix=log_prefix)
        logger.info("=" * 60)
        logger.info("비디오 필터 분석 시작")
        logger.info("=" * 60)

        start_total_time = time.time()

        # 비디오 정보
        video_info = get_video_info(input_path)
        fps = video_info['fps']
        width = video_info['width']
        height = video_info['height']
        total_frames = video_info['total_frames']

        start_frame, end_frame = calculate_frame_range(video_info, start_time, end_time)
        process_frames = end_frame - start_frame

        logger.info(f"입력: {input_path}")
        logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
        logger.info(f"처리 프레임: {process_frames:,}")
        logger.info(f"분석 간격: {self.cg_interval} 프레임")

        # JSON 출력 경로
        if output_json is None:
            suffix = f"_{int(start_time//60)}m-{int(end_time//60)}m" if start_time else ""
            output_json = str(Path(input_path).parent / f"{input_stem}{suffix}_filter_params.json")

        # 색보정 분석
        logger.info("-" * 60)
        logger.info("색보정 파라미터 분석 중...")

        keyframes, analyses = analyze_color(input_path, self.cg_interval, verbose)
        params_list = interpolate_params(keyframes, analyses, total_frames, self.cg_smooth)

        # JSON 저장
        filter_data = {
            'version': '1.0',
            'type': 'color_grade',
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
                'interval': self.cg_interval,
                'smooth_window': self.cg_smooth,
            },
            'keyframes': len(keyframes),
            'params': params_list,
        }

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(filter_data, f, ensure_ascii=False)

        total_time = time.time() - start_total_time

        logger.info("=" * 60)
        logger.info(f"분석 완료!")
        logger.info(f"   키프레임: {len(keyframes):,}")
        logger.info(f"   파라미터: {len(params_list):,} 프레임")
        logger.info(f"   소요 시간: {total_time:.1f}초")
        logger.info(f"   JSON 저장: {output_json}")
        logger.info("=" * 60)

        return output_json

    def apply_filter(
        self,
        input_path: str,
        output_path: str = None,
        params_json: str = None,
        start_time: float = None,
        end_time: float = None,
        use_hevc: bool = False,
        log_file: str = None,
        verbose: bool = False,
    ) -> str:
        """
        비디오에 필터 적용 (색보정)
        
        Returns:
            str: 생성된 비디오 파일 경로
        """
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"filter_apply_{timestamp}.log")

        input_stem = Path(input_path).stem
        log_prefix = input_stem.split('_')[0] if '_' in input_stem else input_stem

        logger = setup_logger(log_file, verbose, prefix=log_prefix)
        logger.info("=" * 60)
        logger.info("비디오 필터 적용 시작")
        logger.info("=" * 60)

        start_total_time = time.time()

        # 비디오 정보
        video_info = get_video_info(input_path)
        fps = video_info['fps']
        width = video_info['width']
        height = video_info['height']
        total_frames = video_info['total_frames']

        start_frame, end_frame = calculate_frame_range(video_info, start_time, end_time)
        process_frames = end_frame - start_frame

        # 출력 경로
        if output_path is None:
            suffix = f"_{int(start_time//60)}m-{int(end_time//60)}m" if start_time else ""
            output_path = str(Path(input_path).parent / f"{input_stem}{suffix}_filtered.mp4")

        logger.info(f"입력: {input_path}")
        logger.info(f"출력: {output_path}")
        logger.info(f"해상도: {width}x{height}, FPS: {fps:.2f}")
        logger.info(f"처리 프레임: {process_frames:,}")

        # 파라미터 로드 또는 분석
        if params_json and Path(params_json).exists():
            logger.info(f"파라미터 로드: {params_json}")
            with open(params_json, 'r', encoding='utf-8') as f:
                filter_data = json.load(f)
                params_list = filter_data['params']
        else:
            logger.info("파라미터 분석 중...")
            keyframes, analyses = analyze_color(input_path, self.cg_interval, verbose)
            params_list = interpolate_params(keyframes, analyses, total_frames, self.cg_smooth)

        # 인코딩 설정
        encode_settings = self.optimal_settings if self.optimal_settings else get_default_encode_settings()

        logger.info(f"디바이스: {self.device}")
        logger.info(f"NVDEC: {'활성화' if self.use_nvdec else '비활성화'}")
        logger.info(f"NVENC: {encode_settings.get('nvenc_preset', 'p4')}")
        logger.info("-" * 60)

        if self.high_performance:
            return self._apply_filter_high_performance(
                input_path, output_path, params_list,
                start_frame, end_frame, process_frames,
                width, height, fps, encode_settings, use_hevc, logger
            )
        else:
            return self._apply_filter_standard(
                input_path, output_path, params_list,
                start_frame, end_frame, process_frames,
                width, height, fps, encode_settings, use_hevc, logger
            )

    def _apply_filter_standard(
        self, input_path, output_path, params_list,
        start_frame, end_frame, process_frames,
        width, height, fps, encode_settings, use_hevc, logger
    ):
        """표준 모드 필터 적용"""
        start_total_time = time.time()

        cap = cv2.VideoCapture(input_path)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        processed = 0
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # 색보정 적용
            if frame_idx < len(params_list):
                params = CorrectionParams(**params_list[frame_idx])
                if self.use_gpu:
                    frame = apply_correction_gpu(frame, params)
                else:
                    frame = apply_correction(frame, params)

            out.write(frame)
            processed += 1

            if processed % 100 == 0:
                progress = processed / process_frames * 100
                elapsed = time.time() - start_total_time
                avg_fps = processed / elapsed if elapsed > 0 else 0
                logger.info(f"[{progress:5.1f}%] {processed:,}/{process_frames:,} | {avg_fps:.1f} fps")

        cap.release()
        out.release()

        total_time = time.time() - start_total_time
        logger.info("=" * 60)
        logger.info(f"완료! 프레임: {processed:,}, 시간: {total_time/60:.1f}분")
        logger.info(f"출력: {output_path}")
        logger.info("=" * 60)

        return output_path

    def _apply_filter_high_performance(
        self, input_path, output_path, params_list,
        start_frame, end_frame, process_frames,
        width, height, fps, encode_settings, use_hevc, logger
    ):
        """
        최대 성능 모드 필터 적용
        - 멀티 필터 워커 (CPU 코어 활용)
        - GPU LUT 캐시 (파라미터 변화 시에만 재생성)
        - 대용량 큐 + 배치 처리
        - NVDEC/NVENC 하드웨어 가속
        """
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor
        
        start_total_time = time.time()
        
        # 시스템 자원 확인
        cpu_count = mp.cpu_count()
        num_filter_workers = min(cpu_count - 2, 12)  # 디코더/인코더용 2개 제외
        
        logger.info("=" * 60)
        logger.info("🚀 최대 성능 모드")
        logger.info(f"   CPU 코어: {cpu_count}")
        logger.info(f"   필터 워커: {num_filter_workers}")
        logger.info(f"   큐 크기: {self.queue_size * 2}")
        logger.info("=" * 60)

        # 대용량 큐 설정
        queue_size = self.queue_size * 2  # 1024
        decode_queue = Queue(maxsize=queue_size)
        filter_queue = Queue(maxsize=queue_size)
        encode_queue = Queue(maxsize=queue_size)
        
        done_decode = [False]
        done_filter = [0]  # 완료된 필터 워커 수
        filter_lock = Lock()

        # 통계
        stats = {'processed': 0, 'decoded': 0, 'encoded': 0}
        stats_lock = Lock()

        # NVDEC 디코더 또는 cv2
        use_nvdec = self.use_nvdec
        start_time_sec = start_frame / fps if start_frame > 0 else None
        end_time_sec = end_frame / fps

        if use_nvdec:
            nvdec_decoder = NVDECDecoder(
                input_path, width, height,
                start_time=start_time_sec, end_time=end_time_sec
            )
        else:
            nvdec_decoder = None
            cap = cv2.VideoCapture(input_path)
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # NVENC 인코더 (빠른 프리셋)
        fast_settings = encode_settings.copy()
        fast_settings['nvenc_preset'] = 'p4'  # 속도 우선
        fast_settings['lookahead'] = 16  # 줄임
        
        encoder = NVENCEncoder(output_path, width, height, fps, fast_settings, use_hevc)
        encoder.start()

        # GPU 사용 여부
        use_gpu = self.use_gpu

        def decoder_thread():
            """고속 프레임 디코딩 (배치)"""
            nonlocal nvdec_decoder
            count = 0

            if use_nvdec:
                nvdec_decoder.start()
                while count < process_frames:
                    frame = nvdec_decoder.read_frame()
                    if frame is None:
                        break
                    decode_queue.put((count, frame))
                    count += 1
                    with stats_lock:
                        stats['decoded'] = count
                nvdec_decoder.close()
            else:
                while count < process_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    decode_queue.put((count, frame))
                    count += 1
                    with stats_lock:
                        stats['decoded'] = count

            done_decode[0] = True

        def filter_worker(worker_id):
            """병렬 필터 워커 (GPU LUT 캐시 활용)"""
            local_processed = 0
            
            # 워커별 GPU 스트림 (있으면)
            try:
                local_stream = cv2.cuda.Stream()
            except:
                local_stream = None
            
            while True:
                try:
                    item = decode_queue.get(timeout=0.1)
                except Empty:
                    if done_decode[0] and decode_queue.empty():
                        break
                    continue

                frame_idx, frame = item
                actual_idx = start_frame + frame_idx

                # 색보정 적용
                if actual_idx < len(params_list):
                    params = CorrectionParams(**params_list[actual_idx])
                    if use_gpu:
                        frame = apply_correction_gpu(frame, params)
                    else:
                        frame = apply_correction(frame, params)

                filter_queue.put((frame_idx, frame))
                local_processed += 1
                
                with stats_lock:
                    stats['processed'] += 1

            with filter_lock:
                done_filter[0] += 1

        def encoder_thread():
            """고속 NVENC 인코딩 (순서 보장 + 배치)"""
            pending = {}
            next_idx = 0
            batch_size = 8  # 배치 처리

            while True:
                # 배치로 가져오기
                batch_count = 0
                while batch_count < batch_size:
                    try:
                        item = filter_queue.get(timeout=0.05)
                        frame_idx, frame = item
                        pending[frame_idx] = frame
                        batch_count += 1
                    except Empty:
                        if done_filter[0] >= num_filter_workers and filter_queue.empty():
                            break
                        if batch_count > 0:
                            break
                        continue

                # 순서대로 인코딩
                while next_idx in pending:
                    encoder.write_frame(pending.pop(next_idx))
                    next_idx += 1
                    with stats_lock:
                        stats['encoded'] = next_idx

                # 종료 조건
                if done_filter[0] >= num_filter_workers and filter_queue.empty() and not pending:
                    break

            # 남은 프레임 처리
            while pending:
                if next_idx in pending:
                    encoder.write_frame(pending.pop(next_idx))
                    next_idx += 1
                else:
                    # 누락된 프레임 대기
                    time.sleep(0.01)

        # 스레드 시작
        decoder = Thread(target=decoder_thread, name='Decoder')
        encoder_t = Thread(target=encoder_thread, name='Encoder')
        
        # 필터 워커 풀
        filter_workers = [
            Thread(target=filter_worker, args=(i,), name=f'Filter-{i}')
            for i in range(num_filter_workers)
        ]

        decoder.start()
        for w in filter_workers:
            w.start()
        encoder_t.start()

        # 진행 상황 모니터링 (상세)
        last_log_time = time.time()
        while decoder.is_alive() or any(w.is_alive() for w in filter_workers) or encoder_t.is_alive():
            time.sleep(0.5)
            
            current_time = time.time()
            if current_time - last_log_time >= 5:  # 5초마다 로그
                with stats_lock:
                    processed = stats['processed']
                    decoded = stats['decoded']
                    encoded = stats['encoded']
                
                elapsed = current_time - start_total_time
                avg_fps = processed / elapsed if elapsed > 0 else 0
                progress = processed / process_frames * 100
                
                # 큐 상태
                dq_size = decode_queue.qsize()
                fq_size = filter_queue.qsize()
                
                eta_sec = (process_frames - processed) / avg_fps if avg_fps > 0 else 0
                eta_min = eta_sec / 60
                
                logger.info(
                    f"[{progress:5.1f}%] {processed:,}/{process_frames:,} | "
                    f"{avg_fps:.1f} fps | ETA: {eta_min:.1f}분 | "
                    f"Q: D={dq_size} F={fq_size}"
                )
                last_log_time = current_time

        # 스레드 종료 대기
        decoder.join()
        for w in filter_workers:
            w.join()
        encoder_t.join()

        # 인코더 종료
        encoder.close()

        # 리소스 정리
        if not use_nvdec:
            cap.release()

        total_time = time.time() - start_total_time
        avg_fps = stats['processed'] / total_time if total_time > 0 else 0

        logger.info("=" * 60)
        logger.info(f"✅ 완료!")
        logger.info(f"   프레임: {stats['processed']:,}")
        logger.info(f"   처리 속도: {avg_fps:.1f} fps")
        logger.info(f"   소요 시간: {total_time/60:.1f}분")
        logger.info(f"   출력: {output_path}")
        logger.info("=" * 60)

        return output_path

    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        params_json: str = None,
        start_time: float = None,
        end_time: float = None,
        use_hevc: bool = False,
        keep_params: bool = False,
        log_file: str = None,
        verbose: bool = False,
    ) -> str:
        """
        비디오 필터 처리 (분석 + 적용)
        
        Returns:
            str: 생성된 비디오 파일 경로
        """
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(Path(input_path).parent / f"filter_{timestamp}.log")

        logger = setup_logger(log_file, verbose)
        logger.info("=" * 60)
        logger.info("비디오 필터 처리 시작")
        logger.info("=" * 60)

        start_total = time.time()

        # 파라미터가 없으면 분석
        if params_json is None or not Path(params_json).exists():
            logger.info("\n>>> 분석 단계")
            params_json = self.analyze_video(
                input_path,
                start_time=start_time,
                end_time=end_time,
                log_file=log_file,
                verbose=verbose
            )

        # 필터 적용
        logger.info("\n>>> 적용 단계")
        result_path = self.apply_filter(
            input_path,
            output_path=output_path,
            params_json=params_json,
            start_time=start_time,
            end_time=end_time,
            use_hevc=use_hevc,
            log_file=log_file,
            verbose=verbose
        )

        # 파라미터 파일 정리
        if not keep_params and params_json:
            try:
                os.unlink(params_json)
                logger.info(f"임시 파라미터 삭제: {params_json}")
            except Exception:
                pass

        total_time = time.time() - start_total
        logger.info("=" * 60)
        logger.info(f"필터 처리 완료! 총 시간: {total_time/60:.1f}분")
        logger.info(f"출력: {result_path}")
        logger.info("=" * 60)

        return result_path


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
        description="비디오 필터 처리 (색보정)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 색보정 적용
  python filter_video.py video.mp4

  # 분석만 수행 (파라미터 JSON 저장)
  python filter_video.py video.mp4 --analyze-only

  # 기존 파라미터로 색보정 적용
  python filter_video.py video.mp4 --params video_params.json

  # HEVC 인코딩
  python filter_video.py video.mp4 --hevc

  # 고성능 모드 (GPU 가속)
  python filter_video.py video.mp4 --high-performance

  # 특정 구간 처리
  python filter_video.py video.mp4 --start 23:00 --end 28:00
        """
    )

    # 필수 인자
    parser.add_argument("input", help="입력 비디오 파일")
    parser.add_argument("-o", "--output", help="출력 파일 경로")

    # 모드 선택
    parser.add_argument("--analyze-only", action="store_true",
                       help="분석만 수행 (파라미터 JSON 저장)")
    parser.add_argument("--apply-only", action="store_true",
                       help="적용만 수행 (--params 필수)")
    parser.add_argument("--params", type=str,
                       help="파라미터 JSON 파일 경로")
    parser.add_argument("--keep-params", action="store_true",
                       help="처리 완료 후 파라미터 파일 유지")

    # 시간 범위
    parser.add_argument("--start", type=str, help="시작 시간 (예: 23:00)")
    parser.add_argument("--end", type=str, help="종료 시간 (예: 28:00)")

    # 색보정 옵션
    parser.add_argument("--no-color-grade", action="store_true",
                       help="색보정 비활성화")
    parser.add_argument("--cg-interval", type=int, default=1000,
                       help="색보정 분석 간격 (프레임, 기본값: 1000)")
    parser.add_argument("--cg-smooth", type=int, default=300,
                       help="색보정 스무딩 윈도우 (프레임, 기본값: 300)")

    # 시스템 옵션
    parser.add_argument("--high-performance", action="store_true",
                       help="고성능 모드 (멀티스레딩 + GPU)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="GPU 비활성화 (CPU 사용)")
    parser.add_argument("--no-nvdec", action="store_true",
                       help="NVDEC 비활성화 (CPU 디코딩)")
    parser.add_argument("--no-auto", action="store_true",
                       help="자동 최적화 비활성화")

    # 출력 옵션
    parser.add_argument("--hevc", action="store_true", help="HEVC 인코딩")

    # 로깅
    parser.add_argument("--log", type=str, help="로그 파일 경로")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그")

    args = parser.parse_args()

    # 검증
    if args.apply_only and not args.params:
        parser.error("--apply-only 사용 시 --params 필수")

    # 필터 생성
    video_filter = VideoFilter(
        color_grade=not args.no_color_grade,
        cg_interval=args.cg_interval,
        cg_smooth=args.cg_smooth,
        use_gpu=not args.no_gpu,
        use_nvdec=not args.no_nvdec,
        high_performance=args.high_performance,
        auto_optimize=not args.no_auto,
    )

    # 실행
    if args.analyze_only:
        print("📊 분석 모드 (파라미터 JSON 저장)")
        video_filter.analyze_video(
            input_path=args.input,
            output_json=args.output,
            start_time=parse_time(args.start),
            end_time=parse_time(args.end),
            log_file=args.log,
            verbose=args.verbose,
        )
    elif args.apply_only:
        print("🎨 적용 모드 (파라미터 JSON 사용)")
        video_filter.apply_filter(
            input_path=args.input,
            output_path=args.output,
            params_json=args.params,
            start_time=parse_time(args.start),
            end_time=parse_time(args.end),
            use_hevc=args.hevc,
            log_file=args.log,
            verbose=args.verbose,
        )
    else:
        print("🎬 필터 처리 (분석 + 적용)")
        video_filter.process_video(
            input_path=args.input,
            output_path=args.output,
            params_json=args.params,
            start_time=parse_time(args.start),
            end_time=parse_time(args.end),
            use_hevc=args.hevc,
            keep_params=args.keep_params,
            log_file=args.log,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
