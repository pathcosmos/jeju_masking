#!/usr/bin/env python3
"""
적응형 색보정 스크립트 (Adaptive Color Grading)
- C 스타일 기준으로 장면별 적응형 색보정 적용
- 부드러운 전환을 위한 보간 + 스무딩
- CuPy GPU 가속 지원
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

# CuPy GPU 가속 (선택적)
_cupy_available = False
_cp = None
_mempool = None

def _init_cupy():
    """CuPy 초기화 (지연 로딩)"""
    global _cupy_available, _cp, _mempool
    if _cp is not None:
        return _cupy_available
    try:
        import cupy as cp
        _cp = cp
        # 메모리 풀 설정 (효율적인 메모리 재사용)
        _mempool = cp.get_default_memory_pool()
        _cupy_available = True
        print(f"[ColorGrade] CuPy GPU 가속 활성화 (버전: {cp.__version__})")
    except ImportError:
        _cupy_available = False
        print("[ColorGrade] CuPy 미설치 - CPU 모드 사용")
    except Exception as e:
        _cupy_available = False
        print(f"[ColorGrade] CuPy 초기화 실패: {e} - CPU 모드 사용")
    return _cupy_available

# C 스타일 목표값 (참조 기준)
@dataclass
class TargetStyle:
    """C 스타일 목표 파라미터"""
    brightness: float = 115.0      # 목표 밝기 (V채널, 0-255)
    saturation: float = 55.0       # 목표 채도 (S채널, 0-255)
    contrast: float = 1.15         # 기본 대비 배율
    color_temp_a: float = 2.0      # 목표 색온도 a (약간 따뜻하게)
    color_temp_b: float = 3.0      # 목표 색온도 b (약간 노란빛)


@dataclass
class FrameAnalysis:
    """프레임 분석 결과"""
    frame_num: int
    brightness: float
    saturation: float
    luminance: float
    color_temp_a: float
    color_temp_b: float


@dataclass
class CorrectionParams:
    """보정 파라미터"""
    brightness_adj: float    # 밝기 조정값 (-1.0 ~ 1.0)
    saturation_mult: float   # 채도 배율 (0.5 ~ 2.0)
    contrast_mult: float     # 대비 배율 (0.5 ~ 2.0)
    warm_shift: float        # 따뜻한 톤 시프트 (0.0 ~ 1.0)


def analyze_frame(frame: np.ndarray) -> dict:
    """프레임의 색상 특성 분석"""
    # BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    return {
        'brightness': float(np.mean(v)),
        'saturation': float(np.mean(s)),
        'luminance': float(np.mean(l)),
        'color_temp_a': float(np.mean(a)) - 128,
        'color_temp_b': float(np.mean(b)) - 128,
    }


def calculate_correction(analysis: dict, target: TargetStyle) -> CorrectionParams:
    """분석 결과를 기반으로 보정 파라미터 계산"""

    # 밝기 보정 (-0.15 ~ 0.15 범위로 제한)
    brightness_diff = target.brightness - analysis['brightness']
    brightness_adj = np.clip(brightness_diff / 255.0 * 0.5, -0.15, 0.15)

    # 채도 보정 (0.8 ~ 1.6 범위로 제한)
    if analysis['saturation'] > 10:  # 0으로 나누기 방지
        saturation_ratio = target.saturation / analysis['saturation']
        saturation_mult = np.clip(saturation_ratio, 0.8, 1.6)
    else:
        saturation_mult = 1.3

    # 대비 - 밝기에 따라 조정 (어두우면 대비 높게)
    if analysis['brightness'] < 80:
        contrast_mult = 1.2
    elif analysis['brightness'] > 140:
        contrast_mult = 1.1
    else:
        contrast_mult = 1.15

    # 따뜻한 톤 시프트 (현재 색온도가 차가우면 더 따뜻하게)
    warm_shift = np.clip((target.color_temp_b - analysis['color_temp_b']) / 20.0, 0.0, 0.3)

    return CorrectionParams(
        brightness_adj=brightness_adj,
        saturation_mult=saturation_mult,
        contrast_mult=contrast_mult,
        warm_shift=warm_shift
    )


def smooth_params(params_list: List[dict], window: int = 300) -> List[dict]:
    """파라미터 시퀀스를 스무딩"""
    if len(params_list) < 2:
        return params_list

    # 각 파라미터를 배열로 변환
    keys = ['brightness_adj', 'saturation_mult', 'contrast_mult', 'warm_shift']
    arrays = {k: np.array([p[k] for p in params_list]) for k in keys}

    # 이동평균 스무딩
    kernel = np.ones(window) / window
    smoothed = {}
    for k, arr in arrays.items():
        padded = np.pad(arr, window//2, mode='edge')
        smoothed[k] = np.convolve(padded, kernel, mode='valid')[:len(arr)]

    # 다시 리스트로 변환
    result = []
    for i in range(len(params_list)):
        result.append({k: float(smoothed[k][i]) for k in keys})

    return result


def apply_correction(frame: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """프레임에 보정 적용 (CPU 버전)"""
    # float로 변환
    img = frame.astype(np.float32) / 255.0

    # 1. 밝기 조정
    img = img + params.brightness_adj

    # 2. 대비 조정 (중간값 기준)
    img = (img - 0.5) * params.contrast_mult + 0.5

    # 3. 채도 조정 (HSV 공간에서)
    img_clipped = np.clip(img * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_clipped, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params.saturation_mult, 0, 255)
    img_clipped = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img_clipped.astype(np.float32) / 255.0

    # 4. 따뜻한 톤 시프트 (R 채널 약간 증가, B 채널 약간 감소)
    if params.warm_shift > 0:
        img[:, :, 2] = img[:, :, 2] * (1 + params.warm_shift * 0.1)  # R
        img[:, :, 0] = img[:, :, 0] * (1 - params.warm_shift * 0.08)  # B

    # 클리핑 및 uint8 변환
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


# GPU 가속 색보정 (OpenCV CUDA)
_gpu_available = None
_gpu_stream = None

def _check_gpu():
    """GPU 사용 가능 여부 확인"""
    global _gpu_available, _gpu_stream
    if _gpu_available is None:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                _gpu_available = True
                _gpu_stream = cv2.cuda_Stream()
            else:
                _gpu_available = False
        except:
            _gpu_available = False
    return _gpu_available


def apply_correction_fast(frame: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """
    프레임에 보정 적용 (최적화된 CPU 버전)
    NumPy 벡터화 + LUT 기반으로 빠른 처리
    """
    # 1. 밝기/대비 조정 (LUT 사용 - 가장 빠름)
    alpha = params.contrast_mult
    beta = int(params.brightness_adj * 255 + 128 * (1 - params.contrast_mult))
    
    # LUT 생성 (한번만 계산)
    lut = np.clip(np.arange(256) * alpha + beta, 0, 255).astype(np.uint8)
    img = cv2.LUT(frame, lut)

    # 2. 채도 조정 (HSV 공간)
    if abs(params.saturation_mult - 1.0) > 0.01:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # S 채널만 조정 (LUT 사용)
        sat_lut = np.clip(np.arange(256) * params.saturation_mult, 0, 255).astype(np.uint8)
        hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], sat_lut)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. 따뜻한 톤 시프트 (LUT 사용)
    if params.warm_shift > 0.01:
        # R 채널 증가, B 채널 감소
        r_lut = np.clip(np.arange(256) * (1 + params.warm_shift * 0.1), 0, 255).astype(np.uint8)
        b_lut = np.clip(np.arange(256) * (1 - params.warm_shift * 0.08), 0, 255).astype(np.uint8)
        
        img[:, :, 2] = cv2.LUT(img[:, :, 2], r_lut)  # R
        img[:, :, 0] = cv2.LUT(img[:, :, 0], b_lut)  # B

    return img


# GPU LUT 캐시 (파라미터별)
_gpu_lut_cache = {}

def _get_or_create_gpu_lut(params: CorrectionParams):
    """GPU LUT 캐시 조회 또는 생성"""
    cache_key = (
        round(params.brightness_adj, 3),
        round(params.contrast_mult, 3),
        round(params.warm_shift, 3)
    )
    
    if cache_key not in _gpu_lut_cache:
        alpha = params.contrast_mult
        beta = params.brightness_adj * 255 + 128 * (1 - params.contrast_mult)
        warm = params.warm_shift
        
        # BGR 각 채널별 통합 LUT (밝기/대비/색온도)
        b_lut = np.clip(np.arange(256) * alpha * (1 - warm * 0.08) + beta, 0, 255).astype(np.uint8)
        g_lut = np.clip(np.arange(256) * alpha + beta, 0, 255).astype(np.uint8)
        r_lut = np.clip(np.arange(256) * alpha * (1 + warm * 0.1) + beta, 0, 255).astype(np.uint8)
        
        combined_lut = np.stack([b_lut, g_lut, r_lut], axis=1).reshape(256, 1, 3)
        gpu_lut = cv2.cuda.createLookUpTable(combined_lut)
        _gpu_lut_cache[cache_key] = gpu_lut
        
        # 캐시 크기 제한 (최근 100개만 유지)
        if len(_gpu_lut_cache) > 100:
            oldest_key = next(iter(_gpu_lut_cache))
            del _gpu_lut_cache[oldest_key]
    
    return _gpu_lut_cache[cache_key]


def apply_correction_gpu(frame: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """
    프레임에 보정 적용 (OpenCV CUDA GPU 가속 - 최적화)
    - GPU: 밝기/대비/색온도 (통합 LUT)
    - CPU: 채도 (HSV 변환 필요)
    4K 기준 초당 40+ 프레임 성능
    """
    if not _check_gpu():
        return apply_correction_fast(frame, params)
    
    try:
        stream = _gpu_stream
        
        # GPU LUT 가져오기 (캐시됨)
        gpu_lut = _get_or_create_gpu_lut(params)
        
        # GPU 업로드
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame, stream)
        
        # GPU: 밝기/대비/색온도 적용 (통합 LUT)
        gpu_adjusted = gpu_lut.transform(gpu_frame, stream=stream)
        stream.waitForCompletion()
        
        # 다운로드
        img = gpu_adjusted.download()
        
        # CPU: 채도 조정 (HSV 변환 + LUT)
        if abs(params.saturation_mult - 1.0) > 0.01:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            sat_lut = np.clip(np.arange(256) * params.saturation_mult, 0, 255).astype(np.uint8)
            hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], sat_lut)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return img
            
    except Exception as e:
        print(f"[ColorGrade] GPU 오류, CPU fallback: {e}")
        return apply_correction_fast(frame, params)


def analyze_video(video_path: str, interval: int = 1000, verbose: bool = True) -> Tuple[List[int], List[dict]]:
    """비디오 분석하여 키프레임 정보 추출"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if verbose:
        print(f"비디오 분석 중: {video_path}")
        print(f"총 프레임: {total_frames}, FPS: {fps:.2f}")
        print(f"분석 간격: {interval} 프레임 ({interval/fps:.1f}초)")
        print("-" * 60)

    keyframes = []
    analyses = []
    target = TargetStyle()

    frame_nums = list(range(0, total_frames, interval))
    if frame_nums[-1] != total_frames - 1:
        frame_nums.append(total_frames - 1)

    for i, frame_num in enumerate(frame_nums):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        analysis = analyze_frame(frame)
        correction = calculate_correction(analysis, target)

        keyframes.append(frame_num)
        analyses.append({
            'frame': frame_num,
            'analysis': analysis,
            'correction': {
                'brightness_adj': correction.brightness_adj,
                'saturation_mult': correction.saturation_mult,
                'contrast_mult': correction.contrast_mult,
                'warm_shift': correction.warm_shift
            }
        })

        if verbose:
            progress = (i + 1) / len(frame_nums) * 100
            print(f"\r분석 진행: {progress:.1f}% ({i+1}/{len(frame_nums)}) - "
                  f"프레임 {frame_num}: 밝기={analysis['brightness']:.1f}, "
                  f"채도={analysis['saturation']:.1f}", end='', flush=True)

    if verbose:
        print("\n" + "-" * 60)
        print(f"분석 완료: {len(keyframes)}개 키프레임")

    cap.release()
    return keyframes, analyses


def interpolate_params(keyframes: List[int], analyses: List[dict],
                       total_frames: int, smooth_window: int = 300) -> List[dict]:
    """키프레임 파라미터를 전체 프레임으로 보간"""
    print(f"파라미터 보간 중 (스무딩 윈도우: {smooth_window} 프레임)...")

    # 키프레임 보정값 추출
    corrections = [a['correction'] for a in analyses]
    keys = ['brightness_adj', 'saturation_mult', 'contrast_mult', 'warm_shift']

    # 선형 보간
    all_frames = np.arange(total_frames)
    interpolated = {k: np.interp(all_frames, keyframes,
                                  [c[k] for c in corrections]) for k in keys}

    # 스무딩 적용
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = {}
    for k, arr in interpolated.items():
        padded = np.pad(arr, smooth_window//2, mode='edge')
        smoothed[k] = np.convolve(padded, kernel, mode='valid')[:total_frames]

    # 리스트로 변환
    params_list = []
    for i in range(total_frames):
        params_list.append({k: float(smoothed[k][i]) for k in keys})

    print(f"보간 완료: {total_frames}개 프레임 파라미터 생성")
    return params_list


class ColorGrader:
    """
    적응형 색보정 클래스 (video_masker.py에서 사용)

    사용법:
        grader = ColorGrader(video_path, analyze_interval=1000, smooth_window=300)
        for frame_idx, frame in enumerate(frames):
            graded_frame = grader.apply(frame, frame_idx)
    """

    def __init__(self, video_path: str, analyze_interval: int = 1000,
                 smooth_window: int = 300, params_file: str = None,
                 use_gpu: bool = True, verbose: bool = True):
        """
        Args:
            video_path: 비디오 파일 경로
            analyze_interval: 분석 간격 (프레임 수, 기본값: 1000)
            smooth_window: 스무딩 윈도우 크기 (프레임 수, 기본값: 300)
            params_file: 파라미터 JSON 파일 경로 (없으면 자동 생성)
            use_gpu: GPU 가속 사용 여부 (기본값: True)
            verbose: 상세 출력 여부
        """
        self.video_path = video_path
        self.analyze_interval = analyze_interval
        self.smooth_window = smooth_window
        self.verbose = verbose
        self.params_list = None
        self.total_frames = 0

        # GPU 사용 설정
        self.use_gpu = use_gpu and _check_gpu()
        if verbose and self.use_gpu:
            print(f"[ColorGrader] GPU 가속 활성화")

        # 파라미터 파일 경로 설정
        if params_file is None:
            input_path = Path(video_path)
            params_file = str(input_path.parent / f"{input_path.stem}_colorgrade_params.json")
        self.params_file = params_file

        # 초기화 (분석 또는 로드)
        self._initialize()

    def _initialize(self):
        """비디오 분석 또는 기존 파라미터 로드"""
        # 비디오 정보 가져오기
        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 기존 파라미터 파일 확인
        if Path(self.params_file).exists():
            if self.verbose:
                print(f"[ColorGrader] 기존 파라미터 로드: {self.params_file}")
            with open(self.params_file, 'r') as f:
                data = json.load(f)
                self.params_list = data['params']

            # 프레임 수 일치 확인
            if len(self.params_list) != self.total_frames:
                if self.verbose:
                    print(f"[ColorGrader] 프레임 수 불일치, 재분석 필요")
                self.params_list = None

        # 파라미터가 없으면 분석 수행
        if self.params_list is None:
            if self.verbose:
                print(f"[ColorGrader] 비디오 분석 중...")
            keyframes, analyses = analyze_video(
                self.video_path, self.analyze_interval, self.verbose
            )
            self.params_list = interpolate_params(
                keyframes, analyses, self.total_frames, self.smooth_window
            )

            # 파라미터 저장
            with open(self.params_file, 'w') as f:
                json.dump({
                    'input': self.video_path,
                    'total_frames': self.total_frames,
                    'interval': self.analyze_interval,
                    'smooth_window': self.smooth_window,
                    'keyframes': len(keyframes),
                    'params': self.params_list
                }, f)
            if self.verbose:
                print(f"[ColorGrader] 파라미터 저장: {self.params_file}")

    def apply(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        프레임에 적응형 색보정 적용

        Args:
            frame: 입력 프레임 (BGR)
            frame_idx: 프레임 인덱스

        Returns:
            보정된 프레임 (BGR)
        """
        if self.params_list is None:
            return frame

        # 범위 체크
        idx = min(frame_idx, len(self.params_list) - 1)
        idx = max(0, idx)

        # 파라미터 가져오기
        params = CorrectionParams(**self.params_list[idx])

        # 보정 적용 (GPU 또는 CPU)
        if self.use_gpu:
            return apply_correction_gpu(frame, params)
        else:
            return apply_correction(frame, params)

    def get_params(self, frame_idx: int) -> dict:
        """특정 프레임의 보정 파라미터 반환"""
        if self.params_list is None:
            return None
        idx = min(frame_idx, len(self.params_list) - 1)
        idx = max(0, idx)
        return self.params_list[idx]


def process_video(input_path: str, output_path: str, params_list: List[dict],
                  start_frame: int = 0, end_frame: int = None, verbose: bool = True):
    """비디오에 적응형 색보정 적용"""
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame is None:
        end_frame = total_frames

    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if verbose:
        print(f"\n색보정 적용 중: {input_path}")
        print(f"출력: {output_path}")
        print(f"처리 범위: 프레임 {start_frame} ~ {end_frame}")
        print("-" * 60)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # 해당 프레임의 보정 파라미터 가져오기
        params = CorrectionParams(**params_list[frame_num])

        # 보정 적용
        corrected = apply_correction(frame, params)

        out.write(corrected)

        if verbose and frame_num % 100 == 0:
            progress = (frame_num - start_frame + 1) / (end_frame - start_frame) * 100
            print(f"\r처리 진행: {progress:.1f}% (프레임 {frame_num}/{end_frame})",
                  end='', flush=True)

    if verbose:
        print(f"\n처리 완료!")

    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description='적응형 색보정 (Adaptive Color Grading)')
    parser.add_argument('input', help='입력 비디오 파일')
    parser.add_argument('-o', '--output', help='출력 비디오 파일')
    parser.add_argument('-i', '--interval', type=int, default=1000,
                        help='분석 간격 (프레임, 기본값: 1000)')
    parser.add_argument('-s', '--smooth', type=int, default=300,
                        help='스무딩 윈도우 크기 (프레임, 기본값: 300)')
    parser.add_argument('--start', type=int, default=0, help='시작 프레임')
    parser.add_argument('--end', type=int, default=None, help='끝 프레임')
    parser.add_argument('--analyze-only', action='store_true',
                        help='분석만 수행하고 파라미터 저장')
    parser.add_argument('--params-file', help='파라미터 JSON 파일 (로드/저장)')
    parser.add_argument('-q', '--quiet', action='store_true', help='출력 최소화')

    args = parser.parse_args()

    verbose = not args.quiet

    # 출력 파일명 설정
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_graded{input_path.suffix}")

    # 파라미터 파일 경로
    if args.params_file is None:
        input_path = Path(args.input)
        args.params_file = str(input_path.parent / f"{input_path.stem}_params.json")

    # 비디오 정보 가져오기
    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 기존 파라미터 파일이 있으면 로드
    params_list = None
    if Path(args.params_file).exists() and not args.analyze_only:
        if verbose:
            print(f"기존 파라미터 파일 로드: {args.params_file}")
        with open(args.params_file, 'r') as f:
            data = json.load(f)
            params_list = data['params']

    # 파라미터가 없으면 분석 수행
    if params_list is None:
        keyframes, analyses = analyze_video(args.input, args.interval, verbose)
        params_list = interpolate_params(keyframes, analyses, total_frames, args.smooth)

        # 파라미터 저장
        with open(args.params_file, 'w') as f:
            json.dump({
                'input': args.input,
                'total_frames': total_frames,
                'interval': args.interval,
                'smooth_window': args.smooth,
                'keyframes': len(keyframes),
                'analyses': analyses,
                'params': params_list
            }, f)
        if verbose:
            print(f"파라미터 저장: {args.params_file}")

    if args.analyze_only:
        print("분석 완료 (--analyze-only 모드)")
        return

    # 비디오 처리
    end_frame = args.end if args.end else total_frames
    process_video(args.input, args.output, params_list,
                  args.start, end_frame, verbose)

    print(f"\n완료! 출력 파일: {args.output}")


if __name__ == '__main__':
    main()
