#!/usr/bin/env python3
"""
인코딩 유틸리티 모듈
- NVENC 최적화 설정
- FFmpeg 명령어 빌더
- 시스템 정보 기반 자동 최적화
"""

import os
import platform
import subprocess


def get_system_info():
    """시스템 정보 수집 (CPU, RAM, GPU)"""
    info = {'cpu': {}, 'ram': {}, 'gpu': None}

    # CPU 정보
    try:
        import psutil
        info['cpu']['cores'] = psutil.cpu_count(logical=False)
        info['cpu']['threads'] = psutil.cpu_count(logical=True)

        if platform.system() == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        info['cpu']['model'] = line.split(':')[1].strip()
                        break
            try:
                result = subprocess.run(['lscpu'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'CPU max MHz' in line:
                        info['cpu']['max_mhz'] = float(line.split(':')[1].strip().replace(',', '.'))
            except Exception:
                pass
    except ImportError:
        info['cpu']['cores'] = os.cpu_count() or 4
        info['cpu']['threads'] = os.cpu_count() or 8

    # RAM 정보
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['ram']['total_gb'] = mem.total / (1024**3)
        info['ram']['available_gb'] = mem.available / (1024**3)
    except ImportError:
        info['ram']['total_gb'] = 8
        info['ram']['available_gb'] = 4

    # GPU 정보 (NVIDIA)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                info['gpu'] = {
                    'name': parts[0].strip(),
                    'vram_gb': float(parts[1].strip()) / 1024,
                    'compute_capability': parts[2].strip(),
                    'type': 'cuda'
                }
    except Exception:
        pass

    return info


def get_optimal_settings(system_info, frame_width=3840, frame_height=2160):
    """시스템 사양에 맞는 최적 설정 자동 계산"""
    settings = {
        'device': 'cpu',
        'batch_size': 2,
        'detect_scale': 0.5,
        'detect_interval': 3,
        'num_workers': 2,
        'queue_size': 64,
        'use_fp16': False,
        'ffmpeg_threads': 0,
        'nvenc_preset': 'p4',
        'nvenc_lookahead': 32,
        'nvenc_surfaces': 8,
        'nvenc_bframes': 3,
        'use_spatial_aq': True,
        'use_temporal_aq': True,
    }

    cpu = system_info.get('cpu', {})
    ram = system_info.get('ram', {})
    gpu = system_info.get('gpu')

    threads = cpu.get('threads', 8)
    settings['num_workers'] = max(2, min(threads // 2, 12))
    settings['ffmpeg_threads'] = threads

    ram_gb = ram.get('available_gb', 8)
    if ram_gb >= 24:
        settings['queue_size'] = 512
        settings['nvenc_surfaces'] = 16
    elif ram_gb >= 16:
        settings['queue_size'] = 384
        settings['nvenc_surfaces'] = 12
    elif ram_gb >= 8:
        settings['queue_size'] = 192
        settings['nvenc_surfaces'] = 8
    else:
        settings['queue_size'] = 64
        settings['nvenc_surfaces'] = 4

    if gpu:
        settings['device'] = gpu.get('type', 'cpu')
        vram_gb = gpu.get('vram_gb', 4)

        scaled_pixels = (frame_width * 0.5) * (frame_height * 0.5)
        frame_memory_gb = (scaled_pixels * 3 * 4) / (1024**3)
        available_vram = vram_gb * 0.7 - 0.5
        settings['batch_size'] = max(2, min(int(available_vram / (frame_memory_gb * 2.5)), 32))

        compute_cap = gpu.get('compute_capability', '0.0')
        major_version = int(compute_cap.split('.')[0])
        if major_version >= 7 and gpu.get('type') == 'cuda':
            settings['use_fp16'] = True

        if vram_gb >= 11.5:  # 12GB VRAM (11.99GB로 인식되는 경우 포함)
            settings['detect_scale'] = 0.8  # 고해상도 감지
            settings['detect_interval'] = 1
            settings['batch_size'] = 48  # 큰 배치
            settings['queue_size'] = 512  # 안정적인 버퍼
            settings['nvenc_preset'] = 'p5'  # 고품질 인코딩
            settings['nvenc_lookahead'] = 32  # 안정적인 lookahead
            settings['nvenc_bframes'] = 3
        elif vram_gb >= 8:
            settings['detect_scale'] = 0.5
            settings['detect_interval'] = 2
            settings['nvenc_preset'] = 'p3'
            settings['nvenc_lookahead'] = 24
            settings['nvenc_bframes'] = 3
        elif vram_gb >= 6:
            settings['detect_scale'] = 0.5
            settings['detect_interval'] = 3
            settings['nvenc_preset'] = 'p2'
            settings['nvenc_lookahead'] = 16
            settings['nvenc_bframes'] = 2
        else:
            settings['detect_scale'] = 0.4
            settings['detect_interval'] = 4
            settings['nvenc_preset'] = 'p1'
            settings['nvenc_lookahead'] = 8
            settings['nvenc_bframes'] = 0

    return settings


def build_nvenc_command(output_path, width, height, fps, settings, use_hevc=False):
    """
    최적화된 NVENC 인코딩 명령어 생성

    최적화 요소:
    - rc-lookahead: 프레임 미리보기로 품질 향상
    - spatial-aq: 공간 적응형 양자화 (디테일 보존)
    - temporal-aq: 시간 적응형 양자화 (움직임 품질)
    - surfaces: 동시 인코딩 표면 (처리량 증가)
    - b_ref_mode: B-프레임 참조 (압축 효율)
    """
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

    return cmd


def print_system_info(system_info, settings=None):
    """시스템 정보 및 최적 설정 출력"""
    cpu = system_info.get('cpu', {})
    ram = system_info.get('ram', {})
    gpu = system_info.get('gpu')

    print("\n" + "=" * 60)
    print("  시스템 하드웨어 정보")
    print("=" * 60)

    print(f"\n CPU: {cpu.get('model', 'Unknown')}")
    print(f"   코어: {cpu.get('cores', '?')}개 | 스레드: {cpu.get('threads', '?')}개")
    if cpu.get('max_mhz'):
        print(f"   최대 클럭: {cpu.get('max_mhz')/1000:.2f} GHz")

    print(f"\n RAM: {ram.get('total_gb', 0):.1f} GB (사용 가능: {ram.get('available_gb', 0):.1f} GB)")

    if gpu:
        print(f"\n GPU: {gpu.get('name', 'Unknown')}")
        print(f"   VRAM: {gpu.get('vram_gb', 0):.1f} GB")
        if gpu.get('compute_capability'):
            print(f"   Compute Capability: {gpu.get('compute_capability')}")
    else:
        print("\n GPU: 감지되지 않음 (CPU 모드)")

    if settings:
        print("\n" + "-" * 60)
        print(" 자동 최적화 설정")
        print("-" * 60)
        print(f"   [추론] 디바이스: {settings.get('device', 'cpu').upper()}")
        print(f"   [추론] 배치 크기: {settings.get('batch_size', 4)}")
        print(f"   [추론] 감지 스케일: {settings.get('detect_scale', 0.5)}")
        print(f"   [추론] 감지 간격: {settings.get('detect_interval', 3)}프레임마다")
        print(f"   [추론] FP16: {'활성화' if settings.get('use_fp16') else '비활성화'}")
        print(f"   [인코딩] NVENC 프리셋: {settings.get('nvenc_preset', 'p4')}")
        print(f"   [인코딩] Lookahead: {settings.get('nvenc_lookahead', 32)}프레임")
        print(f"   [인코딩] B-프레임: {settings.get('nvenc_bframes', 3)}")
        print(f"   [인코딩] Surfaces: {settings.get('nvenc_surfaces', 8)}")
        print(f"   [인코딩] Spatial AQ: {'활성화' if settings.get('use_spatial_aq') else '비활성화'}")
        print(f"   [인코딩] Temporal AQ: {'활성화' if settings.get('use_temporal_aq') else '비활성화'}")
        print(f"   [시스템] FFmpeg 스레드: {settings.get('ffmpeg_threads', 0)}")
        print(f"   [시스템] 큐 크기: {settings.get('queue_size', 64)}")

    print("=" * 60 + "\n")
