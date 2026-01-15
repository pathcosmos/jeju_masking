#!/usr/bin/env python3
"""
배치 마스킹 작업 큐 시스템
- 최대 N개 동시 처리
- 하나 끝나면 다음 파일 자동 시작
- 파일별 상세 로그 기록
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 로그 파일 잠금
log_lock = Lock()

def get_video_files(input_dir: str) -> list:
    """디렉토리에서 비디오 파일 목록 가져오기"""
    input_path = Path(input_dir)
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    files = []
    for ext in video_extensions:
        files.extend(input_path.glob(f'*{ext}'))
    return sorted(files)


def format_duration(seconds: float) -> str:
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}시간 {minutes}분 {secs}초"
    elif minutes > 0:
        return f"{minutes}분 {secs}초"
    else:
        return f"{secs}초"


def write_log(log_file: str, message: str, also_print: bool = True):
    """로그 파일에 메시지 기록"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"

    with log_lock:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        if also_print:
            print(log_message)


def process_video(input_file: Path, output_dir: Path, log_dir: Path,
                  master_log: str, extra_args: list = None) -> dict:
    """단일 비디오 처리 (마스킹만)"""
    start_time = time.time()

    # 출력 파일명
    output_file = output_dir / f"{input_file.stem}_masked.mp4"

    # 개별 로그 파일
    individual_log = log_dir / f"{input_file.stem}.log"

    write_log(master_log, f"▶ 시작: {input_file.name}")
    write_log(str(individual_log), f"=== 마스킹 작업 시작 ===", also_print=False)
    write_log(str(individual_log), f"입력: {input_file}", also_print=False)
    write_log(str(individual_log), f"출력: {output_file}", also_print=False)

    # 마스킹 명령 구성 (.venv 환경 사용)
    script_dir = Path(__file__).parent
    venv_python = script_dir / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [
        python_exe, "mask_video.py",
        str(input_file),
        "-o", str(output_file),
        "--high-performance",
        "--batch-size", "8",
        "--queue-size", "256",
        "--no-tensorrt",  # PyTorch 모델 사용 (TensorRT 배치 호환 문제 회피)
    ]

    if extra_args:
        cmd.extend(extra_args)

    write_log(str(individual_log), f"명령: {' '.join(cmd)}", also_print=False)
    write_log(str(individual_log), "", also_print=False)

    # 프로세스 실행
    result = {
        'input': input_file.name,
        'output': output_file.name,
        'success': False,
        'duration': 0,
        'error': None
    }

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path(__file__).parent)
        )

        # 실시간 로그 기록
        for line in process.stdout:
            line = line.rstrip()
            if line:
                write_log(str(individual_log), line, also_print=False)

        process.wait()

        duration = time.time() - start_time
        result['duration'] = duration

        if process.returncode == 0:
            result['success'] = True
            write_log(str(individual_log), "", also_print=False)
            write_log(str(individual_log), f"=== 작업 완료 ===", also_print=False)
            write_log(str(individual_log), f"소요 시간: {format_duration(duration)}", also_print=False)
            write_log(master_log, f"✓ 완료: {input_file.name} ({format_duration(duration)})")
        else:
            result['success'] = False
            result['error'] = f"종료 코드: {process.returncode}"
            write_log(str(individual_log), f"=== 작업 실패 (코드: {process.returncode}) ===", also_print=False)
            write_log(master_log, f"✗ 실패: {input_file.name} (코드: {process.returncode})")

    except Exception as e:
        duration = time.time() - start_time
        result['duration'] = duration
        result['error'] = str(e)
        write_log(str(individual_log), f"=== 오류 발생 ===", also_print=False)
        write_log(str(individual_log), f"오류: {e}", also_print=False)
        write_log(master_log, f"✗ 오류: {input_file.name} - {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description='배치 마스킹 작업 큐')
    parser.add_argument('input_dir', help='입력 디렉토리')
    parser.add_argument('-o', '--output-dir', help='출력 디렉토리 (기본: input_dir/masking)')
    parser.add_argument('-w', '--workers', type=int, default=2,
                       help='동시 처리 개수 (기본: 2)')
    parser.add_argument('--log-dir', help='로그 디렉토리')
    parser.add_argument('--dry-run', action='store_true',
                       help='실제 실행 없이 작업 목록만 출력')

    args = parser.parse_args()

    # 경로 설정
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "masking"
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"

    # 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 마스터 로그 파일
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = str(log_dir / f"batch_{timestamp}.log")

    # 비디오 파일 목록
    video_files = get_video_files(input_dir)

    if not video_files:
        print(f"오류: {input_dir}에 비디오 파일이 없습니다.")
        return 1

    # 작업 정보 출력
    print("=" * 60)
    print("배치 마스킹 작업 큐")
    print("=" * 60)
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"로그 디렉토리: {log_dir}")
    print(f"동시 처리: {args.workers}개")
    print(f"마스터 로그: {master_log}")
    print("-" * 60)
    print(f"작업 대상: {len(video_files)}개 파일")
    for i, f in enumerate(video_files, 1):
        print(f"  {i}. {f.name}")
    print("=" * 60)

    if args.dry_run:
        print("(--dry-run 모드: 실제 실행하지 않음)")
        return 0

    # 마스터 로그 시작
    write_log(master_log, "=" * 50)
    write_log(master_log, "배치 마스킹 작업 시작")
    write_log(master_log, f"총 파일: {len(video_files)}개, 동시 처리: {args.workers}개")
    write_log(master_log, "=" * 50)

    # 작업 실행
    total_start = time.time()
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 모든 작업 제출
        futures = {
            executor.submit(
                process_video,
                video_file,
                output_dir,
                log_dir,
                master_log
            ): video_file for video_file in video_files
        }

        # 완료되는 대로 처리
        for future in as_completed(futures):
            video_file = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                # 진행 상황 출력
                remaining = len(video_files) - completed
                write_log(master_log, f"   진행: {completed}/{len(video_files)} 완료, {remaining}개 남음")

            except Exception as e:
                write_log(master_log, f"✗ 예외: {video_file.name} - {e}")
                results.append({
                    'input': video_file.name,
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                })

    # 최종 결과 요약
    total_duration = time.time() - total_start
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count

    write_log(master_log, "")
    write_log(master_log, "=" * 50)
    write_log(master_log, "작업 완료 요약")
    write_log(master_log, "=" * 50)
    write_log(master_log, f"총 소요 시간: {format_duration(total_duration)}")
    write_log(master_log, f"성공: {success_count}개, 실패: {fail_count}개")
    write_log(master_log, "-" * 50)

    for r in results:
        status = "✓" if r['success'] else "✗"
        duration = format_duration(r['duration']) if r['duration'] > 0 else "-"
        write_log(master_log, f"{status} {r['input']}: {duration}")
        if r.get('error'):
            write_log(master_log, f"   오류: {r['error']}")

    write_log(master_log, "=" * 50)

    print(f"\n상세 로그: {master_log}")
    print(f"개별 로그: {log_dir}/")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
