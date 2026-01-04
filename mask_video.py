#!/usr/bin/env python3
"""
비디오 마스킹 CLI v3.1
- 사람 전체 마스킹 (YOLOv8 person 감지)
- 차량 번호판 마스킹 (YOLOv8 vehicle 감지)
- 기본 모드 / 최적화 모드 / 2-Pass 모드 선택 가능

사용 예시:
  # 기본 사용 (최적화 모드, 시스템 자동 최적화)
  python mask_video.py video.mp4

  # 2-Pass 모드 (GPU 최대 활용, 권장)
  python mask_video.py video.mp4 --2pass --hevc

  # 2-Pass 분리 실행 (여러 영상 일괄 처리)
  python mask_video.py video1.mp4 --analyze-only
  python mask_video.py video2.mp4 --analyze-only
  python mask_video.py video1.mp4 --encode-only --mask-json video1_masks.json --hevc
  python mask_video.py video2.mp4 --encode-only --mask-json video2_masks.json --hevc

  # 간단 모드 (빠른 설정, CPU에서도 작동)
  python mask_video.py video.mp4 --simple

  # 특정 구간 처리
  python mask_video.py video.mp4 --start 23:00 --end 28:00

  # HEVC 인코딩
  python mask_video.py video.mp4 --hevc
"""

import argparse
from masking_utils import parse_time
from video_masker import VideoMasker, VideoMaskerOptimized


def main():
    parser = argparse.ArgumentParser(
        description="사람/번호판 마스킹 v3.0 - 사람 전체 마스킹 + 번호판 마스킹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (최적화 모드, GPU 자동 감지)
  python mask_video.py video.mp4

  # 간단 모드 (설정 최소화)
  python mask_video.py video.mp4 --simple

  # 특정 구간 처리
  python mask_video.py video.mp4 --start 23:00 --end 28:00

  # 사람만 마스킹 (번호판 제외)
  python mask_video.py video.mp4 --no-plates

  # 번호판만 마스킹 (사람 제외)
  python mask_video.py video.mp4 --no-persons

  # 모자이크 처리
  python mask_video.py video.mp4 --mask-type mosaic

  # HEVC 인코딩 (RTX GPU에서 NVENC 자동 사용)
  python mask_video.py video.mp4 --hevc

  # CPU 강제 사용
  python mask_video.py video.mp4 --device cpu

  # FP16 활성화 (NVIDIA GPU)
  python mask_video.py video.mp4 --fp16
        """
    )

    # 필수 인자
    parser.add_argument("input", help="입력 비디오 파일")
    parser.add_argument("-o", "--output", help="출력 파일 경로")

    # 모드 선택
    parser.add_argument("--simple", action="store_true",
                       help="간단 모드 (빠른 설정, 최적화 없음)")

    # 2-Pass 모드 (GPU 최대 활용)
    parser.add_argument("--2pass", dest="two_pass", action="store_true",
                       help="2-Pass 모드: Pass1(분석) + Pass2(인코딩) 분리로 GPU 100%% 활용")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Pass 1만 실행: 마스크 좌표를 JSON으로 저장")
    parser.add_argument("--encode-only", action="store_true",
                       help="Pass 2만 실행: JSON 마스크 데이터로 인코딩 (--mask-json 필수)")
    parser.add_argument("--mask-json", type=str,
                       help="마스크 JSON 파일 경로 (--encode-only와 함께 사용)")
    parser.add_argument("--keep-json", action="store_true",
                       help="2-Pass 완료 후 JSON 파일 유지")

    # 시간 범위
    parser.add_argument("--start", type=str, help="시작 시간 (예: 23:00)")
    parser.add_argument("--end", type=str, help="종료 시간 (예: 28:00)")

    # 마스킹 옵션
    parser.add_argument("--no-persons", action="store_true", help="사람 마스킹 비활성화")
    parser.add_argument("--no-plates", action="store_true", help="번호판 마스킹 비활성화")
    parser.add_argument("--mask-type", choices=["blur", "mosaic"], default="blur",
                       help="마스킹 방식 (기본: blur)")
    parser.add_argument("--blur-strength", type=int, default=51,
                       help="블러 강도 (기본: 51)")
    parser.add_argument("--mosaic-size", type=int, default=15,
                       help="모자이크 블록 크기 (기본: 15)")

    # 감지 파라미터
    parser.add_argument("--person-conf", type=float, default=0.4,
                       help="사람 감지 신뢰도 (기본: 0.4)")
    parser.add_argument("--vehicle-conf", type=float, default=0.3,
                       help="차량 감지 신뢰도 (기본: 0.3)")
    parser.add_argument("--person-expand", type=float, default=0.1,
                       help="사람 영역 확장 비율 (기본: 0.1)")
    parser.add_argument("--plate-expand", type=float, default=0.3,
                       help="번호판 영역 확장 비율 (기본: 0.3)")
    parser.add_argument("--plate-detect", type=str, default="auto",
                       choices=["auto", "multi", "legacy"],
                       help="번호판 감지 모드: auto=OpenCV 자동감지, multi=다중위치, legacy=하단만 (기본: auto)")
    parser.add_argument("--plate-smoothing", type=float, default=0.6,
                       help="번호판 위치 스무딩 (0~1, 높을수록 안정적, 기본: 0.6)")
    parser.add_argument("--max-mask-ratio", type=float, default=0.4,
                       help="프레임 대비 최대 마스킹 영역 비율 (0~1, 기본: 0.4=40%%)")

    # 최적화 파라미터 (최적화 모드 전용)
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "mps", "cuda"],
                       help="처리 디바이스 (기본: auto)")
    parser.add_argument("--detect-interval", type=int, default=-1,
                       help="감지 간격 (N프레임마다, -1=자동)")
    parser.add_argument("--detect-scale", type=float, default=-1,
                       help="감지용 다운스케일 (-1=자동)")
    parser.add_argument("--batch-size", type=int, default=-1,
                       help="배치 크기 (-1=자동)")
    parser.add_argument("--high-performance", action="store_true",
                       help="고성능 모드: FFmpeg 파이프라인 + 진정한 배치 추론 (CUDA 필수)")
    parser.add_argument("--fp16", action="store_true",
                       help="FP16 반정밀도 추론 (NVIDIA GPU)")
    parser.add_argument("--tensorrt", action="store_true",
                       help="TensorRT 가속")
    parser.add_argument("--no-auto", action="store_true",
                       help="자동 최적화 비활성화")

    # 트래킹 파라미터
    parser.add_argument("--tracker", type=str, default="bytetrack",
                       choices=["bytetrack", "botsort"],
                       help="트래커 종류 (기본: bytetrack)")
    parser.add_argument("--track-buffer", type=int, default=30,
                       help="트래킹 버퍼 (기본: 30)")

    # 출력 옵션
    parser.add_argument("--hevc", action="store_true", help="HEVC 인코딩")
    parser.add_argument("--preview", action="store_true", help="미리보기")
    parser.add_argument("--max-frames", type=int, help="최대 프레임 수 (테스트용)")

    # 로깅
    parser.add_argument("--log", type=str, help="로그 파일 경로")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그")

    args = parser.parse_args()

    # 2-Pass 모드 검증
    if args.encode_only and not args.mask_json:
        parser.error("--encode-only 사용 시 --mask-json 필수")

    # 마스커 생성
    if args.simple:
        # 간단 모드
        print("🎬 간단 모드로 실행")
        masker = VideoMasker(
            mask_persons=not args.no_persons,
            mask_plates=not args.no_plates,
            mask_type=args.mask_type,
            blur_strength=args.blur_strength,
            mosaic_size=args.mosaic_size,
            person_confidence=args.person_conf,
            vehicle_confidence=args.vehicle_conf,
            person_expand=args.person_expand,
            plate_expand=args.plate_expand,
            plate_detect_mode=args.plate_detect,
            max_mask_ratio=args.max_mask_ratio,
        )
        
        masker.process_video(
            input_path=args.input,
            output_path=args.output,
            start_time=parse_time(args.start),
            end_time=parse_time(args.end),
            max_frames=args.max_frames,
            preview=args.preview,
            use_hevc=args.hevc,
            log_file=args.log,
            verbose=args.verbose,
        )
    else:
        # 최적화 모드
        masker = VideoMaskerOptimized(
            mask_persons=not args.no_persons,
            mask_plates=not args.no_plates,
            mask_type=args.mask_type,
            blur_strength=args.blur_strength,
            mosaic_size=args.mosaic_size,
            person_confidence=args.person_conf,
            vehicle_confidence=args.vehicle_conf,
            person_expand=args.person_expand,
            plate_expand=args.plate_expand,
            plate_detect_mode=args.plate_detect,
            plate_smoothing=args.plate_smoothing,
            max_mask_ratio=args.max_mask_ratio,
            device=args.device,
            detect_interval=args.detect_interval,
            detect_scale=args.detect_scale,
            batch_size=args.batch_size,
            high_performance=args.high_performance,
            use_fp16=args.fp16 if args.fp16 else None,
            use_tensorrt=args.tensorrt,
            auto_optimize=not args.no_auto,
            tracker=args.tracker,
            track_buffer=args.track_buffer,
        )

        # 2-Pass 모드 분기
        if args.two_pass:
            # 2-Pass 모드: Pass1 + Pass2 순차 실행
            print("🔄 2-Pass 모드로 실행 (GPU 최대 활용)")
            masker.process_video_2pass(
                input_path=args.input,
                output_path=args.output,
                start_time=parse_time(args.start),
                end_time=parse_time(args.end),
                use_hevc=args.hevc,
                keep_json=args.keep_json,
                log_file=args.log,
                verbose=args.verbose,
            )
        elif args.analyze_only:
            # Pass 1만 실행: 분석
            print("📊 Pass 1: 분석 모드 (마스크 좌표 JSON 저장)")
            masker.analyze_video(
                input_path=args.input,
                output_json=args.output,  # -o 옵션을 JSON 출력 경로로 사용
                start_time=parse_time(args.start),
                end_time=parse_time(args.end),
                log_file=args.log,
                verbose=args.verbose,
            )
        elif args.encode_only:
            # Pass 2만 실행: 인코딩
            print("🎬 Pass 2: 인코딩 모드 (JSON 마스크 적용)")
            masker.encode_with_masks(
                input_path=args.input,
                mask_json=args.mask_json,
                output_path=args.output,
                use_hevc=args.hevc,
                log_file=args.log,
                verbose=args.verbose,
            )
        else:
            # 기존 최적화 모드
            print("🚀 최적화 모드로 실행")
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
