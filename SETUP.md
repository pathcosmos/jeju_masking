# 설치 및 실행 가이드

## 요구사항

- macOS (Apple Silicon 또는 Intel)
- Python 3.11+
- Homebrew

## 1. 시스템 의존성 설치

```bash
# ffmpeg 설치 (HEVC 인코딩용)
brew install ffmpeg
```

## 2. Python 환경 설정

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성
uv venv --python 3.11

# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치
uv pip install ultralytics opencv-python-headless numpy lap
```

## 3. 실행

### 기본 사용법

```bash
# 가상환경 활성화
source .venv/bin/activate

# 전체 영상 마스킹 (얼굴 + 번호판 블러)
python mask_video_advanced.py input.mp4

# 특정 구간만 처리 (23분~28분)
python mask_video_advanced.py input.mp4 --start 23:00 --end 28:00

# HEVC 인코딩으로 출력
python mask_video_advanced.py input.mp4 --start 23:00 --end 28:00 --hevc
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-o, --output` | 출력 파일 경로 | 자동 생성 |
| `--start` | 시작 시간 (예: 23:00) | 처음부터 |
| `--end` | 종료 시간 (예: 28:00) | 끝까지 |
| `--hevc` | HEVC(H.265) 인코딩 | 비활성화 |
| `--no-faces` | 얼굴 마스킹 비활성화 | - |
| `--no-plates` | 번호판 마스킹 비활성화 | - |
| `--mask-type` | blur 또는 mosaic | blur |
| `--blur-strength` | 블러 강도 (홀수) | 51 |
| `--mosaic-size` | 모자이크 블록 크기 | 15 |
| `--verbose, -v` | 상세 로그 출력 | 비활성화 |
| `--log` | 로그 파일 경로 | 자동 생성 |

### 예시

```bash
# 모자이크 처리
python mask_video_advanced.py input.mp4 --mask-type mosaic

# 번호판만 처리
python mask_video_advanced.py input.mp4 --no-faces

# 상세 로그와 함께 실행
python mask_video_advanced.py input.mp4 --start 10:00 --end 15:00 --hevc --verbose
```

## 모델 정보

- **얼굴 감지**: OpenCV DNN (SSD 기반, 자동 다운로드)
- **차량 감지**: YOLOv8n (자동 다운로드)

## 처리 시간 참고

4K 60fps 영상 기준 (Apple M 시리즈):
- 마스킹: 약 20분/1분 영상
- HEVC 인코딩: 약 2분/1분 영상
