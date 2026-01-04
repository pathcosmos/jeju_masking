# Video Privacy Masking Tool

4K 영상에서 얼굴과 차량 번호판을 자동으로 감지하고 마스킹하는 도구입니다.
NVIDIA CUDA / Apple Silicon MPS GPU 가속, 멀티스레딩, 트래킹 보간 등 다양한 최적화 기법을 적용하여 고해상도 영상도 효율적으로 처리합니다.

## 주요 기능

- **얼굴 감지 및 마스킹**: OpenCV DNN 기반 SSD 모델 사용 (CUDA 가속 지원)
- **차량/번호판 감지 및 마스킹**: YOLOv8n 모델 + ByteTrack/BoT-SORT 트래킹
- **NVIDIA CUDA GPU 가속**: RTX 시리즈에서 FP16 추론 및 NVENC 하드웨어 인코딩
- **OpenCV DNN CUDA 백엔드**: 얼굴 감지를 GPU에서 처리
- **MPS GPU 가속**: Apple Silicon에서 Metal Performance Shaders 활용
- **멀티스레딩 파이프라인**: 읽기/처리/쓰기 분리로 성능 향상
- **프레임 스킵 + 트래킹 보간**: 모든 프레임 감지 없이 추적으로 마스킹 유지
- **해상도 다운스케일 감지**: 감지는 축소 해상도, 출력은 원본 해상도 유지
- **배치 추론**: 여러 프레임 동시 처리로 GPU 활용률 극대화

## 파일 구조

```
jeju_masking/
├── mask_video.py            # CLI 인터페이스 (권장 진입점)
├── video_masker.py          # 핵심 마스킹 클래스 (VideoMasker, VideoMaskerOptimized)
├── masking_utils.py         # 공용 유틸리티 (로거, 시간 파싱 등)
├── encoding_utils.py        # 인코딩 유틸리티 (NVENC 설정, 시스템 최적화)
├── two_pass.py              # 2-Pass 모드 함수
├── high_performance.py      # 고성능 모드 함수
├── models/                  # 감지 모델 (자동 다운로드)
├── movs/                    # 입출력 영상 폴더
├── SETUP.md                 # 환경 설정 가이드
└── README.md                # 이 파일
```

---

## 설치

### 1. 시스템 의존성

#### macOS (Apple Silicon)

```bash
# ffmpeg 설치 (HEVC 인코딩용)
brew install ffmpeg
```

#### Ubuntu/Linux (NVIDIA GPU)

```bash
# ffmpeg 및 빌드 도구 설치
sudo apt update
sudo apt install -y ffmpeg build-essential cmake git pkg-config
```

### 2. Python 환경 설정

#### 기본 설치 (pip OpenCV - CUDA 없음)

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성
uv venv --python 3.12

# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치
uv pip install ultralytics opencv-contrib-python numpy lap pyyaml torch torchvision
```

#### NVIDIA GPU 최적화 설치 (OpenCV CUDA 빌드)

NVIDIA GPU에서 OpenCV DNN CUDA 백엔드를 사용하려면 OpenCV를 소스에서 빌드해야 합니다.
pip로 설치되는 opencv-python 패키지는 CUDA 지원이 포함되어 있지 않습니다.

자세한 빌드 가이드는 아래 [OpenCV CUDA 빌드 가이드](#opencv-cuda-빌드-가이드-nvidia-gpu) 섹션을 참조하세요.

---

## 사용법

### 기본 실행

```bash
source .venv/bin/activate
python mask_video.py input.mp4
```

### 처리 모드 선택

| 모드 | 옵션 | 특징 | 권장 사용 |
|------|------|------|-----------|
| **최적화 모드** | (기본) | 1-Pass, 멀티스레딩 파이프라인 | 일반 처리 |
| **고성능 모드** | `--high-performance` | 1-Pass, 배치 GPU 추론 + NVENC | RTX GPU 최대 활용 |
| **2-Pass 모드** | `--2pass` | Pass1(분석) + Pass2(인코딩) 분리 | JSON 재작업, 대용량 처리 |
| **간단 모드** | `--simple` | 최적화 없음, 기본 설정 | CPU 환경, 빠른 테스트 |

### 옵션 전체 목록

#### 입출력 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `input` | 입력 비디오 파일 (필수) | - |
| `-o, --output` | 출력 파일 경로 | `{input}_masked.mp4` |
| `--start` | 시작 시간 (예: `23:00`, `1:30:00`) | 처음부터 |
| `--end` | 종료 시간 (예: `28:00`) | 끝까지 |
| `--hevc` | HEVC(H.265) 인코딩 사용 | 비활성화 |

#### 마스킹 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--no-faces` | 얼굴 마스킹 비활성화 | - |
| `--no-plates` | 번호판 마스킹 비활성화 | - |
| `--mask-type` | `blur` 또는 `mosaic` | `blur` |
| `--blur-strength` | 블러 강도 (홀수 값) | `51` |
| `--mosaic-size` | 모자이크 블록 크기 | `15` |

#### 감지 파라미터

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--face-conf` | 얼굴 감지 신뢰도 임계값 | `0.4` |
| `--vehicle-conf` | 차량 감지 신뢰도 임계값 | `0.3` |
| `--face-expand` | 얼굴 영역 확장 비율 | `0.2` |
| `--plate-expand` | 번호판 영역 확장 비율 | `0.3` |

#### 2-Pass 모드 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--2pass` | 2-Pass 모드 활성화 (분석→인코딩 분리) | 비활성화 |
| `--analyze-only` | Pass 1만 실행: 마스크 좌표를 JSON으로 저장 | - |
| `--encode-only` | Pass 2만 실행: JSON 마스크 데이터로 인코딩 | - |
| `--mask-json` | 마스크 JSON 파일 경로 (`--encode-only`와 함께) | - |
| `--keep-json` | 2-Pass 완료 후 JSON 파일 유지 | 삭제 |

#### 성능 최적화 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--device` | 연산 디바이스 (`auto`, `mps`, `cuda`, `cpu`) | `auto` |
| `--detect-interval` | 감지 수행 프레임 간격 (1=매 프레임, -1=자동) | `-1` |
| `--detect-scale` | 감지용 해상도 스케일 (0.5 = 50%, -1=자동) | `-1` |
| `--batch-size` | 배치 추론 크기 (-1=자동) | `-1` |
| `--high-performance` | 고성능 모드: FFmpeg 파이프라인 + 배치 GPU 추론 | 비활성화 |
| `--fp16` | FP16 반정밀도 추론 (NVIDIA GPU 전용) | 비활성화 |
| `--tensorrt` | TensorRT 가속 | 비활성화 |
| `--no-auto` | 자동 최적화 비활성화 | - |

#### 트래킹 파라미터

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--tracker` | 트래커 종류 (`bytetrack`, `botsort`) | `bytetrack` |
| `--track-buffer` | 트래킹 버퍼 크기 (손실 허용 프레임) | `30` |
| `--match-thresh` | 트래킹 매칭 임계값 | `0.8` |
| `--iou-thresh` | IoU 임계값 | `0.5` |

#### 로깅 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--log` | 로그 파일 경로 | 자동 생성 |
| `--verbose, -v` | 상세 로그 출력 | 비활성화 |

---

## 사용 예시

### 기본 마스킹

```bash
# 전체 영상 마스킹 (사람 + 번호판 블러)
python mask_video.py video.mp4

# 특정 구간만 처리 (23분~28분)
python mask_video.py video.mp4 --start 23:00 --end 28:00

# HEVC 인코딩 (RTX GPU에서 NVENC 자동 사용)
python mask_video.py video.mp4 --hevc
```

### 고성능 모드 (RTX GPU 권장)

```bash
# 1-Pass 고성능 모드: 배치 GPU 추론 + NVENC 인코딩
# RTX 4070 SUPER 기준 4K 60fps에서 ~48 fps 달성
python mask_video.py video.mp4 --high-performance --hevc

# FP16 반정밀도 추론 (메모리 절약, 속도 향상)
python mask_video.py video.mp4 --high-performance --fp16 --hevc
```

### 2-Pass 모드 (GPU 최대 활용, 재작업 가능)

```bash
# 2-Pass 모드: Pass1(분석) → Pass2(인코딩) 순차 실행
python mask_video.py video.mp4 --2pass --hevc

# JSON 파일 유지 (재인코딩 가능)
python mask_video.py video.mp4 --2pass --hevc --keep-json
```

### 2-Pass 분리 실행 (일괄 처리)

```bash
# 여러 영상을 분석만 먼저 실행 (GPU 100% YOLO 추론)
python mask_video.py video1.mp4 --analyze-only
python mask_video.py video2.mp4 --analyze-only
python mask_video.py video3.mp4 --analyze-only

# 분석 완료 후 순차 인코딩 (GPU 100% NVENC)
python mask_video.py video1.mp4 --encode-only --mask-json video1_masks.json --hevc
python mask_video.py video2.mp4 --encode-only --mask-json video2_masks.json --hevc
python mask_video.py video3.mp4 --encode-only --mask-json video3_masks.json --hevc
```

### 번호판/사람 선택 마스킹

```bash
# 번호판만 마스킹 (사람 제외)
python mask_video.py video.mp4 --no-persons

# 사람만 마스킹 (번호판 제외)
python mask_video.py video.mp4 --no-plates

# 모자이크 처리
python mask_video.py video.mp4 --mask-type mosaic --mosaic-size 20
```

### 정확도 vs 속도 조절

```bash
# 매 프레임 감지 (정확도 최대, 속도 느림)
python mask_video.py video.mp4 --detect-interval 1

# 5프레임마다 감지 (속도 최대, 정확도 감소)
python mask_video.py video.mp4 --detect-interval 5 --detect-scale 0.25
```

### 트래킹 조정

```bash
# 긴 트래킹 버퍼 (객체가 일시적으로 사라져도 추적 유지)
python mask_video.py video.mp4 --track-buffer 60
```

### 실시간 로그 모니터링

```bash
# 상세 로그 + 파일 저장
python mask_video.py video.mp4 -v --log masking.log

# 별도 터미널에서 실시간 확인
tail -f masking.log
```

---

## 성능 참고

### 테스트 환경

- **GPU**: NVIDIA RTX 4070 SUPER (12GB VRAM)
- **CPU**: Intel i5-13400 (10코어, 16스레드)
- **RAM**: 32GB DDR5
- **입력**: 4K 60fps 영상 (3840x2160)

### 처리 모드별 성능 비교

| 모드 | 속도 (fps) | 특징 | 권장 사용 |
|------|------------|------|-----------|
| **고성능 모드** (`--high-performance`) | ~48 fps | 1-Pass, GPU 추론+인코딩 동시 | 빠른 처리 |
| **2-Pass 모드** (`--2pass`) | ~22 fps (분석) + ~20 fps (인코딩) | 2-Pass, JSON 저장 | 재작업 필요 시 |
| **최적화 모드** (기본) | ~30 fps | 1-Pass, 안정적 | 일반 사용 |

### 2-Pass 모드 상세

| Pass | 동작 | GPU 사용 | 출력 |
|------|------|----------|------|
| **Pass 1** (분석) | YOLO 배치 추론 | GPU 100% 추론 | JSON 마스크 파일 |
| **Pass 2** (인코딩) | 마스크 적용 + NVENC | GPU 100% 인코딩 | 마스킹된 영상 |

### NVENC 인코딩 최적화

시스템 사양에 따라 자동으로 최적화된 NVENC 설정이 적용됩니다:

| 항목 | RTX 4070 SUPER (12GB) | RTX 3060 (8GB) | GTX 1660 (6GB) |
|------|----------------------|----------------|----------------|
| 프리셋 | p4 (quality) | p3 (balanced) | p2 (fast) |
| Lookahead | 32 프레임 | 24 프레임 | 16 프레임 |
| B-프레임 | 4 | 3 | 2 |
| Surfaces | 16 | 12 | 8 |
| FP16 추론 | 활성화 | 활성화 | 비활성화 |

### 감지 통계 예시 (1분 4K 영상)

- 처리 프레임: 3,596
- 감지된 사람: 0
- 감지된 번호판: 19,021

---

## 2-Pass 모드 상세

### 작동 원리

2-Pass 모드는 GPU 자원을 최대한 활용하기 위해 분석과 인코딩을 분리합니다:

```
┌─────────────────────────────────────────────────────────────┐
│ Pass 1: 분석 (GPU 100% YOLO 추론)                            │
│                                                             │
│   비디오 → [디코더] → [YOLO 배치 추론] → JSON 마스크 파일     │
│                     (사람/번호판 감지)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Pass 2: 인코딩 (GPU 100% NVENC)                              │
│                                                             │
│   비디오 + JSON → [마스크 적용] → [NVENC 인코딩] → 출력 영상  │
└─────────────────────────────────────────────────────────────┘
```

### JSON 마스크 형식

```json
{
  "metadata": {
    "version": "1.0",
    "input": "video.mp4",
    "width": 3840,
    "height": 2160,
    "fps": 59.94,
    "total_frames": 3596,
    "start_frame": 0,
    "end_frame": 3596,
    "created": "2026-01-04T12:51:53"
  },
  "stats": {
    "total_persons": 0,
    "total_plates": 19021
  },
  "frames": {
    "0": {
      "persons": [],
      "plates": [[1520, 1080, 1680, 1140]]
    },
    "1": {
      "persons": [],
      "plates": [[1522, 1082, 1682, 1142], [2100, 1500, 2200, 1550]]
    }
  }
}
```

### 사용 시나리오

| 시나리오 | 명령어 | 설명 |
|----------|--------|------|
| 일반 처리 | `--2pass` | Pass1 + Pass2 순차 실행, JSON 자동 삭제 |
| 재작업 대비 | `--2pass --keep-json` | JSON 파일 유지 |
| 일괄 분석 | `--analyze-only` | 여러 영상 분석만 먼저 실행 |
| 일괄 인코딩 | `--encode-only --mask-json file.json` | 분석된 영상 순차 인코딩 |
| 파라미터 변경 | `--encode-only --mask-json file.json --mask-type mosaic` | 동일 분석 결과로 다른 마스킹 |

---

## 모델 정보

| 모델 | 용도 | 소스 |
|------|------|------|
| OpenCV DNN SSD | 얼굴 감지 | 자동 다운로드 |
| YOLOv8n | 차량 감지 | Ultralytics (자동 다운로드) |
| ByteTrack | 객체 추적 | Ultralytics 내장 |
| BoT-SORT | 객체 추적 (대안) | Ultralytics 내장 |

---

## 변경 이력

### v3.1 - 2026-01-04 (현재)

**코드 모듈화**
- **파일 분리**: 대규모 단일 파일을 기능별 모듈로 분리
  - `encoding_utils.py`: 시스템 정보 수집, NVENC 설정, FFmpeg 명령어 빌더
  - `two_pass.py`: 2-Pass 모드 함수 (`analyze_video`, `encode_with_masks`, `process_video_2pass`)
  - `high_performance.py`: 고성능 모드 함수 (`process_video_high_performance`)
  - `masking_utils.py`: 공용 유틸리티 (로거 설정, 시간 파싱)
- **CLI 통합**: `mask_video.py`를 단일 진입점으로 통합
  - 기존 `mask_video_optimized.py`, `mask_video_advanced.py` 기능 포함

**2-Pass 멀티스레딩**
- **Pass 1 멀티스레딩**: `decoder_thread` + `analyzer_thread` 비동기 처리
- **Pass 2 멀티스레딩**: `decoder_thread` + `masker_thread` + `encoder_thread` 파이프라인
- **성능 개선**: 2-Pass 총 시간 6.7분 → 5.5분 (18% 향상)

### v3.0 - 2026-01-04

**NVENC 인코딩 최적화**
- **최적화된 NVENC 설정**: `build_nvenc_command()` 함수 추가
  - `rc-lookahead`: 프레임 미리보기로 품질 향상 (최대 32프레임)
  - `spatial-aq`: 공간 적응형 양자화 (디테일 보존)
  - `temporal-aq`: 시간 적응형 양자화 (움직임 품질)
  - `surfaces`: 동시 인코딩 표면 (처리량 증가)
  - `b_ref_mode`: B-프레임 참조 (압축 효율)
- **시스템 자동 최적화**: `get_optimal_settings()` 함수 개선
  - CPU 코어/스레드 기반 FFmpeg 스레드 수 자동 설정
  - RAM 용량 기반 큐 크기 자동 조정
  - VRAM 용량 기반 배치 크기 자동 계산
  - GPU 아키텍처 기반 FP16/프리셋 자동 선택

**2-Pass 모드 추가**
- **Pass 1 (분석)**: GPU 100% YOLO 추론, 마스크 좌표를 JSON으로 저장
  - 프레임별 마스크 바운딩 박스 저장
  - 사람/번호판 감지 통계 포함
  - 메타데이터 (해상도, FPS, 총 프레임 수) 저장
- **Pass 2 (인코딩)**: GPU 100% NVENC 인코딩, JSON 마스크 적용
  - JSON 마스크 데이터 로드
  - 프레임별 마스크 적용 + NVENC 하드웨어 인코딩
- **분리 실행 지원**: `--analyze-only`, `--encode-only`, `--mask-json` 옵션
  - 여러 영상 일괄 분석 후 순차 인코딩 가능
  - JSON 파일 유지로 재인코딩 지원 (`--keep-json`)

**고성능 모드 개선**
- **멀티스레딩 파이프라인**: 디코딩, 추론, 인코딩 비동기 처리
  - `decoder_thread`: 비동기 프레임 읽기
  - `processor_thread`: 배치 GPU 추론 + 마스킹
  - `encoder_thread`: NVENC 순서 보장 인코딩
- **RTX 4070 SUPER 기준**: 4K 60fps에서 ~48 fps 달성

### v2.4.1 - 2026-01-03

**버그 수정**
- **FP16 추론 오류 수정**: `yolo_half` 변수 초기화 시 잘못된 변수 참조 수정
  - 이전: `self.yolo_half = use_fp16 and self.device == 'cuda'` (파라미터 `use_fp16`은 `None`일 수 있음)
  - 수정: `self.yolo_half = self.use_fp16 and self.device == 'cuda'` (인스턴스 변수 사용)
  - 증상: `unsupported operand type(s) for &=: 'NoneType' and 'bool'` 오류로 배치 처리 실패
  - 영향: 모든 프레임에서 마스킹이 적용되지 않고 원본 프레임만 인코딩됨

### v2.4 - 2026-01-03

**NVIDIA CUDA 최적화**
- **OpenCV DNN CUDA 백엔드 지원**: 얼굴 감지를 GPU에서 처리
  - `check_opencv_cuda_support()`: OpenCV CUDA/cuDNN 지원 여부 확인
  - `FaceDetectorDNN` 클래스에 CUDA 백엔드 자동 선택 로직 추가
  - CUDA 사용 불가 시 자동으로 CPU fallback
- **시스템 하드웨어 자동 감지**: `get_system_info()` 함수 추가
  - CPU 모델, 코어/스레드 수, 최대 클럭
  - RAM 총량 및 사용 가능량
  - GPU 모델, VRAM, CUDA 버전, 아키텍처 감지
- **하드웨어 기반 자동 최적화**: `optimize_settings_for_hardware()` 함수
  - CPU 스레드 수 기반 워커 수 자동 조정 (threads // 2)
  - RAM 기반 프레임 큐 크기 계산
  - VRAM 기반 배치 크기 자동 계산 (VRAM 70% 활용)
  - 12GB+ VRAM: detect_interval=1 (매 프레임 감지 가능)
- **GPU 메모리 최적화**:
  - cuDNN benchmark 모드 활성화
  - TF32 연산 활성화 (Ampere+)
  - 주기적 GPU 캐시 정리
- **NVENC 하드웨어 인코딩**: RTX GPU에서 hevc_nvenc 자동 사용
  - NVENC 실패 시 libx265 소프트웨어 인코딩 fallback

**Python 환경 변경**
- Python 3.11 → 3.12로 업그레이드 (OpenCV CUDA 빌드 호환성)
- NumPy 2.x → 1.26.4로 다운그레이드 (OpenCV 바인딩 호환성)

### v2.1 - 2026-01-02

**버그 수정**
- **차량 감지 버그 수정**: `detect_batch()`에서 `predict()` 대신 `track()` 사용
  - 이전: 배치 추론 시 `track_id`가 None이 되어 차량 감지 수가 항상 0
  - 수정: 프레임별 `track()` 호출로 트래킹 ID 유지
- **폴백 ID 생성**: `track_id`가 None인 경우 `_get_next_vehicle_id()`로 고유 ID 할당
- **배치 결과 초기화**: `batch_detections` 리스트 사전 초기화로 인덱스 오류 방지

**개선**
- 실시간 로그 출력을 위한 `FlushStreamHandler` 추가
- 진행률 로그 포맷 개선 (fps, 남은 시간, 감지 수 표시)

### v2.0 - 2026-01-02

**성능 최적화 (mask_video_optimized.py 신규 생성)**
- **MPS GPU 가속**: Apple Silicon에서 Metal Performance Shaders 활용
  - `get_optimal_device()` 함수로 자동 감지
  - YOLO `track()` 호출 시 `device` 파라미터 전달
- **멀티스레딩 파이프라인**:
  - `FrameReader`: 별도 스레드에서 프레임 읽기
  - `FrameWriter`: 별도 스레드에서 프레임 쓰기
  - Queue 기반 비동기 처리
- **프레임 스킵 + 트래킹 보간**:
  - `--detect-interval` 옵션으로 감지 주기 설정
  - `TrackingInterpolator` 클래스로 중간 프레임 위치 예측
  - 속도 기반 선형 보간으로 부드러운 마스킹
- **해상도 다운스케일 감지**:
  - `--detect-scale` 옵션으로 감지용 해상도 축소
  - 출력은 원본 해상도 유지
- **배치 추론**:
  - `--batch-size` 옵션으로 동시 처리 프레임 수 설정
  - `detect_batch()` 메서드로 여러 프레임 동시 처리
- **확장된 트래킹 파라미터**:
  - `--tracker`: bytetrack 또는 botsort 선택
  - `--track-buffer`: 트래킹 버퍼 크기
  - `--match-thresh`: 매칭 임계값
  - `--iou-thresh`: IoU 임계값
  - `create_custom_tracker_config()`: 커스텀 YAML 설정 생성

### v1.0 - 초기 버전

**기본 기능 (mask_video_advanced.py)**
- OpenCV DNN 기반 얼굴 감지
- YOLOv8n 기반 차량 감지
- 기본 ByteTrack 트래킹
- 블러/모자이크 마스킹
- 시작/종료 시간 구간 설정
- HEVC 인코딩 옵션

---

## 문제 해결

### MPS 사용 불가

```
RuntimeError: MPS backend out of memory
```

해결: 배치 크기 줄이기
```bash
python mask_video_optimized.py video.mp4 --batch-size 2
```

### 감지 누락이 많은 경우

해결: 감지 간격 줄이기 + 신뢰도 낮추기
```bash
python mask_video_optimized.py video.mp4 --detect-interval 1 --face-conf 0.3 --vehicle-conf 0.2
```

### 트래킹이 자주 끊기는 경우

해결: 트래킹 버퍼 늘리기 + 매칭 임계값 낮추기
```bash
python mask_video_optimized.py video.mp4 --track-buffer 60 --match-thresh 0.6
```

### 처리 속도가 너무 느린 경우

해결: 감지 간격 늘리기 + 해상도 낮추기
```bash
python mask_video_optimized.py video.mp4 --detect-interval 5 --detect-scale 0.25
```

---

## TODO (개선 필요 사항)

### 1. 자동차 번호판 위치 탐지 정확도 개선

- **현재 문제**: 차량마다 번호판 위치가 상이하여 정확한 탐지가 어려움
- **원인 추정**: YOLOv8n은 차량 전체를 감지하며, 번호판 영역은 차량 바운딩 박스 하단 일부로 추정
- **개선 방향**:
  - 번호판 전용 감지 모델 도입 (예: 번호판 특화 YOLO 모델)
  - 차량 종류별 번호판 위치 비율 학습
  - OCR 기반 번호판 텍스트 영역 감지

### 2. 화면 전체 마스킹 오류 수정

- **현재 문제**: 특정 프레임 또는 연속된 프레임에서 화면 전체가 마스킹되는 현상 발생
- **원인 추정**:
  - 트래킹 보간 시 비정상적으로 큰 바운딩 박스 생성
  - 감지 결과의 좌표값 오류
  - 스케일 변환 시 좌표 계산 오류
- **개선 방향**:
  - 바운딩 박스 크기 상한선 설정 (화면 대비 비율 제한)
  - 이상치 감지 및 필터링 로직 추가
  - 프레임 간 바운딩 박스 크기 급변 감지

### 3. 사람 얼굴 감지 및 마스킹 개선

- **현재 문제**: 얼굴 인식 및 마스킹 기능이 부족함
- **원인 추정**:
  - OpenCV DNN SSD 모델의 한계 (정면 얼굴 위주, 원거리/측면 감지 약함)
  - 4K 영상에서 작은 얼굴 감지 어려움
- **개선 방향**:
  - 사람 객체 전체 감지 후 전신 마스킹 옵션 추가
  - 더 강력한 얼굴 감지 모델 도입 (RetinaFace, YOLO-Face 등)
  - 사람 감지 + 얼굴 감지 2단계 파이프라인 구성
  - `--mask-person` 옵션으로 사람 전체 마스킹 선택 가능하도록

### 4. NVIDIA GPU 최적화 세팅 (v2.4에서 일부 완료)

- **완료된 항목** (v2.4):
  - ✅ OpenCV DNN CUDA 백엔드 지원 (얼굴 감지 GPU 가속)
  - ✅ 시스템 하드웨어 자동 감지 및 최적화
  - ✅ VRAM 기반 배치 크기 자동 계산
  - ✅ cuDNN benchmark 모드, TF32 활성화
  - ✅ NVENC 하드웨어 인코딩 지원
  - ✅ FP16 추론 (half=True)
- **추가 개선 필요**:
  - TensorRT 변환을 통한 추론 속도 향상
  - 멀티 GPU 지원 (대용량 영상 병렬 처리)
  - CUDA 스트림을 활용한 비동기 처리
  - Docker 이미지 제공 (CUDA 환경 포함)

---

## OpenCV CUDA 빌드 가이드 (NVIDIA GPU)

pip로 설치되는 `opencv-python` 패키지는 CUDA 지원이 포함되어 있지 않습니다.
OpenCV DNN CUDA 백엔드를 사용하려면 소스에서 직접 빌드해야 합니다.

### 테스트 환경

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 24.04 LTS |
| GPU | NVIDIA RTX 4070 Super (12GB VRAM) |
| CUDA | 12.0 |
| cuDNN | 9.17.1 |
| Python | 3.12.3 |
| OpenCV | 4.10.0 |
| GCC | 12 (CUDA 12.0 호환) |

### 1. 빌드 의존성 설치

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3.12-dev python3-numpy \
    libtbb-dev libdc1394-dev \
    gcc-12 g++-12
```

> **중요**: CUDA 12.0은 GCC 12 이하만 지원합니다. Ubuntu 24.04는 기본 GCC 13+를 사용하므로 gcc-12를 별도 설치해야 합니다.

### 2. OpenCV 소스 다운로드

```bash
cd /tmp
git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git
git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git
```

### 3. CMake 설정 및 빌드

```bash
mkdir -p /tmp/opencv_build && cd /tmp/opencv_build

# GCC 12 지정 (CUDA 12.0 호환성)
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

# GPU 아키텍처 확인 (예: RTX 4070 = 8.9)
# https://developer.nvidia.com/cuda-gpus 참조

cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
    -D CMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -D CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D CUDA_ARCH_BIN=8.9 \
    -D WITH_CUBLAS=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3.12 \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    /tmp/opencv

# 빌드 (CPU 코어 수에 맞게 -j 옵션 조정)
make -j16

# 설치
sudo make install
sudo ldconfig
```

#### GPU 아키텍처 (CUDA_ARCH_BIN) 참조

| GPU 시리즈 | 아키텍처 | CUDA_ARCH_BIN |
|-----------|----------|---------------|
| RTX 40xx (Ada Lovelace) | sm_89 | 8.9 |
| RTX 30xx (Ampere) | sm_86 | 8.6 |
| RTX 20xx (Turing) | sm_75 | 7.5 |
| GTX 10xx (Pascal) | sm_61 | 6.1 |

### 4. Python 가상환경 설정

```bash
cd /path/to/jeju_masking

# Python 3.12로 가상환경 생성 (OpenCV 빌드 버전과 일치)
uv venv --python python3.12
source .venv/bin/activate

# 필수 패키지 설치 (opencv 제외)
uv pip install ultralytics torch torchvision numpy lap pyyaml

# pip opencv 제거 (의존성으로 설치된 경우)
uv pip uninstall opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

# 시스템 OpenCV CUDA를 venv에 심링크
SITE_PACKAGES=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
ln -sf /usr/local/lib/python3.12/dist-packages/cv2 $SITE_PACKAGES/cv2

# NumPy 다운그레이드 (OpenCV 바인딩 호환성)
uv pip install "numpy<2"
```

### 5. 설치 확인

```bash
.venv/bin/python -c "
import cv2
print('OpenCV version:', cv2.__version__)
print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())

# CUDA 빌드 정보 확인
build = cv2.getBuildInformation()
for line in build.split('\n'):
    if 'CUDA' in line or 'cuDNN' in line:
        print(line)
"
```

예상 출력:
```
OpenCV version: 4.10.0
CUDA devices: 1
  NVIDIA CUDA:                   YES (ver 12.0, CUFFT CUBLAS FAST_MATH)
    NVIDIA GPU arch:             89
  cuDNN:                         YES (ver 9.17.1)
```

### 6. 문제 해결

#### GCC 버전 오류

```
error: #error -- unsupported GNU version! gcc versions later than 12 are not supported!
```

해결: CMake에서 GCC 12 명시적 지정
```bash
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
cmake -D CMAKE_C_COMPILER=/usr/bin/gcc-12 -D CMAKE_CXX_COMPILER=/usr/bin/g++-12 ...
```

#### NumPy 버전 오류

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

해결: NumPy 다운그레이드
```bash
uv pip install "numpy<2"
```

#### Python 버전 불일치

OpenCV는 빌드 시 지정한 Python 버전의 바인딩만 생성합니다.
venv Python 버전이 빌드 시 사용한 버전과 일치해야 합니다.

```bash
# 빌드 시 Python 3.12 사용한 경우
uv venv --python python3.12
```

---

## 라이선스

이 프로젝트는 개인 프라이버시 보호를 위한 목적으로 제작되었습니다.

### 사용된 오픈소스

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - AGPL-3.0
- [OpenCV](https://opencv.org/) - Apache 2.0
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - MIT
