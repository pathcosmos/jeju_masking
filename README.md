# Video Privacy Masking Tool

4K 영상에서 얼굴과 차량 번호판을 자동으로 감지하고 마스킹하는 도구입니다.
Apple Silicon MPS GPU 가속, 멀티스레딩, 트래킹 보간 등 다양한 최적화 기법을 적용하여 고해상도 영상도 효율적으로 처리합니다.

## 주요 기능

- **얼굴 감지 및 마스킹**: OpenCV DNN 기반 SSD 모델 사용
- **차량/번호판 감지 및 마스킹**: YOLOv8n 모델 + ByteTrack/BoT-SORT 트래킹
- **MPS GPU 가속**: Apple Silicon에서 Metal Performance Shaders 활용
- **멀티스레딩 파이프라인**: 읽기/처리/쓰기 분리로 성능 향상
- **프레임 스킵 + 트래킹 보간**: 모든 프레임 감지 없이 추적으로 마스킹 유지
- **해상도 다운스케일 감지**: 감지는 축소 해상도, 출력은 원본 해상도 유지
- **배치 추론**: 여러 프레임 동시 처리로 GPU 활용률 극대화

## 파일 구조

```
jeju_masking/
├── mask_video_optimized.py  # 최적화 버전 (권장)
├── mask_video_advanced.py   # 기본 버전
├── mask_video.py            # 초기 버전
├── models/                  # 얼굴 감지 모델 (자동 다운로드)
├── mov/                     # 입출력 영상 폴더
├── SETUP.md                 # 환경 설정 가이드
└── README.md                # 이 파일
```

---

## 설치

### 1. 시스템 의존성

```bash
# ffmpeg 설치 (HEVC 인코딩용)
brew install ffmpeg
```

### 2. Python 환경 설정

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성
uv venv --python 3.11

# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치
uv pip install ultralytics opencv-python-headless numpy lap pyyaml torch torchvision
```

---

## 사용법

### 기본 실행

```bash
source .venv/bin/activate
python mask_video_optimized.py input.mp4
```

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

#### 성능 최적화 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--device` | 연산 디바이스 (`auto`, `mps`, `cuda`, `cpu`) | `auto` |
| `--detect-interval` | 감지 수행 프레임 간격 (1=매 프레임) | `3` |
| `--detect-scale` | 감지용 해상도 스케일 (0.5 = 50%) | `0.5` |
| `--batch-size` | 배치 추론 크기 | `4` |

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
# 전체 영상 마스킹 (얼굴 + 번호판 블러)
python mask_video_optimized.py video.mp4
```

### 구간 지정

```bash
# 23분~28분 구간만 처리
python mask_video_optimized.py video.mp4 --start 23:00 --end 28:00
```

### HEVC 출력

```bash
# HEVC 인코딩으로 파일 크기 감소
python mask_video_optimized.py video.mp4 --hevc
```

### 모자이크 처리

```bash
# 블러 대신 모자이크
python mask_video_optimized.py video.mp4 --mask-type mosaic --mosaic-size 20
```

### 번호판만 처리

```bash
# 얼굴 마스킹 비활성화 (번호판만)
python mask_video_optimized.py video.mp4 --no-faces
```

### 고성능 설정

```bash
# 매 프레임 감지 + 원본 해상도 감지 (정확도 최대, 속도 느림)
python mask_video_optimized.py video.mp4 --detect-interval 1 --detect-scale 1.0
```

### 빠른 처리 설정

```bash
# 5프레임마다 감지 + 25% 해상도 (속도 최대, 정확도 감소)
python mask_video_optimized.py video.mp4 --detect-interval 5 --detect-scale 0.25
```

### 트래킹 조정

```bash
# 긴 트래킹 버퍼 (객체가 일시적으로 사라져도 추적 유지)
python mask_video_optimized.py video.mp4 --track-buffer 60 --match-thresh 0.7
```

### 실시간 로그 모니터링

```bash
# 상세 로그 + 파일 저장
python mask_video_optimized.py video.mp4 -v --log masking.log

# 별도 터미널에서 실시간 확인
tail -f masking.log
```

---

## 성능 참고

### 테스트 환경

- **장비**: Apple M 시리즈 (Apple Silicon)
- **입력**: 4K 60fps 영상 (3840x2160)
- **설정**: 기본값 (detect-interval=3, detect-scale=0.5)

### 처리 속도

| 영상 길이 | 처리 시간 | 평균 fps | 실시간 대비 |
|-----------|-----------|----------|-------------|
| 1분 | ~2분 | ~30 fps | 0.5x |
| 15분 | ~31분 | ~29 fps | 0.49x |

### 감지 통계 예시 (15분 4K 영상)

- 처리 프레임: 53,975
- 감지된 얼굴: 2,925
- 감지된 차량: 126,342

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

### v2.1 - 2026-01-02 (현재)

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

### 4. NVIDIA GPU 최적화 세팅

- **현재 상태**: Apple Silicon MPS 가속만 구현됨
- **필요 사항**:
  - CUDA 환경에서의 성능 최적화 테스트 및 튜닝
  - TensorRT 변환을 통한 추론 속도 향상
  - GPU 메모리 사용량 최적화
  - 멀티 GPU 지원 (대용량 영상 병렬 처리)
- **개선 방향**:
  - `--device cuda` 옵션 최적화 검증
  - FP16/INT8 양자화 적용
  - CUDA 스트림을 활용한 비동기 처리
  - Docker 이미지 제공 (CUDA 환경 포함)

---

## 라이선스

이 프로젝트는 개인 프라이버시 보호를 위한 목적으로 제작되었습니다.

### 사용된 오픈소스

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - AGPL-3.0
- [OpenCV](https://opencv.org/) - Apache 2.0
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - MIT
