#!/bin/bash
# 배치 필터링 스크립트
# 파일을 순차적으로 처리

cd /home/lanco/taketimes/jeju_masking
source .venv/bin/activate

INPUT_DIR="/home/lanco/output_encoded_movs/4k_420_10bit/day"
OUTPUT_DIR="/home/lanco/taketimes/jeju_masking/output_masked"
LOG_FILE="$OUTPUT_DIR/batch_filter_$(date +%Y%m%d_%H%M%S).log"

# 처리할 파일 목록
FILES=(
    "01-222_4k_420_10bit.mp4"
    "01-333_4k_420_10bit.mp4"
    "01-444_4k_420_10bit.mp4"
    "01-555_4k_420_10bit.mp4"
    "01-666_4k_420_10bit.mp4"
    "01-777_4k_420_10bit.mp4"
    "111_4k_420_10bit.mp4"
    "222_4k_420_10bit.mp4"
    "333_4k_420_10bit.mp4"
)

echo "============================================================" | tee -a "$LOG_FILE"
echo "배치 필터링 시작: $(date)" | tee -a "$LOG_FILE"
echo "총 파일 수: ${#FILES[@]}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

TOTAL=${#FILES[@]}
CURRENT=0
START_TIME=$(date +%s)

for FILE in "${FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    INPUT_PATH="$INPUT_DIR/$FILE"
    
    # 출력 파일명 생성 (확장자 제거 후 _filtered 추가)
    BASENAME="${FILE%_4k_420_10bit.mp4}"
    OUTPUT_PATH="$OUTPUT_DIR/${BASENAME}_filtered.mp4"
    
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$CURRENT/$TOTAL] 처리 중: $FILE" | tee -a "$LOG_FILE"
    echo "시작 시간: $(date)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    
    # 파일 존재 확인
    if [ ! -f "$INPUT_PATH" ]; then
        echo "⚠️  파일 없음: $INPUT_PATH" | tee -a "$LOG_FILE"
        continue
    fi
    
    # 필터링 실행 (최대 성능 모드)
    python filter_video.py "$INPUT_PATH" \
        -o "$OUTPUT_PATH" \
        --high-performance \
        --cg-interval 1000 \
        --cg-smooth 300 \
        --verbose 2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 완료: $FILE" | tee -a "$LOG_FILE"
    else
        echo "❌ 실패: $FILE (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
    fi
    
    echo "종료 시간: $(date)" | tee -a "$LOG_FILE"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "배치 필터링 완료!" | tee -a "$LOG_FILE"
echo "총 소요 시간: ${ELAPSED_MIN}분" | tee -a "$LOG_FILE"
echo "종료 시간: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
