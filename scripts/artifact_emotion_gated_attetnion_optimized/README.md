# 재시작 가능한 Artifact–Emotion gated 실험

현재 실행 중인 `scripts/artifact_emotion_gated_attetnion`의 코드는 수정하지 않고 이 폴더에 별도 구현했다. 동일한 FAV train/dev/test split, frozen encoder, branch별 label, 128차원 projection/gate, OR score `1-(1-Vs)(1-As)`를 사용한다. 과거 워크스테이션 소스/미디어 해시는 요구하지 않는다. 새 실행의 저장 파일·설정 일관성 체크는 유지한다.

## 실행

저장소 루트의 PowerShell에서 실행한다. 스크립트는 이 프로젝트 `.venv/Scripts/python.exe`를 사용한다.

```powershell
$runner = './scripts/artifact_emotion_gated_attetnion_optimized/run_experiment.ps1'
& $runner -Output reports/artifact_emotion_gated_optimized_v1 -LegacyVideoTrain reports/artifact_emotion_gated_v2/cache/video/train
```

중단 후에는 **동일한 인자**에 `-Resume`만 추가한다.

```powershell
& $runner -Output reports/artifact_emotion_gated_optimized_v1 -LegacyVideoTrain reports/artifact_emotion_gated_v2/cache/video/train -Resume
```

기존 장기 실행은 자동으로 종료하지 않는다. 전환하려면 기존 실행 터미널에서 Ctrl+C로 중단한 뒤 위 명령을 사용한다. 원본 output은 삭제하거나 덮어쓰지 않는다. 기존에 완료된 감정 token NPZ를 읽어 활용하며, 기존 RAM에만 있던 artifact embedding은 다시 계산한다. 원본 코드 자체에는 새 resume 기능이 소급 적용되지 않는다.

## 저장 및 복구

- `cache/<modality>/<split>/samples/<sample digest>.npz`: artifact embedding/probability, emotion embedding, 시간별 tokens와 valid mask를 샘플마다 저장한다.
- `.partial` 쓰기 → flush/fsync → atomic rename → checksum commit marker 순서로 저장한다. 중단으로 미완료된 샘플은 재계산한다. 이미 완료된 샘플은 불러온다.
- OS writer lock으로 동일 output의 동시 쓰기를 차단한다. 프로세스 종료 시 lock은 해제된다.
- `progress.json` 및 터미널에 진행 수, `resumed`, `legacy_tokens`, `new`를 표시한다. 재시작 초반 ETA에는 빠른 캐시 읽기가 포함되므로 남은 추출 속도를 대표하지 않을 수 있다.
- split이 끝나면 기존 학습 인터페이스와 호환되는 `features.npz`, `manifest.json`, `COMPLETED.json`을 생성한다.
- 전체 pipeline은 완료한 extraction/head 단계를 건너뛴다. **head 학습은 epoch 중간 복구가 아니라, 중단된 attempt를 보존하고 새 attempt에서 다시 학습한다.** 상대적으로 짧은 단계다. 평가 중단도 새 attempt에 다시 기록한다.
- 재개 시 코드, 설정, split, checkpoint 등이 동일해야 한다. 같은 실행에서 저장 후 미디어 크기/수정시각이 바뀌면 차단한다. 이것은 과거 워크스테이션 재현 검증이 아니라 현재 실행의 일관성 검사다.

## 최적화

영상 디코딩 3회에서 2회로 축소하고, artifact용 선택 프레임과 얼굴 crop을 같은 decode pass에서 만든다. HSEmotion은 사용하지 않는 감정 확률 계산 및 중복 embedding forward를 제거했다. 기존 frame 선택, Haar 검출 설정, normalization, temporal pooling은 유지한다. 오디오 입력 recipe는 변경하지 않았다.

기본 `-FaceBatchSize 1`은 기존 감정 embedding과 동일한 GPU 연산 형태를 유지한다. 기존 감정 token 재사용은 이 설정에서만 허용한다. 별도 새 실험에서 `-FaceBatchSize 32`를 사용할 수 있지만 `-LegacyVideoTrain`은 함께 사용하지 않는다. batch 크기에 따른 TF32 연산 경로 차이로 기존 embedding과 작은 차이가 측정되었다. 따라서 기존 결과와 엄밀히 동일하다고 간주하지 않는다. 모든 encoder는 eval/frozen이고 새 학습 대상은 gated head다.

## 검증과 측정

2026-09-28, RTX 3090, 기존 장기 추출과 GPU/CPU를 공유한 상태에서 FAV train 영상 8개(1.88~14.04초)를 길이별로 선택하여 기존/새 경로를 교대로 측정했다. 이는 전체 데이터셋 처리량의 대표 추정치가 아니다.

| 경로 | 평균 초/영상 | 비고 |
|---|---:|---|
| 기존 (호환 모드와 짝지은 측정) | 4.396 | full extraction |
| 새 기본 설정, batch=1 | 3.365 | 약 1.31배, 23.5% 시간 감소 |
| 기존 감정 token 재사용 시 artifact 재추출 | 0.397 | token 읽기·저장 비용 별도 |
| 새 샘플 atomic 저장 | 0.034 | token export 및 최종 병합 비용 별도 |
| 새 완료 샘플 재개 읽기 | 0.014 | checksum 포함 |
| batch=32 | 2.255 | 별도 paired baseline 3.971초 대비 1.76배; 수치 차이 있음 |

기본 설정 8/8 영상에서 video tensor, frame/time/valid mask, artifact embedding, emotion embedding 모두 기존과 정확히 일치했다. batch=32에서는 artifact는 같았으나 emotion embedding이 달랐다. 별도 진단에서 TF32를 끄면 single/batch의 최대 차이가 약 8.3e-7로 줄었지만, 기존 TF32 single embedding과는 달랐으므로 기본 precision 정책은 변경하지 않았다.

AASIST+emotion2vec 추출은 같은 8개에서 평균 0.355초/영상, 첫 샘플 제외 0.205초였다. 오디오 수치는 저장 비용을 제외한 추론/전처리 측정이며 첫 샘플 warmup이 포함된다. CPU gated head의 synthetic timing으로 30 epochs를 환산하면 video 약 51초, audio 약 32초의 순수 연산량이다. 실제 학습 완료 시간이나 정확도 측정이 아니며 파일 쓰기·metric 계산·실제 데이터 로딩 비용은 제외한다.

전체 20,997개와 당시 존재한 기존 token 4,922개에 평균 속도를 단순 적용하면 남은 전 과정이 약 17시간 규모다. 사용자가 보낸 장기 실행 로그의 더 느린 평균에 측정한 1.31배 개선율을 적용하면 약 27시간이다. 계획 범위는 대략 17~28시간으로 두되, 실제 전환 후 충분한 새 샘플의 속도로 다시 산출해야 한다. 소표본·동시 실행·길이 분포 차이 때문에 확정 ETA로 사용하면 안 된다.

관련 회귀 테스트 71개 통과. 중단 후 재개 시 완료 샘플 미재계산, 손상 샘플 재계산, 설정/입력 변경 거절, writer lock 해제, 기존 학습/OR 평가 동작을 검사했다.

측정 원자료:
- `reports/gated_optimized_benchmark_20260928_compatible/results.json`
- `reports/gated_optimized_benchmark_20260928/results.json`

재측정(새 output 필요):

```powershell
./scripts/run.ps1 scripts/artifact_emotion_gated_attetnion_optimized/benchmark.py --output reports/gated_speed_check_v2 --face-batch-size 1
```

실제 CUDA smoke test도 통과했다. 첫 실행에서 기존 감정 토큰으로 1개 샘플 저장 후 중단하고, --resume 실행에서 첫 샘플을 재사용하고 다음 1개만 추가 저장했다. 근거: reports/gated_optimized_resume_smoke_20260928/SMOKE_RESULT.json.
