# Scientific Reports Deepfake Research — Fresh Start

작성일: 2026-09-07  
연구 기간: 2026년 9월–2027년 1월  
목표: Artifact와 Emotion-derived evidence의 딥페이크 탐지 성능 및 데이터셋 간 일반화를 검증하고, Scientific Reports 제출용 원고와 재현 가능한 실험 자료를 준비한다.

## 1. 이 프로젝트의 출발점

이 프로젝트는 **새 폴더에서 처음부터 구축하는 독립적인 연구 프로젝트**다. 기존 README, 코드, 논문, 실험 결과는 참고 자료이며, 새 프로젝트의 구현 완료 또는 검증 완료를 뜻하지 않는다. 사용자 확인에 따르면 세 데이터셋은 workstation의 `/home/hdd1/sikhye/deepfake-research/datasets` 아래에 이미 적재되어 있다. 적재 상태는 확인된 전제로 사용하되, 새 코드·학습 환경·개별 데이터셋 하위 경로·무결성·모델 가중치·실험 결과는 workstation에서 검증해야 한다. X3A 재현 기준은 사용자가 제공한 SAC 2025 논문이며 아래 Phase 1에 확인한 조건을 정리했다.

연구자는 각 모델의 입력, 전처리, 학습 목적, 출력과 평가 방법을 다시 이해하면서 단계적으로 진행한다. 기존 결과를 재사용하여 단계를 완료 처리하지 않는다. 기존 구현을 참고하거나 일부 재사용하는 경우 출처, 변경점과 새 검증 결과를 기록한다. 기존 원본과 데이터는 보존한다.

전체 순서:

1. 기존 적재 데이터셋의 라벨·분할·무결성·로더 검증
2. X3D / AASIST artifact baseline 재현
3. HSEmotion / CRNN 감정 분류기 검증과 emotion trajectory 추출
4. Emotion-discontinuity hypothesis 검증
5. Emotion-aware deepfake detector 구성과 temporal ablation
6. Artifact / Emotion의 hierarchical fusion 재현
7. 3개 데이터셋의 3×3 in-domain / cross-dataset 평가
8. 실패 원인·상보성 분석과 논문 작성

## 2. 연구 질문과 결과 해석 원칙

핵심 질문은 **“감정에서 추출한 정보가 딥페이크 탐지에 어떤 정보를 제공하며, 그 정보가 다른 데이터셋에서도 유효한가?”**다. Fake가 반드시 더 불연속적이라는 결론을 전제하지 않는다.

| ID | 검증할 질문 | 필요한 증거 |
| --- | --- | --- |
| RQ1 | X3D와 AASIST는 각 모달리티의 조작을 탐지하고 다른 데이터셋으로 일반화하는가? | 독립 baseline과 3×3 평가 |
| H1 | Real과 Fake의 temporal emotion dynamics에 차이가 있는가? | 사전 지정한 trajectory 통계와 효과크기·신뢰구간 |
| H1a | Fake의 국소적 감정 불연속성이 Real보다 큰가? | 방향을 미리 정한 discontinuity 지표의 비교 |
| H1b | 관찰한 차이의 방향과 크기가 데이터셋 간에 일관되는가? | 데이터셋별 효과크기와 이질성 분석 |
| RQ2 | Emotion detector가 시간 순서에 담긴 정보를 활용하는가? | shuffle / reverse / temporal mean 비교 |
| RQ3 | Emotion evidence가 Artifact의 오류를 보완하며 fusion의 성능을 개선하는가? | 동일 샘플에서 branch별 오류와 fusion 개선·악화 분석 |

### 가설이 지지되지 않을 때

- 유의하지 않다는 이유만으로 “차이가 없다” 또는 “가설이 거짓이다”라고 단정하지 않는다. 표본 수, 효과크기와 신뢰구간을 함께 보고, 차이 부재를 주장하려면 실질적 동등성 범위와 검정 계획이 필요하다.
- 먼저 감정 분류기 품질, 전처리, 얼굴 검출 실패, 무음과 결측 처리 문제를 점검한다. 모델이 출력한 trajectory는 사람이 느끼는 감정 자체의 정답이 아니다.
- 불연속성 가설이 지지되지 않아도 representation의 탐지 성능 및 artifact와의 상보성은 별도로 검증한다.
- 일부 데이터셋·모달리티에서만 효과가 있으면 해당 범위로 주장을 제한한다.
- 순서 ablation에서 차이가 없으면 “emotion flow를 활용한다”는 주장을 보류하고, 감정 분포 또는 representation에 기반한 정보로 해석한다.
- 탐지 성능과 상보성도 확인되지 않으면 부정적 결과와 적용 한계를 보고한다. 성능 개선이나 논문 채택을 보장하지 않는다.
- 결과를 본 후 만든 새 특징·가설은 탐색적 분석으로 구분한다. 확인적 주장에는 개발에 사용하지 않은 평가 자료가 필요하다.

## 3. 2026년 9월–2027년 1월 마일스톤

아래는 연구 운영을 위한 목표 일정이다. 세 데이터셋은 이미 적재되어 있으나 장비 성능, 실제 읽기 권한과 감정 가중치 확보 상태는 아직 검증되지 않았으므로 9월 처리량 측정 후 계산 예산을 확정한다. **2027년 1월 말은 내부 제출 목표이며, 저널 또는 Collection의 공식 마감일을 뜻하지 않는다.** 실제 모집 상태와 투고 요건은 제출 준비 시 공식 페이지에서 확인한다.

| 기간 | 주 작업 | 월말 완료 기준·산출물 |
| --- | --- | --- |
| **2026년 9월** | 새 환경과 디렉터리 구성, 기존 적재본의 하위 경로·무결성·메타데이터 확인, 공통 manifest와 누수 검사, FakeAVCeleb 로더 완성, X3D/AASIST import·소규모 학습·첫 in-domain 평가 | 데이터 감사 보고서, 고정된 FAV 분할, 두 baseline 및 X3A OR 결합의 학습·평가 전체 경로와 첫 결과, 저장 공간·학습 시간 추정. 세 데이터셋의 접근 장애를 조기에 식별 |
| **2026년 10월** | X3A 재현 조건 대조 및 FAV baseline 확정, HSEmotion/CRNN 감정 분류 품질 확인, trajectory 추출, 가설 검증용 지표·교란요인 계획 고정 및 개발 자료 분석 | 재현 비교표, 감정 모델 카드, trajectory 품질 보고서, 가설의 중간 결과와 11월 진행 방향. 나머지 두 데이터셋의 로더·분할 확정 |
| **2026년 11월** | Emotion detector와 temporal ablation, 기존 hierarchical fusion 명세 확인 및 구현, source별 학습 시작, 개발 자료로 최종 설정 선정 | 단일 모델·branch·fusion 비교, temporal ablation, 최종 설정과 분석 계획 고정. 늦어도 11월 말에는 3개 source 학습 및 9개 평가 조합의 실행 가능성 확인 |
| **2026년 12월** | 고정 설정으로 3×3 최종 평가, 핵심 결과 3 seeds, 신뢰구간·하위집단·상보성 분석, 표·그림과 원고 작성 | **12월 20일 목표: 주요 실험 동결. 12월 31일 목표: 전체 원고 초안과 보충자료 초안** |
| **2027년 1월** | 공동저자 검토, 주장·통계·재현성 점검, 필요한 오류 수정과 재검증, 투고 양식·데이터/코드 공개 계획 정리 | **1월 15일 목표: 수정 원고. 1월 25일 목표: 제출 패키지 확정. 1월 말 내부 제출 목표** |

운영 원칙:

- 논문의 Methods와 실험 일지는 9월부터 작성한다. 최종 결과가 나올 때까지 문서화를 미루지 않는다.
- 대규모 작업 전 대표 샘플로 시간·메모리·저장량을 측정한다. GPU 수나 데이터셋 전체 학습 가능성을 가정하지 않는다.
- 개발 실험은 1 seed로 점검하고, 최종 핵심 비교는 고정된 3 seeds(초기 제안: 42, 123, 2026)로 수행한다. 모델 간 비교에는 같은 seed 집합을 사용한다.
- 일정이 밀리면 새로운 architecture 탐색과 부가 ablation부터 줄인다. 누수 검사와 기본 평가를 생략하지 않는다.
- 전체 데이터 사용이 어렵다면 source train에서 그룹 단위로 고정한 subset을 사용하고 크기·추출 기준을 명시한다. subset 결과를 전체 데이터 학습 결과로 표기하지 않는다.
- 데이터셋 접근 지연으로 3×3을 못 채우면 누락과 사유를 표시하고 일정·범위 변경을 기록한다. 미실행 결과를 완료 처리하지 않는다.
- 최종 test 결과를 확인한 뒤 개선 모델을 선택하지 않는다. test 관찰 후 바꾼 방법은 탐색적 후속 연구로 구분한다.

## 4. 단계별 작업과 완료 기준

### Phase 0 — 기존 적재 데이터셋 검증 및 Data Loader

대상: FakeAVCeleb(FAV), AV-Deepfake1M(AVDF1M), PolyGlotFake(PGF). 정확한 배포 버전과 라벨 정의는 실제 배포 문서 및 metadata로 확인한다. 기존 README의 경로·파일 수·라벨 매핑을 정답으로 복사하지 않는다.

- [x] 세 데이터셋이 `/home/hdd1/sikhye/deepfake-research/datasets` 아래에 적재되어 있음 — 사용자 확인, 2026-09-07
- [ ] 실제 하위 디렉터리, 읽기 권한, 배포 버전, 사용 조건과 원본 경로 기록
- [ ] 기존 파일을 먼저 재사용하고, 누락·손상이 확인된 경우에만 해당 자료의 보완 확보 검토
- [ ] 비디오·오디오·클립 라벨 제공 여부와 라벨 근거 확인
- [ ] 공식 split 유무와 identity / 원본 영상 / 파생 영상 관계 조사
- [ ] 공통 manifest 생성 및 missing / corrupt / no-audio / no-face 통계 산출
- [ ] 원본·파생 영상이 split 사이에 걸치지 않도록 그룹 분할
- [ ] 중복 및 데이터셋 간 겹침 검사; 확인할 수 없는 identity는 한계로 기록
- [ ] 모달리티별 decoding, sampling, transform, batch collate 구현
- [ ] 짧은 영상, 가변 FPS, 무음, 결측, padding, mask 처리 확인
- [ ] label·method·언어·길이·split별 개수와 대표 샘플 수동 점검

권장 manifest 필드:

| 필드 | 정의 |
| --- | --- |
| `sample_id`, `dataset`, `dataset_version` | 안정적인 샘플 식별자와 출처 |
| `video_path`, `audio_path` | 미디어 위치; 별도 오디오가 없으면 추출 규칙 기록 |
| `video_label`, `audio_label`, `clip_label` | 0=real, 1=fake, 미확인=null |
| `label_source` | 공식 metadata 필드 또는 검증한 매핑 규칙 |
| `split`, `group_id`, `source_id`, `identity_id` | 분할과 누수 검사 키; 알 수 없는 값은 명시 |
| `method`, `language`, `duration`, `fps`, `sample_rate` | 하위집단 분석 및 전처리 정보 |
| `has_audio`, `decode_ok`, `exclusion_reason` | 품질과 제외 이력 |
| `manipulated_intervals` | 제공되는 경우 모달리티별 조작 구간과 시간 단위 |

라벨 원칙:

- X3D와 video emotion detector는 `video_label`, AASIST와 audio emotion detector는 `audio_label`로 학습한다.
- branch 및 최종 fusion은 `clip_label`로 평가한다. 여기서 clip fake는 평가 정의상 어느 한 모달리티라도 조작된 경우다.
- 두 modality label이 있으면 OR로 clip label을 산출할 수 있다. 하나가 1이면 clip=1이지만, 하나가 0이고 다른 하나가 미확인이면 clip을 0으로 추정하지 않는다. 공식 clip label이 따로 있으면 출처와 일관성을 검증한다.
- clip label만으로 modality label을 역추정하지 않는다. 해당 modality 정답이 없으면 그 지도학습·평가에서 제외하거나 별도 프로토콜로 보고한다.
- 부분 조작 영상에서 무작위 crop에 조작 구간이 없을 수 있다. 구간 라벨이 있으면 활용하고, 없으면 clip supervision의 한계와 clip score 집계 방식을 명시한다.
- 공식 split과 연구용 group-disjoint split이 다르면 각각의 용도와 차이를 기록한다. 공식 split을 사용했다는 이유만으로 identity-disjoint라고 주장하지 않는다.

**완료 기준:** manifest 버전·해시, 분할 생성 조건, 라벨 매핑 근거, 누수 검사와 제외 통계가 저장되어 있고 실제 batch를 읽을 수 있다.

### Phase 1 — X3D / AASIST 및 X3A 재현

- [x] 사용자 첨부 X3A 논문의 방법·실험 조건 확인 — 아래 재현 명세 참조
- [ ] 저자 코드 확보 여부, 고정 revision, 미명시 학습 조건과 정확한 split 확인
- [ ] 원 논문의 modality별 baseline과 결합 모델 실험을 구분
- [ ] X3D의 프레임 수·간격·crop·정규화·사전학습·head·fine-tuning 범위 기록
- [ ] AASIST의 sample rate·입력 길이·crop/pad·가중치 출처·출력 class 순서 확인
- [ ] 실제 정답과 모델 config로 fake score 방향 검증; 특정 logit 인덱스를 무조건 fake로 가정하지 않기
- [ ] 소규모 데이터 overfit, 유한 loss, gradient, checkpoint 저장·재로드 검사
- [ ] FAV에서 두 모델 독립 학습과 modality별 in-domain 평가
- [ ] 동일 샘플의 두 점수를 probabilistic OR로 결합한 X3A clip-level baseline 평가
- [ ] 원 논문과 동일 조건 / 확인 불가 / 의도적 변경을 표로 기록

각 모델 문서에는 Input, Sampling, Transform, Backbone, Classifier, Loss, Optimizer, LR, Scheduler, Epoch, Batch size, Seed, Label, Checkpoint selection, Clip aggregation을 남긴다.

**완료 기준:** 고정 config로 독립 학습·평가를 다시 실행할 수 있고 sample별 점수와 재현 비교표가 있다. 단순 import 성공이나 유사 architecture 구현을 X3A 전체 재현 완료라고 부르지 않는다. 원문은 확인되었으나 저자 코드·정확한 split·미명시 설정이 확보되지 않으면 그 차이를 공개하고 “논문 명세 기반 재구현”으로 구분한다.

### Phase 1-A — 첨부 X3A 논문에서 확인한 재현 명세

참고문헌: Chan Park, Bohyun Moon, Minsun Jeon, Jee-weon Jung, Simon S. Woo. **X3A: Efficient Multimodal Deepfake Detection with Score-Level Fusion.** SAC ’25, pp. 767–774, 2025. DOI: `10.1145/3672608.3707934`.

원문은 사용자 첨부 `X3A.pdf`다. 아래 위치 표기는 논문 인쇄 페이지 기준이며, 첨부 PDF는 ACM 표지 때문에 인쇄 p.770이 PDF 5페이지에 해당한다. Workstation에도 원문을 참고 자료로 두고, 현재 Mac의 첨부 경로를 workstation 경로로 가정하지 않는다.

| 항목 | 논문에 명시된 내용 | 구현 및 재현 시 처리 |
| --- | --- | --- |
| 시각 입력 | 전체 영상에서 **128프레임 균등 샘플링**, §3.1, p.770 | 짧은 clip만 추출하는 일반 X3D 예제를 그대로 사용하지 않는다. 짧은 영상의 반복/패딩과 실제 index 규칙은 별도 명시 |
| 시각 모델 | X3D encoder → MLP → sigmoid, Eq.(4); BCE, Eq.(5) | X3D-M이라고 확정하지 않는다. variant·MLP 상세·사전학습·freeze 범위는 추가 확인 |
| 청각 입력 | FFmpeg로 raw waveform 추출, 조작 구간 누락을 피하기 위한 maximum padding, §3.2, p.770 | 일반적인 고정 길이 random crop으로 조용히 대체하지 않는다. padding 목표 길이·값·단위와 긴 입력 처리 확인 |
| 청각 모델 | AASIST, 2차원 출력 → softmax, Eq.(6); BCE 사용 서술 | fake class index, BCE 적용 tensor와 reduction 등은 구현에서 확인 |
| 결합 | `p_x3a = 1 - (1 - p_video) * (1 - p_audio)`, Eq.(7), p.771 | hard OR 또는 max와 구분. 추가 fusion 학습 없이 두 score를 결합하는 baseline부터 구성 |
| 데이터셋 | FakeAVCeleb, AV-Deepfake1M, **LAV-DF**, §4.1 | 이번 연구는 FAV·AVDF1M·**PGF**가 대상. PGF는 X3A 원문 재현이 아닌 확장 평가이며 LAV-DF 추가는 현재 필수가 아님 |
| 분할 | test에 unseen speaker; 공식 split이 없는 경우 custom split, §4.1 | 정확한 ID 목록·비율·seed는 추가 확인. 동일 목록이 없으면 speaker-disjoint 원칙 재현과 정확한 split 재현을 구분 |
| AVDF1M 분할 | 저자 실험 당시 train/validation을 합쳐 train/validation/test로 재분할, §4.1 | 논문의 당시 구성 설명이다. 적재 버전의 공식 label·split 상태를 먼저 조사하고 별도 manifest로 재현; 원본은 이동하지 않음 |
| 시각 평가 subset | **audio-only 조작 RVFA 제외**, §4.2.1 | RVRA + FVRA + FVFA에서 `video_label`로 평가 |
| 청각 평가 subset | **video-only 조작 FVRA 제외**, §4.2.1 | RVRA + RVFA + FVFA에서 `audio_label`로 평가 |
| 주 보고 지표 | AUC와 Accuracy, §4.2 | 원문 비교용 Acc.를 추가하고, 연구 주 평가의 AP·Balanced Accuracy·Macro-F1·MCC도 유지 |
| Cross-dataset | §4.3, p.773에서 향후 과제로 제시 | 이번 3×3 일반화 실험은 원문에 제시된 cross-dataset 수치의 재현이 아니라 확장 검증 |

#### 평가 프로토콜은 두 가지를 함께 보관

- **X3A 비교용:** 원문의 visual/audio subset을 각각 적용하고, X3A는 두 모달리티가 있는 전체 평가 집합에서 `clip_label`로 평가한다. 모델별 표본 수가 다르므로 단일 모델 표와 fusion 표의 수치를 동일 집합 비교로 해석하지 않는다.
- **이번 연구의 modality 평가:** 라벨이 확인된 전체 평가 집합에서 각 modality label을 사용한다. 예를 들어 RVFA의 video label은 real이다. 이 결과는 X3A subset 결과와 구별한다.
- 원문 §4.2.1은 위 subset을 **평가용**으로 설명한다. 이를 근거로 학습 집합도 동일하게 필터링했다고 단정하지 않는다. 정확한 학습 subset은 저자 구현에서 확인하고, 확보 전에는 선택한 방식을 명시한다.
- Calibration을 추가하거나 dev에서 다른 threshold를 선택한 변형은 raw X3A OR 결과와 별도 기록한다. 원문의 Accuracy threshold가 불명확하면 정확한 수치 재현을 보장하지 않는다.

#### 원문 결과: 비교 기준이며 목표 성능을 강제하는 값이 아님

| 데이터셋 | X3D AUC / Acc. (Table 1) | AASIST AUC / Acc. (Table 2) | X3A AUC / Acc. (Table 3) |
| --- | --- | --- | --- |
| FakeAVCeleb | 0.9970 / 0.9878 | 1.0000 / 0.9992 | 0.9968 / 0.9890 |
| AV-Deepfake1M | 0.9999 / 0.9911 | 1.0000 / 0.9952 | 0.9999 / 0.9960 |
| LAV-DF (참고) | 0.9999 / 0.9969 | 1.0000 / 0.9998 | 0.9999 / 0.9976 |

출처: Tables 1–3, pp.771–772. 다른 split·버전·제외 기준에서 얻은 새 결과와 단순 수치 일치 여부만으로 재현 성공을 판정하지 않는다.

#### 원문만으로 확정할 수 없는 설정과 주의점

1. 첨부 본문에는 X3D variant, MLP 세부 구조, 입력 해상도·정규화·augmentation, pretrained checkpoint, optimizer, learning rate, batch size, epoch, scheduler, seed, 정확한 split 목록 등이 충분히 명시되어 있지 않다. AASIST sample rate·padding 길이/방식·학습 가중치 출처도 확인이 필요하다. 일반 라이브러리 기본값이나 기존 README 값을 논문 설정으로 표기하지 않는다.
2. Eq.(2)는 `floor(i * N_frames / M) + 1`, `i=1,...,M`으로 인쇄되어 있어 마지막 index가 `N_frames+1`이 되는 경계 문제가 있다. 식을 그대로 실행하지 말고 저자 코드를 확인한다. 코드가 없으면 유효 범위 내 endpoint-inclusive uniform sampling 등 선택한 규칙과 원문 차이를 기록한다.
3. Maximum padding이 dataset 전체 기준인지 batch 기준인지, zero/repeat padding인지 명확하지 않다. 조작 구간을 보존한다는 취지를 유지하면서 구체적인 방식을 고정하고, test 통계로 학습 설정을 결정하지 않는다. 128프레임도 짧은 조작 구간을 반드시 포함한다고 보장하지 않는다.
4. 논문은 probabilistic OR의 false-positive 감소를 서술하지만, 식 자체는 동일 입력·동일 threshold에서 `p_x3a >= max(p_video, p_audio)`이다. 따라서 같은 threshold의 max보다 false positive가 자동으로 줄어든다고 설명하지 않는다. 보정된 결합 확률이라는 해석도 별도 가정·검증이 필요하다.
5. X3A의 두 모델 OR 재현은 **Phase 1**에서 완료한다. 기존 Hierarchical 논문의 Artifact/Emotion 2단계 결합은 **Phase 5**의 별도 재현 과제다. 모든 단계가 단순 probabilistic OR뿐이면 `1 - product(1-p_i)`로 평탄화되므로, 계층화 자체가 새로운 표현력을 준다는 주장은 할 수 없다. 기존 원고의 calibration·가중치·중간 변환 유무를 먼저 확인한다.

### Phase 2 — 감정 분류기와 Emotion Trajectory

- [ ] HSEmotion variant와 가중치 출처, CRNN의 구체적 architecture와 감정 학습 데이터 확정
- [ ] CRNN은 architecture 계열명이므로 단일 표준 모델로 취급하지 않기
- [ ] 감정 사전학습 가중치가 없으면 별도 감정 데이터의 speaker/identity 분할로 학습
- [ ] 감정 정답이 있는 held-out 자료에서 Macro-F1·혼동행렬 등 감정 분류 성능 확인
- [ ] 연구 영상의 대표 샘플에서 얼굴 추적, 무음, 언어 차이, confidence와 trajectory 품질 점검
- [ ] frozen emotion backbone에서 timestamp, probability, embedding, valid mask 추출
- [ ] 오디오 window 길이·hop과 비디오 sampling 간격을 초 단위로 명시
- [ ] cache에 입력 manifest·전처리·checkpoint 해시를 연결

감정 정답이 없는 딥페이크 데이터에서 감정 분류 정확도를 계산했다고 주장하지 않는다. 시각·청각의 감정 클래스 정의가 다르면 임의로 같은 인덱스끼리 비교하지 않는다. 초기 가설 검증은 frozen backbone으로 수행하고, deepfake label fine-tuning은 별도 탐지 실험으로 구분한다.

**완료 기준:** 감정 모델 품질과 한계가 문서화되어 있고, 실제 시간축과 유효 mask를 포함한 trajectory를 재추출할 수 있다.

### Phase 3 — Emotion-Discontinuity Hypothesis

초기 주 지표 제안은 일정한 시간 간격에서 인접 감정 확률 벡터의 L1 변화량 평균이다. 비디오·오디오별 시간 간격을 고정하고 동일 모달리티 안에서 비교한다.

```text
D = mean(||p(t) - p(t-1)||_1), 유효한 인접 쌍만 사용
```

결측 구간을 건너뛴 두 관측치를 정상적인 인접 쌍으로 연결하지 않는다. 유효 쌍이 없는 샘플은 0점 대신 계산 불가로 처리한다. Audio window 중첩에 따른 smoothing 효과도 기록한다.

- [ ] 주 지표, 방향 가설, 최소 유효 길이, 제외 기준과 분석 단위를 개발 단계에서 고정
- [ ] mean/median·분포 그림, 효과크기와 95% CI 산출
- [ ] 같은 원본의 real/fake pair가 있으면 paired 비교; 반복 identity/원본은 그룹 의존성을 반영
- [ ] 길이, 언어, 화자, 조작 방법, 얼굴·오디오 품질을 층화 또는 조정
- [ ] 여러 지표·하위집단 검정의 다중 비교 처리와 탐색적 분석 여부 기록
- [ ] D 단독 ROC-AUC로 판별 가능성 확인; target 결과로 점수 방향을 뒤집지 않기

프레임 또는 겹치는 window를 각각 독립 표본으로 세어 유의성을 과장하지 않는다. 이질적인 데이터셋을 합친 유의성 하나로 보편성을 주장하지 않는다.

**완료 기준:** 효과의 크기·불확실성·범위에 근거한 가설 판정 보고서가 있다. 가설 지지는 다음 단계의 필수 통과 조건이 아니지만, 특징을 측정할 수 있을 만큼의 trajectory 품질은 필요하다.

### Phase 4 — Emotion-aware Deepfake Detectors

- [ ] 간단한 discontinuity/statistics classifier와 temporal mean representation baseline
- [ ] trajectory 입력 video/audio detector 구현; 복잡한 구조는 baseline 확인 후 추가
- [ ] frozen backbone을 기본으로 시작하고 fine-tuning 실험은 별도 표기
- [ ] 정상 순서, random shuffle, reverse, temporal mean 비교
- [ ] 파라미터 수·학습 예산·sampling 차이와 복수 shuffle 반복 기록
- [ ] modality 성능과 clip-level 평가를 구분하여 보고

**추론 시 순서 교란**은 정상 학습 모델의 순서 민감도를 보고, **변형 조건으로 재학습**은 그 정보 없이도 학습 가능한지 본다. 두 종류를 구분한다. Shuffle은 유효 구간 안에서만 수행하고 padding mask는 유지한다.

Reverse는 인접 차이의 평균처럼 순서 반전에 불변인 지표를 바꾸지 않는다. 따라서 reverse 결과 하나로 시간 정보 사용 여부를 결론 내리지 않는다. Embedding 기반 성능만으로 감정 의미를 사용한다고 단정하지 않는다.

**완료 기준:** detector의 성능과 시간 구조 의존성을 따로 설명할 수 있는 비교표가 있다.

### Phase 5 — Hierarchical Fusion

```text
Video → X3D                              → video artifact score ┐
Audio → AASIST                           → audio artifact score ┴→ Artifact branch ┐
Video → HSEmotion → trajectory → detector → video emotion score  ┐                  ├→ Final clip score
Audio → CRNN     → trajectory → detector → audio emotion score  ┴→ Emotion branch  ┘
```

- [ ] 기존 Hierarchical 논문의 식, score 방향, 정규화, branch/final 결합 규칙 확인
- [ ] 명세가 확인된 원 구조를 먼저 재현; 임의의 평균/max/OR를 동일 구조라고 부르지 않기
- [ ] 단일 모델, artifact-only, emotion-only, hierarchical fusion과 단순 결합 baseline 비교
- [ ] trainable fusion은 base-model 학습에 쓰지 않은 source fusion-fit 자료 또는 out-of-fold prediction으로 학습
- [ ] source dev에서만 calibration·threshold·hyperparameter 선택
- [ ] 결측 모달리티 처리 규칙을 고정하고 완전 관측 집합과 별도 보고

**완료 기준:** 기존 구조와 구현의 대응표, fusion 입력·출력 정의와 비교 결과가 있다. 가중 평균이나 max 출력은 자동으로 보정된 확률이 되는 것이 아니므로 검증 전에는 fake score로 표기한다.

### Phase 6 — 3×3 In-domain / Cross-dataset Evaluation

행은 학습 source, 열은 test target이다. 대각선은 in-domain 3개, 나머지는 방향이 서로 다른 cross-dataset 6개다.

| Train ↓ / Test → | FAV | AVDF1M | PGF |
| --- | --- | --- | --- |
| FAV | FAV → FAV | FAV → AVDF1M | FAV → PGF |
| AVDF1M | AVDF1M → FAV | AVDF1M → AVDF1M | AVDF1M → PGF |
| PGF | PGF → FAV | PGF → AVDF1M | PGF → PGF |

한 source/seed에서 학습·선정한 동일 checkpoint를 세 target test에 적용한다. 각 셀마다 target에 맞춰 새로 학습하는 것이 아니다. 세 데이터셋 모두 학습 source 역할을 하므로 개발자가 다른 실험에서 본 target 정보가 설계에 유입되지 않도록, 최종 test는 설정 동결 후 일괄 평가한다.

- [ ] 모든 source에서 train / fusion-fit(필요시) / dev / test 역할과 그룹 분리를 명확히 정의
- [ ] target test로 모델, threshold, score 방향, feature 또는 epoch를 선택하지 않기
- [ ] 최종 test에서 확인할 가설·통계·하위집단 분석을 미리 고정
- [ ] 모델 비교에 같은 평가 sample 집합 사용; 추가 제외는 수와 사유 보고
- [ ] ROC-AUC를 주 지표로, AP, Balanced Accuracy, Macro-F1, MCC와 혼동행렬 함께 저장
- [ ] threshold는 source dev에서 사전 지정 기준으로 선택; target에서는 고정
- [ ] AP와 trapezoidal PR-AUC를 혼용하지 않고 계산 정의·클래스 비율 명시
- [ ] 한 클래스만 있는 하위집단의 AUC는 계산 불가로 표기
- [ ] always-real / always-fake 및 단순 결합 baseline 포함
- [ ] 핵심 비교 3 seeds mean ± SD와 그룹 bootstrap 95% CI 보고; 두 불확실성의 의미를 구분
- [ ] 모델 간 성능 차이는 같은 sample/group을 함께 재표집하는 paired 비교 사용

주 3×3 표는 `clip_label` 기준 최종 탐지 성능으로 통일한다. 각 단일 modality 모델은 해당 modality label 평가도 별도 표에 제시한다. 시각 모델이 정상 영상·가짜 오디오 clip을 놓치는 것은 시각 조작 탐지 오류와 구별한다.

**완료 기준:** 9개 조합의 고정 manifest, checkpoint/config, sample별 prediction, 집계 표가 서로 추적 가능하다. 미실행·평가 불가 셀은 명확히 표시한다.

### Phase 7 — 결과 분석과 원고

- [ ] In-domain / cross-dataset에서 Artifact와 Emotion의 성공·실패 비교
- [ ] real/fake, modality 조작 조합, method, language, 길이·품질별 분석
- [ ] Artifact 오류를 fusion이 고친 비율과 Artifact 정답을 fusion이 틀리게 만든 비율을 함께 계산
- [ ] 낮은 AUC의 score 방향 반전 분석은 사후 진단으로 표시; 반전 점수를 주 결과로 대체하지 않기
- [ ] cherry-picking을 피하도록 실패 사례 선정 규칙 기록
- [ ] Methods, Results, Discussion, 한계, 표·그림, 보충자료 작성
- [ ] 관련 기존 논문과의 중복·차별점, 데이터/코드/가중치 공개 가능 범위 정리
- [ ] 제출 시점의 공식 저널 지침에 맞춰 저자 정보와 각종 선언·제출 서류 확인

**완료 기준:** 주장의 범위가 실제 실험과 일치하고, 성공·부정적 결과 모두 원고와 재현 자료에 반영되어 있다.