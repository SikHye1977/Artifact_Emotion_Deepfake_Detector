# FakeAVCeleb X3A — 최소 실행 저장소

모델/loader/통합 학습·최종 평가와 필요한 공통 코드, 공유 설정만 포함한다.
데이터/기존 실험 결과/가중치/외부 저장소/테스트 이력은 별도다.

## 1. 환경 준비

Linux, Python3.10, NVIDIA GPU, git, ffmpeg/ffprobe를 준비한다.
검증된 Python 패키지 버전은 requirements.txt에 기록했다.
CUDA12.1 wheel을 지원하는 NVIDIA 드라이버가 필요하다.

    python3.10 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    .venv/bin/python -m pip install -r requirements.txt

## 2. 공식 AASIST 및 데이터 준비

    git clone https://github.com/clovaai/aasist third_party/aasist
    git -C third_party/aasist checkout a04c9863f63d44471dde8a6abcb3b082b07cd1d1
    cp configs/local_paths.example.yaml configs/local_paths.yaml

local_paths.yaml의 fav_root를 외부 FakeAVCeleb_v1.2 폴더로 수정한다.
공식 AASIST requirements를 일괄 재설치하지 않는다.
X3D Kinetics weight는 최초 실행 때 다운로드되어 checkpoints/pretrained에 저장된다.

**기존 고정 데이터 분할/eligibility 파일도 원본 데이터와 함께 제공받아야 한다.**
현재 워크스테이션의 archives/fav_runtime_metadata.tar.gz를 별도로 전달해
새 머신 프로젝트 루트에서 푼다:

    tar -xzf /path/to/fav_runtime_metadata.tar.gz

묶음 내용:

- reports/fixed_splits_v1_release/FAV/split_manifest.jsonl
- reports/phase0_v1/eligibility_v4.jsonl
- reports/phase1_preflight/input_recipe.json

metadata는 대용량이며 로컬 경로가 포함되어 Git에서 제외한다.
Git의 configs JSON은 기존 SHA256 계약을 유지한다. 새 split 생성/검증 우회 금지.
미디어 복사는 파일 크기와 mtime_ns를 보존해야 한다. Loader가 둘 다 검사한다.
다른 배포본/수정시각이면 eligibility를 별도 감사해야 하며 검사만 끄면 안 된다.

    CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/reproduce_x3a/check_setup.py

이 검사는 CPU import/metadata/vendor hash와 경로/도구 확인이다.
전체 데이터 decoding이나 CUDA smoke/새 머신 설치 성공을 보장하는 테스트는 아니다.

## 3. 새 실험

4 GPU(물리0~3), 새 run ID:

    env CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=4 --redirects 3 --tee 3 --log-dir reports/launch_logs/fav_new_001 scripts/reproduce_x3a/train_x3a.py --devices 4 --model all --run-id fav_new_001

X3D20epoch → AASIST20epoch → dev OR. Test는 자동 실행하지 않는다.
X3D batch1/rank accumulation2 AMP, AASIST batch4/rank accumulation1 FP32.
진행상황/오류는 터미널 출력. GPU 설정은 코드가 변경하지 않는다.
현재 recipe thermal은 logging-only이며 하드웨어 보호는 변경하지 않는다.
GPU 수나 메모리가 다르면 실행 recipe 호환성을 먼저 확인한다.

1-GPU 학습 경로는 torchrun 없이 다음과 같이 실행한다(별도 single_gpu YAML):

    env CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/reproduce_x3a/train_x3a.py --devices 1 --model all --run-id fav_single_new_001

동일 코드/config로 생성한 중단 run은 같은 명령에 --resume 추가.
과거 완료 checkpoint는 이전 source hash로 고정되어 이 checkout과 resume 호환을 뜻하지 않는다.

## 4. 최종 test (4-GPU 학습 run 전용)

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/reproduce_x3a/evaluate_x3a_final.py --run-id fav_new_001 --execute-test

먼저 dev 기준 threshold와 checkpoint를 동결하고, GPU1장 batch1로 순차 평가한다.
재실행은 차단된다. 실패 시 자동 재시도하지 않고 기록을 조사한다.
결과는 reports/experiments/x3a_reproduction/fav_indomain_v1/unified_4gpu/fav_new_001/ 아래다.
논문용 subset/연구용 전체 modality 결과를 구분한다.
이 구현은 paper-spec reimplementation이며 논문의 정확한 분할/미공개 세부조건과 다르다.
