# 원본 데이터 준비

이 프로젝트는 MIT–Stanford 배터리 수명 데이터 중 다음 두 파일만 처리합니다.

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat` — Batch1, 학습·검증
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat` — Batch2, 최종 평가

파일을 이 디렉터리의 `raw/`에 배치합니다.

```text
data/raw/
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
└── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
```

원본 파일은 크기가 수 GB이므로 Git에 포함하지 않습니다. 데이터는 논문의 공식 공개 위치인 [data.matr.io/1](https://data.matr.io/1) 또는 프로젝트에서 사용한 배포처에서 확보합니다.

현재 `config.py`에는 검증된 개별 파일 직접 다운로드 주소가 등록되어 있지 않습니다. 파일이 없으면 프로그램은 필요한 파일명을 안내하고 중단하며 다른 날짜의 배치를 대신 사용하지 않습니다.

원본을 다른 위치에 보관했다면 실행 전에 환경변수로 디렉터리를 지정할 수 있습니다.

```bash
export ESS_DATA_DIR=/path/to/battery-data
python main.py
```

Windows PowerShell에서는 다음과 같이 지정합니다.

```powershell
$env:ESS_DATA_DIR = "C:\path\to\battery-data"
python main.py
```

Batch3 파일을 `raw/`에 함께 보관해도 현재 코드는 읽지 않습니다. 추가 가변충전 파일도 처리 대상이 아닙니다.
