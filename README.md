# Timeseries_Synthesis_DP

## Introduction
- 본 코드는 데이터 합성을 위한 코드로서, 특히 로컬-차분프라이버시 모델을 활용하여 시계열 데이터 합성을 하게 하기 위한 코드입니다.

## How to use
### prerequisite
- git clone 및 가상환경 설정
```
# git clone
git clone https://github.com/gememy85/DP_synthesize.git

# 가상환경 설정
python -m venv .env
```
- 먼저 필요한 모듈 설치를 해준다.
```
pip intsll -r requirements.txt
```

### settings
- timeseries_dp.py의 args를 수정한다.
```
args = Namespace(
    original_file = "원본 파일" : csv 파일,
    output_file = "생산된 파일의 경로 + 이름" : csv 파일,
    epsilon = Privacy Budget인 epsilon을 정의해줘야 함
)

```

### 실행
- 셋팅을 끝낸 후 저장한 후, terminal에서 실행시켜준다.
```
python timeseries_dp.py
```
