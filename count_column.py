import pandas as pd

# 1. 기본 읽기
df = pd.read_csv('all_tickers_historical_features.csv')
print(f"기본: {len(df.columns)}개")
print(f"마지막 5개: {list(df.columns)[-5:]}")

# 2. BOM 제거
df_bom = pd.read_csv('all_tickers_historical_features.csv', encoding='utf-8-sig')
print(f"BOM 제거: {len(df_bom.columns)}개")

# 3. 숫자형만 확인
numeric_cols = df.select_dtypes(include=['number']).columns
print(f"숫자형: {len(numeric_cols)}개")

# 4. 모든 컬럼 출력
print("전체 컬럼:")
for i, col in enumerate(df.columns):
    print(f"{i+1:2d}: '{col}'")