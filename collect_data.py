import pandas as pd
import os
import unicodedata

input_dir = "./data"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

file_map = {
    "김포_국내": "gimpo_d",
    "김포_국제": "gimpo_i",
    "김해": "gimhae",
    "제주": "jeju",
    "청주": "cheongju",
}

target_sheets = [str(i) for i in range(14, 26)]

print("=== 데이터 합치기 시작 ===")

for fname in os.listdir(input_dir):
    if not fname.endswith(".xlsx"):
        continue

    # 유니코드 정규화 (한글 자모 분리 문제 방지)
    base_name = unicodedata.normalize("NFC", fname.replace(".xlsx", ""))

    airport_code = None
    for key, eng_name in file_map.items():
        if key in base_name:   # 포함 여부 확인
            airport_code = eng_name
            break

    if airport_code is None:
        print(f"[무시] 매칭되지 않는 파일: {fname}")
        continue

    print(f"\n▶ 처리 중: {fname}  →  {airport_code}.csv")

    dfs = []
    try:
        xls = pd.ExcelFile(os.path.join(input_dir, fname))
        for sheet in target_sheets:
            if sheet in xls.sheet_names:
                df = pd.read_excel(os.path.join(input_dir, fname), sheet_name=sheet)
                df["airport"] = airport_code
                dfs.append(df)
                print(f"  - 시트 {sheet} 합침 (행 수: {len(df)})")
    except Exception as e:
        print(f"[에러] {fname} 처리 중 문제 발생: {e}")
        continue

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        outpath = os.path.join(output_dir, f"{airport_code}.csv")
        final_df.to_csv(outpath, index=False, encoding="utf-8-sig")
        print(f"✅ 저장 완료: {outpath} (총 행 수: {len(final_df)})")

print("\n=== 모든 처리 완료 ===")
