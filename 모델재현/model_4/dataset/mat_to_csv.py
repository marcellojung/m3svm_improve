import scipy.io
import pandas as pd

# .mat 파일 로드
mat_file_path = 'data.mat'  # .mat 파일 경로
mat_data = scipy.io.loadmat(mat_file_path)

# .mat 파일 내의 데이터 추출 (키 이름 확인 필요)
# 일반적으로 유용한 데이터는 딕셔너리의 특정 키에 저장됩니다.
# 'your_data_key'는 사용자의 .mat 파일 구조에 맞게 변경해야 합니다.
data_key = 'your_data_key'
if data_key in mat_data:
    data = mat_data[data_key]
    # 데이터프레임으로 변환
    df = pd.DataFrame(data)

    # CSV 파일로 저장
    csv_file_path = 'output.csv'  # 저장할 CSV 파일 경로
    df.to_csv(csv_file_path, index=False)
    print(f"{csv_file_path}로 변환 완료.")
else:
    print(f"'{data_key}' 키를 찾을 수 없습니다. .mat 파일 구조를 확인해주세요.")