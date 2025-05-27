import os
import shutil
import glob
import re

# --- Configuration ---
# 필터링할 파일 번호 리스트 (사용자 제공)
FILE_NUMBERS_TO_KEEP = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    48, 49, 50, 51, 52, 53, 54,
    173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194
]

# 소스 디렉토리 (워크스페이스 루트 기준)
NON_SEIZURE_SRC_DIR = "CHB-MIT Dataset/Patient Non specific/Non-Seizure Patterns/Non-Seizure(Ictal)/"
SEIZURE_SRC_DIR = "CHB-MIT Dataset/Patient Non specific/Seizure Patterns/Ictal/"

# 기본 대상 디렉토리 이름 (루트 아래에 생성될 폴더)
BASE_DEST_DIR_NAME = "data"
WORKSPACE_ROOT = "D:/github_practice/github_desktop/ESC_arima/" 

# 최종 저장될 기본 디렉토리 경로
BASE_DEST_DIR = os.path.join(WORKSPACE_ROOT, BASE_DEST_DIR_NAME)

# 하위 디렉토리 이름
NON_SEIZURE_SUBDIR = "non_seizure"
SEIZURE_SUBDIR = "seizure"

def get_file_number(filename):
    """파일명에서 숫자 부분을 추출합니다. 예: '99.csv' -> 99"""
    match = re.match(r"(\d+)\.csv", filename)
    if match:
        return int(match.group(1))
    return None

def process_files(src_dir_relative, specific_dest_dir_abs, suffix, file_numbers_to_keep):
    """
    소스 디렉토리에서 파일을 필터링, 복사 및 이름 변경합니다.
    src_dir_relative: 워크스페이스 루트 기준 소스 디렉토리 상대 경로
    specific_dest_dir_abs: 대상 디렉토리 절대 경로
    suffix: 파일명에 추가할 접미사 (예: '_non', '_ict')
    file_numbers_to_keep: 유지할 파일 번호 리스트
    """
    src_dir_abs = os.path.join(WORKSPACE_ROOT, src_dir_relative)
    if not os.path.isdir(src_dir_abs):
        print(f"오류: 소스 디렉토리를 찾을 수 없습니다: {src_dir_abs}")
        return 0

    if not os.path.exists(specific_dest_dir_abs):
        try:
            os.makedirs(specific_dest_dir_abs)
            print(f"하위 디렉토리가 생성되었습니다: {specific_dest_dir_abs}")
        except Exception as e:
            print(f"하위 디렉토리 생성 중 오류 발생 {specific_dest_dir_abs}: {e}")
            return 0
    else:
        print(f"하위 디렉토리가 이미 존재합니다: {specific_dest_dir_abs}")

    count = 0
    for filename in os.listdir(src_dir_abs):
        if filename.endswith(".csv"):
            file_number = get_file_number(filename)
            if file_number is not None and file_number in file_numbers_to_keep:
                original_file_path = os.path.join(src_dir_abs, filename)
                new_filename = f"{file_number}{suffix}.csv"
                dest_file_path = os.path.join(specific_dest_dir_abs, new_filename)
                
                try:
                    shutil.copy2(original_file_path, dest_file_path)
                    count += 1
                except Exception as e:
                    print(f"파일 복사 중 오류 발생 {original_file_path} -> {dest_file_path}: {e}")
    return count

def main():
    print(f"기본 대상 디렉토리: {BASE_DEST_DIR}")
    if not os.path.exists(BASE_DEST_DIR):
        try:
            os.makedirs(BASE_DEST_DIR)
            print(f"기본 디렉토리가 생성되었습니다: {BASE_DEST_DIR}")
        except Exception as e:
            print(f"기본 디렉토리 생성 중 오류 발생 {BASE_DEST_DIR}: {e}")
            return
    else:
        print(f"기본 디렉토리가 이미 존재합니다: {BASE_DEST_DIR}")

    # Non-Seizure 파일 저장 경로 설정 및 처리
    dest_dir_non_seizure = os.path.join(BASE_DEST_DIR, NON_SEIZURE_SUBDIR)
    print(f"\nNon-Seizure 파일 처리 시작 (소스: {NON_SEIZURE_SRC_DIR}, 대상: {dest_dir_non_seizure})...")
    non_seizure_processed_count = process_files(NON_SEIZURE_SRC_DIR, dest_dir_non_seizure, "_non", FILE_NUMBERS_TO_KEEP)
    print(f"Non-Seizure 파일 처리 완료. {non_seizure_processed_count}개의 파일이 {dest_dir_non_seizure}에 복사되었습니다.")

    # Seizure 파일 저장 경로 설정 및 처리
    dest_dir_seizure = os.path.join(BASE_DEST_DIR, SEIZURE_SUBDIR)
    print(f"\nSeizure 파일 처리 시작 (소스: {SEIZURE_SRC_DIR}, 대상: {dest_dir_seizure})...")
    seizure_processed_count = process_files(SEIZURE_SRC_DIR, dest_dir_seizure, "_ict", FILE_NUMBERS_TO_KEEP)
    print(f"Seizure 파일 처리 완료. {seizure_processed_count}개의 파일이 {dest_dir_seizure}에 복사되었습니다.")

    total_copied = non_seizure_processed_count + seizure_processed_count
    print(f"\n총 {total_copied}개의 파일이 {BASE_DEST_DIR} 내의 각 하위 디렉토리로 복사 및 이름 변경되었습니다.")
    print("스크립트 실행이 완료되었습니다.")

if __name__ == "__main__":
    main() 