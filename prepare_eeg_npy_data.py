import os
import numpy as np
import glob
from tqdm import tqdm

# --- Configuration ---
WORKSPACE_ROOT = "D:/github_practice/github_desktop/ESC_arima/" # 이전 스크립트와 동일하게 설정
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")
SEIZURE_DIR = os.path.join(DATA_DIR, "seizure")
NON_SEIZURE_DIR = os.path.join(DATA_DIR, "non_seizure")

OUTPUT_DATA_NPY_FILE = os.path.join(WORKSPACE_ROOT, "all_eeg_data.npy")
OUTPUT_LABELS_NPY_FILE = os.path.join(WORKSPACE_ROOT, "all_eeg_labels.npy")

N_CHANNELS = 23
INPUT_LENGTH = 10240

def load_single_csv_data(file_path):
    """
    단일 CSV 파일을 읽어 (N_CHANNELS, INPUT_LENGTH) 형태의 NumPy 배열로 변환합니다.
    CSV 파일은 23개의 열(채널)과 10240개의 행(시계열)으로 구성되며 헤더가 없다고 가정합니다.
    """
    try:
        # CSV는 (10240, 23) 형태로 로드됨 (행, 열)
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        
        # 기대하는 형태: (INPUT_LENGTH, N_CHANNELS)
        if data.shape == (INPUT_LENGTH, N_CHANNELS):
            # (채널, 시계열 길이) 형태로 전치: (23, 10240)
            return data.T 
        else:
            print(f"Warning: File {file_path} has unexpected shape {data.shape}. Expected ({INPUT_LENGTH}, {N_CHANNELS}). Skipping.")
            return None
    except Exception as e:
        print(f"Error loading or processing {file_path}: {e}")
        return None

def process_csv_directory(directory_path, file_pattern, label):
    """
    지정된 디렉토리에서 CSV 파일들을 읽어 데이터와 레이블 리스트를 반환합니다.
    """
    all_data_list = []
    all_labels_list = []
    
    file_paths = glob.glob(os.path.join(directory_path, file_pattern))
    if not file_paths:
        print(f"Warning: No files found matching pattern {file_pattern} in {directory_path}")
        return [], []

    print(f"Processing {len(file_paths)} files from {directory_path} with label {label}...")
    for file_path in tqdm(file_paths):
        sample_data = load_single_csv_data(file_path)
        if sample_data is not None:
            all_data_list.append(sample_data)
            all_labels_list.append(label)
            
    return all_data_list, all_labels_list

def main():
    # Seizure 데이터 처리 (레이블 1)
    seizure_data_list, seizure_labels_list = process_csv_directory(SEIZURE_DIR, "*_ict.csv", 1)
    
    # Non-Seizure 데이터 처리 (레이블 0)
    non_seizure_data_list, non_seizure_labels_list = process_csv_directory(NON_SEIZURE_DIR, "*_non.csv", 0)
    
    if not seizure_data_list and not non_seizure_data_list:
        print("Error: No data loaded from any directory. Exiting.")
        return

    # 모든 데이터와 레이블 합치기
    final_all_data = []
    if seizure_data_list:
        final_all_data.extend(seizure_data_list)
    if non_seizure_data_list:
        final_all_data.extend(non_seizure_data_list)
    
    final_all_labels = []
    if seizure_labels_list:
        final_all_labels.extend(seizure_labels_list)
    if non_seizure_labels_list:
        final_all_labels.extend(non_seizure_labels_list)

    if not final_all_data:
        print("Error: No valid data samples to save. Exiting.")
        return
        
    # NumPy 배열로 변환
    final_data_array = np.array(final_all_data, dtype=np.float32)
    final_labels_array = np.array(all_labels_list, dtype=np.int32) # 수정: all_labels_list -> final_all_labels
    final_labels_array = np.array(final_all_labels, dtype=np.int32)


    # 데이터 유효성 검사 (형태 확인)
    if final_data_array.ndim != 3 or final_data_array.shape[1] != N_CHANNELS or final_data_array.shape[2] != INPUT_LENGTH:
        print(f"Error: Final data array has unexpected shape: {final_data_array.shape}")
        print(f"Expected shape: (num_samples, {N_CHANNELS}, {INPUT_LENGTH})")
        print("Please check the CSV loading logic and individual file shapes.")
        return
    
    print(f"\nSuccessfully processed {final_data_array.shape[0]} samples.")
    print(f"Shape of combined data: {final_data_array.shape}")
    print(f"Shape of combined labels: {final_labels_array.shape}")

    # .npy 파일로 저장
    try:
        np.save(OUTPUT_DATA_NPY_FILE, final_data_array)
        print(f"All EEG data saved to: {OUTPUT_DATA_NPY_FILE}")
        
        np.save(OUTPUT_LABELS_NPY_FILE, final_labels_array)
        print(f"All EEG labels saved to: {OUTPUT_LABELS_NPY_FILE}")
    except Exception as e:
        print(f"Error saving .npy files: {e}")
        return

    print("\nData preparation script finished.")
    print(f"Next step: Update 'DUMMY_NPY_FILENAME' in 'eeg_conv_autoencoder_colab.py' to '{os.path.basename(OUTPUT_DATA_NPY_FILE)}'")
    print("The autoencoder itself will not use the labels, but they are saved for potential future use.")

if __name__ == '__main__':
    main() 