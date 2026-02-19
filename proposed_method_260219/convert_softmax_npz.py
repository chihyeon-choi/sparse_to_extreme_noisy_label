import numpy as np
import os

def convert_softmax_to_argmax(file_path):
    """
    개별 .npz 파일의 소프트맥스 분포를 argmax 인덱스로 변환하여 덮어쓰는 함수
    """
    try:
        # 데이터 로드
        data = np.load(file_path, allow_pickle=True)
        new_dict = {}

        # 'model1', 'model2' 등 저장된 키값 확인
        for model_key in ['model1', 'model2']:
            if model_key in data:
                model_data = data[model_key].item()
                new_model_data = {}
                for epoch, dist in model_data.items():
                    # 전체 분포에서 argmax를 취해 인덱스만 추출 (용량 절감)
                    # int16 타입을 사용하여 메모리 효율을 극대화합니다.
                    new_model_data[epoch] = np.argmax(dist, axis=1).astype(np.int16)
                new_dict[model_key] = new_model_data
        
        # 원본 파일에 덮어쓰기 (압축 저장)
        np.savez_compressed(file_path, **new_dict)
        # print(f"변환 및 덮어쓰기 완료: {file_path}")
        
    except Exception as e:
        print(f"파일 처리 중 오류 발생 ({file_path}): {e}")

def process_subfolders(parent_dir):
    """
    부모 폴더 내의 모든 하위 폴더를 탐색하여 대상 파일을 처리하는 함수
    """
    if not os.path.exists(parent_dir):
        print(f"오류: '{parent_dir}' 폴더가 존재하지 않습니다.")
        return

    # 부모 폴더 내의 모든 항목 확인
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        
        # 하위 항목이 폴더인 경우에만 진행
        if os.path.isdir(folder_path):
            target_file = os.path.join(folder_path, "softmax_distribution_all_epochs.npz")
            
            # 대상 파일이 해당 폴더 안에 존재하는지 확인
            if os.path.isfile(target_file):
                # print(f"처리 중: {folder_name}...")
                convert_softmax_to_argmax(target_file)
            else:
                # print(f"건너뜀: {folder_name} (파일 없음)")
                pass

# --- 사용 예시 ---
# 'checkpoint_noisy_label' 폴더가 부모 폴더인 경우
dir_list = ['checkpoint_noisy_label', 'checkpoint_noisy_label_adjusted_JSD', 'checkpoint_noisy_label_adjusted_JSD_lu_sigmoid', 'checkpoint_noisy_label_adjusted_lu', 'checkpoint_noisy_label_confidence_threshold', 'checkpoint_noisy_label_wo_lu']
for parent_directory in dir_list:
    # parent_directory = 'checkpoint_noisy_label_wo_lu' 
    process_subfolders(parent_directory)