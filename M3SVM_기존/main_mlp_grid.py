import numpy as np
import argparse
from train_chg_mlp_grid import *
import itertools

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    parser = argparse.ArgumentParser(
        description='Hyperparameter search',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.000)
    parser.add_argument('--state', type=int, default=42)
    parser.add_argument('--If_scale', default=True)
    parser.add_argument('--scale_type', type=str, default='standard', 
                        choices=['standard', 'minmax'], help='Choose the type of scaler: standard or minmax')
    para = parser.parse_args()
    para.test_size = 0.2

    datasets = ['Cornell', 'ISOLET', 'HHAR', 'USPS', 'ORL', 'Dermatology', 'Vehicle', 'Glass']

    # Hyperparameter ranges for grid search
    lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
    lam_list = [0.0001, 0.001, 0.005, 0.01, 0.05]
    p_list = [2, 3, 4, 6]

    def grid_search(para, dataset, lr_list, lam_list, p_list):
        best_acc = 0
        best_params = {}

        for lr, lam, p in itertools.product(lr_list, lam_list, p_list):
            para.lr = lr
            para.lam = lam
            para.p = p
            para.data = dataset

            print(f"Testing parameters: lr={lr}, lam={lam}, p={p}")
            acc = R_MLR(para)  # R_MLR 함수가 최종 테스트 정확도를 반환하도록 설정

            if acc > best_acc:
                best_acc = acc
                best_params = {'lr': lr, 'lam': lam, 'p': p}

        print(f"Best parameters for {dataset}: {best_params} with accuracy {best_acc:.4f}")
        return best_params

    # 각 데이터셋에 대해 Grid Search 수행
    for dataset in datasets:
        print(f"Running Grid Search for dataset: {dataset}")
        best_params = grid_search(para, dataset, lr_list, lam_list, p_list)