import json
import json
from datetime import datetime
import torch.nn as nn
import pickle
import numpy as np

from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer

# cognite.correlation 대체 구현 사용
from correlation_alternative import columns_by_max_cross_correlation, plot_cross_correlations, cognite_mock as cognite

from datetime import datetime, timedelta
import seaborn as sns
import time

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

########### warning hide ##############
import warnings
warnings.filterwarnings(action='ignore')
#######################################

start = time.time()
id = datetime.now().strftime("%d%m%Y_%H%M%S")
torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)

    if dataset == "SMD":
        # SMD 처리
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset == "SMAP":
        # SMAP 처리
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    elif dataset == "MSL":
        # MSL 처리
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    elif dataset == "WADI":
        # WADI 처리
        output_path = f'output/WADI'
        (x_train, _), (x_test, y_test) = get_data('WADI', normalize=normalize)
        
        # Check for NaN/Inf in data and clean if necessary
        if np.isnan(x_train).any() or np.isinf(x_train).any():
            print("WARNING: NaN or Inf values found in training data! Cleaning...")
            x_train = np.nan_to_num(x_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.isnan(x_test).any() or np.isinf(x_test).any():
            print("WARNING: NaN or Inf values found in test data! Cleaning...")
            x_test = np.nan_to_num(x_test, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"
    os.makedirs(save_path, exist_ok=True)

    ############################### TLCC 산출 및 상관관계를 반영한 x_train 생성 ####################################

    one_count = []
    for i in y_test:
        if i == 1:
            one_count.append(i)

    
    data_df = pd.DataFrame(x_train)
    
    # TLCC threshold 인자 변수 정의
    tlcc_threshold = getattr(args, 'tlcc_threshold', None)

    # 진짜 TLCC 사용 여부 확인
    use_true_tlcc = getattr(args, 'use_true_tlcc', False)
    
    if use_true_tlcc:
        print("🔥 Using TRUE Time-Lagged Cross-Correlation (TLCC)")
        from true_tlcc_implementation import columns_by_max_cross_correlation_tlcc
        
        # 데이터셋별 캐싱을 위한 파라미터 전달
        corr_adj_df = columns_by_max_cross_correlation_tlcc(
            data_df, 
            max_lag=10, 
            dataset_name=dataset,
            output_dir=output_path  # 실험 결과와 같은 폴더에 저장
        )
        corr_adj_np = corr_adj_df.values
    else:
        print("📊 Using simple correlation (not true TLCC)")
        corr_adj_df = pd.DataFrame()
        for i in range(data_df.shape[1]):
            # 대체 구현 사용
            from correlation_alternative import cross_correlate
            corr = cross_correlate(data_df, data_df.iloc[:, i], lag_idx=1)
            corr_adj_df[i] = corr

        corr_adj_df = corr_adj_df.fillna(0)
        corr_adj_np = corr_adj_df.values
    
    corr_adj_np = np.nan_to_num(corr_adj_np)
    
    # TLCC threshold 및 binary 변환 옵션
    tlcc_binary = getattr(args, 'tlcc_binary', False)
    
    # threshold 적용: tlcc_threshold가 None이 아닐 때만 적용, None이면 모든 상관계수 유지
    if tlcc_threshold is not None:
        if tlcc_binary:
            # threshold 넘으면 1로 변환, 안 넘으면 0으로 변환
            print(f"🔥 TLCC Binary Mode: threshold={tlcc_threshold}, above→1, below→0")
            corr_adj_np = np.where(np.abs(corr_adj_np) >= tlcc_threshold, 1.0, 0.0)
        else:
            # threshold 넘으면 원래 상관계수 값 유지, 안 넘으면 0으로 변환 (기존 방식)
            print(f"📊 TLCC Value Mode: threshold={tlcc_threshold}, above→original_value, below→0")
            corr_adj_np = np.where(np.abs(corr_adj_np) >= tlcc_threshold, corr_adj_np, 0)
    else:
        print("📈 TLCC No Threshold: using all correlation values")
    
    np.fill_diagonal(corr_adj_np, 1)

    # 히트맵 저장 (상관계수 값 그대로, 반드시 save_path 생성 이후에 저장)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_adj_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True)
    plt.title(f"Correlation Matrix (TLCC, 원본)")
    plt.tight_layout()
    plt.savefig(f"{save_path}/corr_adj_heatmap.png", dpi=300)
    plt.close()

    # threshold 적용된 히트맵 저장 (실제 사용되는 인접행렬)
    plt.figure(figsize=(16, 12))
    sns.heatmap(pd.DataFrame(corr_adj_np), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True)
    
    if tlcc_threshold is not None:
        if tlcc_binary:
            title_suffix = f"Binary Mode (threshold={tlcc_threshold})"
        else:
            title_suffix = f"Value Mode (threshold={tlcc_threshold})"
    else:
        title_suffix = "no threshold (all correlations included)"
    
    plt.title(f"Adjacency Matrix Used in Model (TLCC, {title_suffix})")
    plt.tight_layout()
    plt.savefig(f"{save_path}/corr_adj_heatmap_thresholded.png", dpi=300)
    plt.close()
    
    # 연결성 통계 출력
    total_connections = np.count_nonzero(corr_adj_np) - corr_adj_np.shape[0]  # 대각선 제외
    total_possible = corr_adj_np.shape[0] * corr_adj_np.shape[1] - corr_adj_np.shape[0]  # 대각선 제외
    connection_ratio = total_connections / total_possible * 100
    
    print(f"\n📊 Adjacency Matrix Statistics:")
    print(f"   Matrix shape: {corr_adj_np.shape}")
    print(f"   Total connections: {total_connections}/{total_possible} ({connection_ratio:.1f}%)")
    if tlcc_threshold is not None:
        if tlcc_binary:
            print(f"   Mode: Binary (1 if |corr| >= {tlcc_threshold}, else 0)")
            print(f"   Values: {np.count_nonzero(corr_adj_np == 1)} ones, {np.count_nonzero(corr_adj_np == 0)} zeros")
        else:
            print(f"   Mode: Value (original corr if |corr| >= {tlcc_threshold}, else 0)")
            print(f"   Non-zero range: {corr_adj_np[corr_adj_np != 0].min():.4f} to {corr_adj_np[corr_adj_np != 0].max():.4f}")
    else:
        print(f"   Mode: All correlations (no threshold)")
        print(f"   Value range: {corr_adj_np.min():.4f} to {corr_adj_np.max():.4f}")

    # 실제 학습 데이터는 원본 그대로 사용, 인접행렬만 모델에 전달
    # x_train, x_test는 원본 그대로 사용 (곱셈 제거)
    
    x_train_df = pd.DataFrame(x_train)
    x_test_df = pd.DataFrame(x_test)
    y_test_df = pd.DataFrame(y_test)
    ###########################################

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    
    n_features = x_train.shape[1]

    target_dims = get_target_dims(dataset)

    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
    
    train_loader, val_loader, test_loader = create_data_loaders(
    train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )
    ############## Robust #######################

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        #corr_adg_tensor,
        corr_adj_np,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )
    print('Model:\n', model)
    
    # Check model weights for NaN/Inf
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"WARNING: NaN or Inf in model parameter {name}")
    
    # Test model with dummy input to ensure stability
    dummy_input = torch.randn(2, window_size, n_features) * 0.1
    try:
        with torch.no_grad():
            dummy_output = model(dummy_input)
            if torch.isnan(dummy_output).any() or torch.isinf(dummy_output).any():
                print("WARNING: Model produces NaN or Inf outputs!")
            else:
                print("Model forward pass test: PASSED")
    except Exception as e:
        print(f"Model forward pass test: FAILED - {e}")
        exit(1)

    # Use more stable optimizer settings
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.init_lr,
        weight_decay=1e-6,  # Add small weight decay for regularization
        eps=1e-8  # Increase epsilon for numerical stability
    )
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )
    
    import torch
    torch.autograd.set_detect_anomaly(True)  # ← 이 줄 추가

    # 학습 시작
    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test total loss: {test_loss[1]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "WADI": (0.90, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "WADI": 0}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
        "use_wadi_optimization": (dataset.upper() == "WADI"),  # WADI 최적화 추가
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
print('\n\n Total time : ', time.time() - start)