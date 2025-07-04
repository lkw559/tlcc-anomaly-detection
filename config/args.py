import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data params ---
    parser.add_argument("--dataset", type=str.upper, default="SMD")
    parser.add_argument("--group", type=str, default="1-1", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False)

    # -- Model params ---
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=64)  # Reduced from 150 to 64
    # Forecasting Model
    parser.add_argument("--fc_n_layers", type=int, default=2)  # Reduced from 3 to 2
    parser.add_argument("--fc_hid_dim", type=int, default=64)  # Reduced from 150 to 64
    # Reconstruction Model
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=64)  # Reduced from 150 to 64
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=64)  # Reduced batch size from 256 to 64
    parser.add_argument("--init_lr", type=float, default=1e-4)  # Reduced learning rate from 1e-3 to 1e-4
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.2)  # Reduced dropout from 0.3 to 0.2
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)

    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)    # --- Other ---
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--tlcc_threshold", type=float, default=None, help="TLCC 인접행렬에 반영할 상관계수 임계값 (None이면 모든 상관계수 포함)")
    parser.add_argument("--use_true_tlcc", type=str2bool, default=False, help="Use true Time-Lagged Cross-Correlation instead of simple correlation")
    parser.add_argument("--tlcc_binary", type=str2bool, default=False, help="TLCC threshold 넘으면 상관계수 값 대신 1로 변환 (False면 원래 상관계수 값 유지)")

    return parser
