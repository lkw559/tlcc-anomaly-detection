import json
from tqdm import tqdm
from eval_methods import *
from utils import *


class Predictor:
    """MTAD-GAT predictor class.

    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name
        self.use_epsilon = pred_args.get("use_epsilon", True)
        self.use_pot = pred_args.get("use_pot", True)
        self.use_wadi_optimization = pred_args.get("use_wadi_optimization", False)

    def get_score(self, values):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        #recons = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                #y_hat, _ = self.model(x)
                y_hat = self.model(x)

                # Shifting input to include the observed value (y) when doing the reconstruction
                #recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                #_, window_recon = self.model(recon_x)

                preds.append(y_hat.detach().cpu().numpy())
                # Extract last reconstruction only
                #recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        #recons = np.concatenate(recons, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size:]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)
        df = pd.DataFrame()
        for i in range(preds.shape[1]):
            df[f"Forecast_{i}"] = preds[:, i]
            #df[f"Recon_{i}"] = recons[:, i]
            df[f"True_{i}"] = actual[:, i]
            #a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt(
            #    (recons[:, i] - actual[:, i]) ** 2)  ########### Inculde Preds + Recon
            #a_score = np.sqrt((recons[:, i] - actual[:, i]) ** 2) ####### Remove Preds
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) ####### Remove Recon

            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1+iqr)

            anomaly_scores[:, i] = a_score
            df[f"A_Score_{i}"] = a_score

        anomaly_scores = np.mean(anomaly_scores, 1)
        df['A_Score_Global'] = anomaly_scores

        return df

    def predict_anomalies(self, train, test, true_anomalies, load_scores=False, save_output=True,
                          scale_scores=False):
        """ Predicts anomalies

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        :param scale_scores: Whether to feature-wise scale anomaly scores
        """

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")
            print('------------------------------------')
            print('train_pred_df.shape:',train_pred_df.shape)
            print('test_pred_df.shape:', test_pred_df.shape)

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            train_pred_df = self.get_score(train)
            test_pred_df = self.get_score(test)


            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

            train_anomaly_scores = adjust_anomaly_scores(train_anomaly_scores, self.dataset, True, self.window_size)
            test_anomaly_scores = adjust_anomaly_scores(test_anomaly_scores, self.dataset, False, self.window_size)

            # Update df
            train_pred_df['A_Score_Global'] = train_anomaly_scores
            test_pred_df['A_Score_Global'] = test_anomaly_scores

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
        print('------------------------------------')
        print('train_pred_df.shape:',train_pred_df.shape)
        print('test_pred_df.shape:', test_pred_df.shape)
        print('------------------------------------')
        print('train_anomaly_scores:', len(train_anomaly_scores))
        print('test_anomaly_scores:', len(test_anomaly_scores))
        

        # Find threshold and predict anomalies at feature-level (for plotting and diagnosis purposes)
        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))
        for i in range(out_dim):
            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
            epsilon = find_epsilon(train_feature_anom_scores, reg_level=2)

            train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(int)
            test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)

            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds

            train_pred_df[f"Thresh_{i}"] = epsilon
            test_pred_df[f"Thresh_{i}"] = epsilon

            all_preds[:, i] = test_feature_anom_preds

        # Global anomaly 평가 (Epsilon, POT 모두 지원)
        summary = {}
        threshold_used = None
        
        # WADI 최적화를 위한 train-validation split (연구윤리 준수)
        validation_scores = None
        validation_labels = None
        original_train_scores = train_anomaly_scores.copy()  # 원본 보존
        
        if self.use_wadi_optimization and true_anomalies is not None:
            # 훈련 데이터의 80%를 실제 훈련용, 20%를 검증용으로 분할
            split_idx = int(len(train_anomaly_scores) * 0.8)
            actual_train_scores = train_anomaly_scores[:split_idx]
            validation_scores = train_anomaly_scores[split_idx:]
            
            # 검증 라벨 생성 (정상 데이터는 모두 0으로 가정)
            validation_labels = np.zeros(len(validation_scores))
            
            print(f"🔄 연구윤리 준수: Train-Validation Split 적용")
            print(f"   Train: {len(actual_train_scores)} samples")
            print(f"   Validation: {len(validation_scores)} samples")
            
            # 임계값 계산용으로만 분할된 데이터 사용
            train_for_threshold = actual_train_scores
        else:
            train_for_threshold = train_anomaly_scores
        
        # Epsilon 방식
        if self.use_epsilon:
            e_eval = epsilon_eval(train_for_threshold, test_anomaly_scores, true_anomalies, 
                                use_wadi_optimization=self.use_wadi_optimization,
                                validation_scores=validation_scores,
                                validation_labels=validation_labels)
            optimization_status = "WADI optimized (CV-based)" if self.use_wadi_optimization else "standard"
            print(f"Results using epsilon method ({optimization_status}):\n {e_eval}")
            
            # 확장된 지표 계산 및 출력
            if true_anomalies is not None:
                from eval_methods import calculate_additional_metrics, print_extended_metrics, epsilon_eval_extended
                
                print("\n🚀 추가 평가지표 계산 중...")
                extended_result = epsilon_eval_extended(
                    train_for_threshold, test_anomaly_scores, true_anomalies,
                    reg_level=1, use_wadi_optimization=self.use_wadi_optimization,
                    validation_scores=validation_scores, validation_labels=validation_labels
                )
                
                # 확장된 지표 출력
                print_extended_metrics(extended_result, f"Epsilon ({optimization_status})")
                
                # summary에 확장된 지표도 추가
                summary["epsilon_extended"] = {k: float(v) if not isinstance(v, list) else v 
                                             for k, v in extended_result.items()}
            
            for k, v in e_eval.items():
                if not type(e_eval[k]) == list:
                    e_eval[k] = float(v)
            summary["epsilon_result"] = e_eval
            threshold_used = e_eval["threshold"]
        # POT 방식
        if self.use_pot:
            if self.use_wadi_optimization:
                # WADI 최적화된 POT 파라미터 사용
                pot_q = 0.05  # WADI에 최적화된 q 값
                optimization_status = "WADI optimized"
            else:
                pot_q = self.q
                optimization_status = "standard"
                
            pot_eval_result = pot_eval(
                train_for_threshold,  # 수정: 분할된 데이터 사용
                test_anomaly_scores,
                true_anomalies,
                q=pot_q,
                level=self.level,
                dynamic=self.dynamic_pot
            )
            print(f"Results using POT method ({optimization_status}):\n {pot_eval_result}")
            
            # POT 확장된 지표 계산 및 출력
            if true_anomalies is not None:
                from eval_methods import pot_eval_extended, print_extended_metrics
                
                print("\n🚀 POT 추가 평가지표 계산 중...")
                pot_extended_result = pot_eval_extended(
                    train_for_threshold, test_anomaly_scores, true_anomalies,
                    q=pot_q, level=self.level, dynamic=self.dynamic_pot
                )
                
                # 확장된 지표 출력
                print_extended_metrics(pot_extended_result, f"POT ({optimization_status})")
                
                # summary에 확장된 지표도 추가
                summary["pot_extended"] = {k: float(v) if not isinstance(v, list) else v 
                                         for k, v in pot_extended_result.items()}
            
            for k, v in pot_eval_result.items():
                if not type(pot_eval_result[k]) == list:
                    pot_eval_result[k] = float(v)
            summary["pot_result"] = pot_eval_result
            if not self.use_epsilon:
                threshold_used = pot_eval_result["threshold"]
            elif threshold_used is None:
                threshold_used = pot_eval_result["threshold"]
        
        # 종합 성능 평가 및 가이드 출력 (노트북 실행 시)
        if true_anomalies is not None and (self.use_epsilon or self.use_pot):
            print("\n" + "="*80)
            print("🎯 종합 성능 평가 및 가이드")
            print("="*80)
            
            # 사용된 방법 표시
            methods_used = []
            if self.use_epsilon:
                methods_used.append("Epsilon")
            if self.use_pot:
                methods_used.append("POT")
            print(f"📊 적용된 평가 방법: {', '.join(methods_used)}")
            
            # 최고 성능 방법 식별
            best_method = None
            best_f1 = 0
            
            if "epsilon_extended" in summary:
                eps_f1 = summary["epsilon_extended"].get("f1", 0)
                if eps_f1 > best_f1:
                    best_f1 = eps_f1
                    best_method = "Epsilon"
            
            if "pot_extended" in summary:
                pot_f1 = summary["pot_extended"].get("f1", 0)
                if pot_f1 > best_f1:
                    best_f1 = pot_f1
                    best_method = "POT"
            
            if best_method:
                print(f"🏆 최고 성능 방법: {best_method} (F1: {best_f1:.4f})")
                best_result = summary.get(f"{best_method.lower()}_extended", {})
                
                # 성능 가이드 (더 정확한 기준 적용)
                print(f"\n📈 성능 지표 평가:")
                roc_auc = best_result.get("roc_auc", 0)
                pr_auc = best_result.get("pr_auc", 0) 
                mcc = best_result.get("mcc", 0)
                
                # ROC-AUC 평가 (이상탐지 특화 기준)
                if roc_auc >= 0.9:
                    print(f"   ROC-AUC {roc_auc:.4f}: 탁월 🟢")
                elif roc_auc >= 0.8:
                    print(f"   ROC-AUC {roc_auc:.4f}: 우수 🟢")
                elif roc_auc >= 0.7:
                    print(f"   ROC-AUC {roc_auc:.4f}: 양호 🟡")
                elif roc_auc >= 0.6:
                    print(f"   ROC-AUC {roc_auc:.4f}: 보통 🟠")
                else:
                    print(f"   ROC-AUC {roc_auc:.4f}: 개선필요 🔴")
                
                # PR-AUC 평가 (불균형 데이터 특화 기준)
                if pr_auc >= 0.7:
                    print(f"   PR-AUC {pr_auc:.4f}: 탁월 🟢")
                elif pr_auc >= 0.5:
                    print(f"   PR-AUC {pr_auc:.4f}: 우수 🟢")
                elif pr_auc >= 0.3:
                    print(f"   PR-AUC {pr_auc:.4f}: 양호 🟡")
                elif pr_auc >= 0.2:
                    print(f"   PR-AUC {pr_auc:.4f}: 보통 🟠")
                else:
                    print(f"   PR-AUC {pr_auc:.4f}: 개선필요 🔴")
                
                # MCC 평가 (균형 지표)
                if mcc >= 0.8:
                    print(f"   MCC {mcc:.4f}: 탁월 🟢")
                elif mcc >= 0.6:
                    print(f"   MCC {mcc:.4f}: 우수 🟢")
                elif mcc >= 0.4:
                    print(f"   MCC {mcc:.4f}: 양호 🟡")
                elif mcc >= 0.2:
                    print(f"   MCC {mcc:.4f}: 보통 🟠")
                else:
                    print(f"   MCC {mcc:.4f}: 개선필요 🔴")
            
            print("\n💡 실험 참고사항:")
            if self.use_wadi_optimization:
                print("   ✅ 연구윤리 준수: CV 기반 임계값 선택 적용됨")
            print("   📋 모든 확장 지표가 summary 파일에 저장됨")
            print("   🔄 다른 TLCC threshold 값 실험 권장")
            print("="*80)
        
        # 결과 저장
        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        # Save anomaly predictions made using epsilon method
        if save_output:
            global_epsilon = threshold_used
            test_pred_df["A_True_Global"] = true_anomalies
            train_pred_df["Thresh_Global"] = global_epsilon
            test_pred_df["Thresh_Global"] = global_epsilon
            train_pred_df[f"A_Pred_Global"] = (original_train_scores >= global_epsilon).astype(int)
            test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
            # Adjust predictions according to evaluation strategy
            if true_anomalies is not None:
                test_preds_global = adjust_predicts(None, true_anomalies, global_epsilon, pred=test_preds_global)
            test_pred_df[f"A_Pred_Global"] = test_preds_global

            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("-- Done.")

