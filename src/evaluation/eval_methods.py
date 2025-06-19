import numpy as np
import more_itertools as mit
from spot import SPOT, dSPOT


def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): boolean list of true anomalies in score
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return dict: pot result dict
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": pot_th,
            "latency": p_latency,
        }
    else:
        return {
            "threshold": pot_th,
        }


"""def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """"""
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """"""

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        #print('search_step:', i)
        #print('threshold:', threshold)
        threshold += search_range / float(search_step)
        target, latency = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": m[3],
        "TN": m[4],
        "FP": m[5],
        "FN": m[6],
        "threshold": m_t,
        "latency": m_l,
    }"""


def calc_seq(score, label, threshold):
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency


def epsilon_eval(train_scores, test_scores, test_labels, reg_level=1, use_wadi_optimization=False, 
                 validation_scores=None, validation_labels=None):
    """
    연구윤리를 준수하는 epsilon 평가 함수
    
    Args:
        train_scores: 훈련 데이터 스코어
        test_scores: 테스트 데이터 스코어  
        test_labels: 테스트 라벨
        reg_level: 정규화 레벨
        use_wadi_optimization: WADI 최적화 사용 여부
        validation_scores: 검증 데이터 스코어 (WADI 최적화 시 필요)
        validation_labels: 검증 데이터 라벨 (WADI 최적화 시 필요)
    """
    if use_wadi_optimization:
        best_epsilon = find_epsilon_adaptive(
            train_scores, reg_level, 
            use_wadi_optimization=True,
            validation_errors=validation_scores,
            validation_labels=validation_labels
        )
    else:
        best_epsilon = find_epsilon(train_scores, reg_level)
    pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=True)
    if test_labels is not None:
        p_t = calc_point2point(pred, test_labels)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": best_epsilon,
            "latency": p_latency,
            "reg_level": reg_level,
        }
    else:
        return {"threshold": best_epsilon, "reg_level": reg_level}


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


def find_threshold_cv(train_errors, validation_errors, validation_labels, method='percentile'):
    """
    연구윤리에 맞는 Cross-Validation 기반 임계값 선택
    
    Args:
        train_errors: 훈련 데이터의 reconstruction errors
        validation_errors: 검증 데이터의 reconstruction errors  
        validation_labels: 검증 데이터의 실제 라벨
        method: 'percentile' 또는 'epsilon'
    
    Returns:
        최적 임계값
    """
    if method == 'percentile':
        # 다양한 percentile을 검증 데이터로 평가
        candidate_percentiles = [90, 92, 94, 95, 96, 97, 98, 99]
        best_f1 = 0
        best_threshold = None
        
        for p in candidate_percentiles:
            threshold = np.percentile(train_errors, p)
            pred = (validation_errors >= threshold).astype(int)
            
            # F1 계산
            tp = np.sum((pred == 1) & (validation_labels == 1))
            fp = np.sum((pred == 1) & (validation_labels == 0))
            fn = np.sum((pred == 0) & (validation_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold if best_threshold is not None else np.percentile(train_errors, 95)
    
    else:  # epsilon method
        return find_epsilon(train_errors, reg_level=1)


def find_optimal_threshold_for_wadi(errors, percentile=95):
    """
    ⚠️ DEPRECATED: 연구윤리 문제로 사용 금지
    Data snooping 문제가 있는 함수 - find_threshold_cv 사용 권장
    """
    import warnings
    warnings.warn("find_optimal_threshold_for_wadi는 연구윤리 문제로 deprecated됨. find_threshold_cv 사용 권장", 
                  DeprecationWarning, stacklevel=2)
    return np.percentile(errors, percentile)


def find_epsilon_adaptive(errors, reg_level=1, use_wadi_optimization=False, validation_errors=None, validation_labels=None):
    """
    연구윤리를 준수하는 적응형 epsilon 방법
    """
    if use_wadi_optimization and validation_errors is not None and validation_labels is not None:
        # Cross-validation 기반 임계값 선택 (연구윤리 준수)
        return find_threshold_cv(errors, validation_errors, validation_labels, method='percentile')
    elif use_wadi_optimization:
        # 검증 데이터가 없으면 경고 후 기본 방법 사용
        import warnings
        warnings.warn("WADI 최적화를 위해서는 validation 데이터가 필요합니다. 기본 epsilon 방법을 사용합니다.", 
                      UserWarning, stacklevel=2)
        return find_epsilon(errors, reg_level)
    else:
        # 기존 epsilon 방법
        return find_epsilon(errors, reg_level)


def pot_eval_optimized_for_wadi(init_score, score, label, q=1e-3, level=0.99, dynamic=False, use_wadi_optimization=False):
    """
    WADI 최적화된 POT 평가 함수
    """
    if use_wadi_optimization:
        # WADI에 최적화된 q 값 사용
        q = 0.05
        print(f"Using WADI-optimized POT parameters: q={q}, level={level}")
    
    return pot_eval(init_score, score, label, q=q, level=level, dynamic=dynamic)

# ============================================================================
# 추가 평가지표 함수들 (ROC-AUC, PR-AUC, MCC)
# ============================================================================

def calculate_additional_metrics(y_true, y_pred, y_scores):
    """
    추가 평가지표 계산 함수
    
    Args:
        y_true: 실제 라벨 (0 또는 1)
        y_pred: 예측 라벨 (0 또는 1) 
        y_scores: 이상 점수 (연속값)
    
    Returns:
        dict: 추가 평가지표들
    """
    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve, auc,
        average_precision_score, matthews_corrcoef
    )
    import numpy as np
    
    metrics = {}
    
    try:
        # 1. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
        if len(np.unique(y_true)) > 1:  # 라벨이 0과 1 모두 있어야 계산 가능
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        else:
            metrics['roc_auc'] = 0.0
            
        # 2. PR-AUC (Precision-Recall Area Under Curve)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        metrics['pr_auc'] = auc(recall, precision)
        
        # 3. Average Precision Score (AP) - PR-AUC의 다른 계산법
        metrics['average_precision'] = average_precision_score(y_true, y_scores)
        
        # 4. Matthews Correlation Coefficient (MCC)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
    except Exception as e:
        print(f"Warning: Error calculating additional metrics: {e}")
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
        metrics['average_precision'] = 0.0
        metrics['mcc'] = 0.0
    
    return metrics


def epsilon_eval_extended(train_scores, test_scores, test_labels, reg_level=1, use_wadi_optimization=False, 
                         validation_scores=None, validation_labels=None):
    """
    확장된 평가지표를 포함한 epsilon 평가 함수
    
    Args:
        train_scores: 훈련 데이터 스코어
        test_scores: 테스트 데이터 스코어  
        test_labels: 테스트 라벨
        reg_level: 정규화 레벨
        use_wadi_optimization: WADI 최적화 사용 여부
        validation_scores: 검증 데이터 스코어 (WADI 최적화 시 필요)
        validation_labels: 검증 데이터 라벨 (WADI 최적화 시 필요)
    
    Returns:
        dict: 기존 지표 + 추가 지표들
    """
    # 기존 epsilon_eval 결과 계산
    result = epsilon_eval(train_scores, test_scores, test_labels, reg_level, 
                         use_wadi_optimization, validation_scores, validation_labels)
    
    if test_labels is not None:
        # 예측값 계산
        threshold = result['threshold']
        pred_result = adjust_predicts(test_scores, test_labels, threshold, calc_latency=False)
        if isinstance(pred_result, tuple):
            pred = pred_result[0]
        else:
            pred = pred_result
        
        # 추가 지표 계산
        additional_metrics = calculate_additional_metrics(test_labels, pred, test_scores)
        
        # 기존 결과에 추가 지표 병합
        result.update(additional_metrics)
    
    return result


def pot_eval_extended(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    """
    확장된 평가지표를 포함한 POT 평가 함수
    
    Args:
        init_score: 초기 스코어 (훈련 데이터)
        score: 테스트 스코어
        label: 테스트 라벨
        q: Detection level (risk)
        level: Probability associated with the initial threshold
        dynamic: Dynamic threshold 사용 여부
    
    Returns:
        dict: 기존 지표 + 추가 지표들
    """
    # 기존 pot_eval 결과 계산
    result = pot_eval(init_score, score, label, q, level, dynamic)
    
    if label is not None:
        # 예측값 계산
        threshold = result['threshold']
        pred_result = adjust_predicts(score, label, threshold, calc_latency=False)
        if isinstance(pred_result, tuple):
            pred = pred_result[0]
        else:
            pred = pred_result
        
        # 추가 지표 계산
        additional_metrics = calculate_additional_metrics(label, pred, score)
        
        # 기존 결과에 추가 지표 병합
        result.update(additional_metrics)
    
    return result


def print_extended_metrics(result, method_name=""):
    """
    확장된 평가지표를 보기 좋게 출력하는 함수
    
    Args:
        result: epsilon_eval_extended 또는 pot_eval_extended 결과
        method_name: 방법 이름 (출력용)
    """
    print(f"\n=== {method_name} 확장 평가지표 ===")
    
    # 기본 지표
    print(f"📊 기본 지표:")
    print(f"   F1 Score: {result.get('f1', 0):.4f}")
    print(f"   Precision: {result.get('precision', 0):.4f}")
    print(f"   Recall: {result.get('recall', 0):.4f}")
    
    # 추가 지표 (AUC 계열)
    print(f"📈 AUC 지표:")
    print(f"   ROC-AUC: {result.get('roc_auc', 0):.4f}")
    print(f"   PR-AUC: {result.get('pr_auc', 0):.4f}")
    print(f"   Average Precision: {result.get('average_precision', 0):.4f}")
    
    # 균형 지표
    print(f"⚖️  균형 지표:")
    print(f"   MCC: {result.get('mcc', 0):.4f}")
    
    # 세부 정보
    print(f"🔢 세부 정보:")
    print(f"   TP: {result.get('TP', 0)}, FP: {result.get('FP', 0)}")
    print(f"   TN: {result.get('TN', 0)}, FN: {result.get('FN', 0)}")
    print(f"   Threshold: {result.get('threshold', 0):.6f}")


def compare_methods_extended(train_scores, test_scores, test_labels, 
                           validation_scores=None, validation_labels=None):
    """
    Epsilon과 POT 방법을 확장된 지표로 비교
    
    Args:
        train_scores: 훈련 스코어
        test_scores: 테스트 스코어
        test_labels: 테스트 라벨
        validation_scores: 검증 스코어 (옵션)
        validation_labels: 검증 라벨 (옵션)
    
    Returns:
        dict: 두 방법의 비교 결과
    """
    print("🔍 확장된 지표로 방법 비교 중...")
    
    # Epsilon 방법 (윤리적 CV 기반)
    epsilon_result = epsilon_eval_extended(
        train_scores, test_scores, test_labels, 
        reg_level=1, use_wadi_optimization=True,
        validation_scores=validation_scores,
        validation_labels=validation_labels
    )
    
    # POT 방법
    pot_result = pot_eval_extended(train_scores, test_scores, test_labels, q=0.05, level=0.9)
    
    # 결과 출력
    print_extended_metrics(epsilon_result, "Epsilon (윤리적 CV)")
    print_extended_metrics(pot_result, "POT")
    
    # 비교 요약
    print(f"\n🏆 지표별 우승자:")
    metrics_to_compare = ['f1', 'roc_auc', 'pr_auc', 'mcc']
    
    for metric in metrics_to_compare:
        epsilon_val = epsilon_result.get(metric, 0)
        pot_val = pot_result.get(metric, 0)
        
        if epsilon_val > pot_val:
            winner = "Epsilon"
            diff = epsilon_val - pot_val
        elif pot_val > epsilon_val:
            winner = "POT"
            diff = pot_val - epsilon_val
        else:
            winner = "동점"
            diff = 0
        
        print(f"   {metric.upper()}: {winner} (차이: {diff:+.4f})")
    
    return {
        'epsilon': epsilon_result,
        'pot': pot_result
    }
