# Conjunto de funciones de metricas 
import numpy as np
from itertools import combinations

def metricas_one_vs_all(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    classes = np.unique(np.vstack((y_true, y_pred)), axis=0)
    result = []
    for cls in classes:
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        tn = sum((yt != cls and yp != cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))
        
        tp = float(tp[0])
        tn = float(tn[0])
        fp = float(fp[0])
        fn = float(fn[0])
        
        result.append({
            'CLASE': cls,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'ACCURACY' : (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0.0 else 0.0,
            'PRECISION': tp / (tp + fp) if (tp + fp) != 0.0 else 0.0,
            'RECALL'   : tp / (tp + fn) if (tp + fn) != 0.0 else 0.0,
            'F1_SCORE' : 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0.0 else 0.0,
            })
        
    return result


def metricas_one_vs_other(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    classes = np.unique(np.vstack((y_true, y_pred)), axis=0)
    
    result = []
    for cls, other_cls in combinations(classes, 2):
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        tn = sum((yt == other_cls and yp == other_cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt == other_cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls and yp == other_cls) for yt, yp in zip(y_true, y_pred))
        
        tp = float(tp[0])
        tn = float(tn[0])
        fp = float(fp[0])
        fn = float(fn[0])
        
        result.append({
            'CLASE': cls,
            'CLASE_VS': other_cls,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'ACCURACY' : (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0.0 else 0.0,
            'PRECISION': tp / (tp + fp) if (tp + fp) != 0.0 else 0.0,
            'RECALL'   : tp / (tp + fn) if (tp + fn) != 0.0 else 0.0,
            'F1_SCORE' : 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0.0 else 0.0,
            })
        
    return result
