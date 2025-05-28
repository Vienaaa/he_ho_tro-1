import numpy as np

# Bảng chỉ số ngẫu nhiên RI theo số tiêu chí
RI_dict = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

def calculate_weights(matrix):
    """Tính trọng số từ ma trận so sánh cặp"""
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(eigvals.real)
    max_eigval = eigvals.real[max_index]
    weights = np.array(eigvecs[:, max_index].real)
    weights = weights / np.sum(weights)
    return weights, max_eigval

def consistency_ratio(matrix):
    n = matrix.shape[0]
    weights, max_eigval = calculate_weights(matrix)
    CI = (max_eigval - n) / (n - 1)
    RI = RI_dict.get(n, 1.49)  # Nếu vượt quá 10 tiêu chí thì dùng RI=1.49
    if RI == 0:
        CR = 0.0
    else:
        CR = CI / RI
    return CR

def rank_options(criteria_weights, option_scores):
    """Tính điểm tổng hợp các phương án theo trọng số tiêu chí"""
    scores = option_scores.dot(criteria_weights)
    ranks = (-scores).argsort().argsort() + 1  # Xếp hạng (1 là cao nhất)
    return scores, ranks
