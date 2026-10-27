"""
Câu 2. Nhận dạng ký tự bằng Naive Bayes và k-Nearest Neighbors
Dữ liệu: UCI Letter Recognition (20.000 mẫu, 26 lớp A-Z, 16 đặc trưng)
Chia dữ liệu theo khuyến nghị của tác giả: 16.000 mẫu đầu để train, 4.000 mẫu cuối để test.

Cả hai thuật toán đều tự cài đặt bằng NumPy.
Chạy: python letter_recognition.py [đường_dẫn_letter-recognition.data]
"""
import sys
import time
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "letter-recognition.data"
N_TRAIN = 16000


# ----------------------------------------------------------------- dữ liệu
def load_data(path):
    labels, feats = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 17:
                continue
            labels.append(ord(parts[0]) - ord("A"))      # A..Z -> 0..25
            feats.append([float(v) for v in parts[1:]])
    return np.array(feats), np.array(labels)


# ------------------------------------------------------------- Naive Bayes
class GaussianNaiveBayes:
    """P(c|x) ~ P(c) * prod_j N(x_j | mu_cj, var_cj); dự đoán bằng log-xác suất."""

    def __init__(self, var_smoothing=1e-3):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = np.zeros(len(self.classes))
        self.mu = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros_like(self.mu)
        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            self.prior[i] = len(Xc) / len(X)
            self.mu[i] = Xc.mean(axis=0)
            self.var[i] = Xc.var(axis=0)
        # làm mịn phương sai để tránh chia cho 0
        self.var += self.var_smoothing * X.var(axis=0).max()
        return self

    def predict(self, X):
        log_prior = np.log(self.prior)
        # (n, 1, d) - (1, C, d) -> (n, C, d)
        diff2 = (X[:, None, :] - self.mu[None, :, :]) ** 2
        log_lik = -0.5 * (np.log(2 * np.pi * self.var)[None] + diff2 / self.var[None]).sum(axis=2)
        return self.classes[np.argmax(log_lik + log_prior, axis=1)]


class CategoricalNaiveBayes:
    """Đặc trưng là số nguyên 0..15 nên coi mỗi đặc trưng là biến rời rạc,
    ước lượng P(x_j = v | c) bằng đếm + làm mịn Laplace."""

    def __init__(self, alpha=1.0, n_values=16):
        self.alpha, self.n_values = alpha, n_values

    def fit(self, X, y):
        X = X.astype(int)
        self.classes = np.unique(y)
        C, d, V = len(self.classes), X.shape[1], self.n_values
        self.log_prior = np.zeros(C)
        self.log_lik = np.zeros((C, d, V))
        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[i] = np.log(len(Xc) / len(X))
            for j in range(d):
                cnt = np.bincount(Xc[:, j], minlength=V) + self.alpha
                self.log_lik[i, j] = np.log(cnt / cnt.sum())
        return self

    def predict(self, X):
        X = X.astype(int)
        scores = np.tile(self.log_prior, (len(X), 1))
        for i in range(len(self.classes)):
            scores[:, i] += self.log_lik[i, np.arange(X.shape[1]), X].sum(axis=1)
        return self.classes[np.argmax(scores, axis=1)]


# --------------------------------------------------------------------- KNN
class KNN:
    """k láng giềng gần nhất, khoảng cách Euclid, bỏ phiếu đa số."""

    def __init__(self, k=3, chunk=500):
        self.k, self.chunk = k, chunk

    def fit(self, X, y):
        self.X, self.y = X, y
        self.sq = (X ** 2).sum(axis=1)
        return self

    def kneighbors_labels(self, X, kmax):
        """Nhãn của kmax láng giềng gần nhất, đã sắp theo khoảng cách."""
        out = []
        for s in range(0, len(X), self.chunk):
            B = X[s:s + self.chunk]
            d2 = (B ** 2).sum(axis=1)[:, None] + self.sq[None, :] - 2 * B @ self.X.T
            idx = np.argpartition(d2, kmax, axis=1)[:, :kmax]
            order = np.argsort(np.take_along_axis(d2, idx, axis=1), axis=1)
            out.append(self.y[np.take_along_axis(idx, order, axis=1)])
        return np.vstack(out)

    @staticmethod
    def vote(neigh_labels, k, n_classes=26):
        sub = neigh_labels[:, :k]
        counts = np.zeros((len(sub), n_classes), dtype=int)
        for j in range(k):
            np.add.at(counts, (np.arange(len(sub)), sub[:, j]), 1)
        # hòa phiếu -> lấy lớp của láng giềng gần nhất (cộng thưởng nhỏ)
        counts = counts * 2.0
        counts[np.arange(len(sub)), sub[:, 0]] += 1
        return counts.argmax(axis=1)

    def predict(self, X):
        return self.vote(self.kneighbors_labels(X, self.k), self.k)


# ---------------------------------------------------------------- đánh giá
def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, n=26):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_report(cm):
    letters = [chr(ord("A") + i) for i in range(26)]
    rows = []
    for i, ch in enumerate(letters):
        tp = cm[i, i]
        rec = tp / cm[i].sum() if cm[i].sum() else 0
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0
        rows.append((ch, prec, rec))
    return rows


def top_confusions(cm, n=8):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:n], off.shape))[0]
    return [(chr(65 + i), chr(65 + j), off[i, j]) for i, j in idx]


def main():
    X, y = load_data(PATH)
    print(f"Dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng, {len(np.unique(y))} lớp")
    Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
    print(f"Train: {len(Xtr)} | Test: {len(Xte)}\n")

    results = {}

    # ---- Naive Bayes
    for name, model in [("Naive Bayes (Gaussian)", GaussianNaiveBayes()),
                        ("Naive Bayes (Categorical, Laplace)", CategoricalNaiveBayes())]:
        t = time.time()
        pred = model.fit(Xtr, ytr).predict(Xte)
        results[name] = (accuracy(yte, pred), pred)
        print(f"{name:38s} accuracy = {results[name][0]*100:6.2f}%   ({time.time()-t:.2f}s)")

    # ---- KNN: thử nhiều k, chọn k bằng validation (không dùng test để chọn)
    print("\nChọn k cho KNN bằng validation (12.000 train / 4.000 val):")
    knn_val = KNN().fit(Xtr[:12000], ytr[:12000])
    ks = [1, 3, 5, 7, 9, 11, 15]
    neigh = knn_val.kneighbors_labels(Xtr[12000:], max(ks))
    val_acc = {k: accuracy(ytr[12000:], KNN.vote(neigh, k)) for k in ks}
    for k in ks:
        print(f"  k = {k:2d}: {val_acc[k]*100:6.2f}%")
    best_k = max(val_acc, key=val_acc.get)
    print(f"  -> k tốt nhất = {best_k}\n")

    knn = KNN(k=best_k).fit(Xtr, ytr)
    neigh_te = knn.kneighbors_labels(Xte, max(ks))
    print("KNN trên tập test:")
    for k in ks:
        print(f"  k = {k:2d}: {accuracy(yte, KNN.vote(neigh_te, k))*100:6.2f}%")
    pred_knn = KNN.vote(neigh_te, best_k)
    results[f"KNN (k={best_k})"] = (accuracy(yte, pred_knn), pred_knn)

    # ---- KNN với chuẩn hoá (z-score) để so sánh
    mu, sd = Xtr.mean(0), Xtr.std(0)
    knn_z = KNN(k=best_k).fit((Xtr - mu) / sd, ytr)
    pred_z = knn_z.predict((Xte - mu) / sd)
    results[f"KNN (k={best_k}, z-score)"] = (accuracy(yte, pred_z), pred_z)

    # ---- Tổng kết
    print("\n" + "=" * 62)
    print("TỔNG KẾT (tập test 4.000 mẫu)")
    print("=" * 62)
    for name, (acc, _) in results.items():
        print(f"{name:38s} {acc*100:6.2f}%")

    best_name = max(results, key=lambda n: results[n][0])
    cm = confusion_matrix(yte, results[best_name][1])
    print(f"\nMô hình tốt nhất: {best_name}")
    print("Các cặp nhầm lẫn nhiều nhất (thật -> dự đoán):")
    for a, b, c in top_confusions(cm):
        print(f"  {a} -> {b}: {c} lần")

    cm_nb = confusion_matrix(yte, results["Naive Bayes (Gaussian)"][1])
    print("\nNaive Bayes (Gaussian) – các cặp nhầm lẫn nhiều nhất:")
    for a, b, c in top_confusions(cm_nb):
        print(f"  {a} -> {b}: {c} lần")

    print("\nPrecision / Recall theo lớp của mô hình tốt nhất:")
    for ch, p, r in per_class_report(cm):
        print(f"  {ch}: P={p:.3f}  R={r:.3f}")

    np.savetxt("confusion_matrix_best.csv", cm, fmt="%d", delimiter=",")


if __name__ == "__main__":
    main()
