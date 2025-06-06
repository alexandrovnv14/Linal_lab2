import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class LogisticRegression:
    def __init__(self, input_size):
        self.w = np.random.randn(input_size) * 0.01
        self.b = 0.0
        self.loss_history = []  # Для сохранения истории потерь

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -100, 100)))

    def _compute_loss(self, y, y_hat):
        eps = 1e-7
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))

    def fit(self, X, y, epochs=10, learning_rate=0.1, batch_size=10000, verbose=True):
        n_samples = len(X)
        self.loss_history = []  # Сброс истории перед обучением
        for epoch in range(epochs):
            epoch_start = time.time()
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                z = np.dot(X_batch, self.w) + self.b
                y_hat = self._sigmoid(z)
                error = y_hat - y_batch
                
                dw = np.dot(X_batch.T, error) / len(y_batch)
                db = np.mean(error)
                
                self.w -= learning_rate * dw
                self.b -= learning_rate * db

            if verbose and epoch % 1 == 0:
                # Используем подмножество для вычисления loss
                z = np.dot(X[:10000], self.w) + self.b
                y_hat = self._sigmoid(z)
                loss = self._compute_loss(y[:10000], y_hat)
                self.loss_history.append(loss)  # Сохраняем потери
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s")

    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.w) + self.b
        y_hat = self._sigmoid(z)
        return (y_hat >= threshold).astype(int)
    
    def predict_proba(self, X):
        """Возвращает вероятности положительного класса"""
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)

# Функции для визуализации
def plot_loss_history(loss_history):
    """График изменения потерь в процессе обучения"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o', linestyle='-', color='b')
    plt.title('Изменение функции потерь в процессе обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери (бинарная кросс-энтропия)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(len(loss_history)))
    plt.savefig('loss_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График потерь сохранен как 'loss_history.png'")

def plot_confusion_matrix(y_true, y_pred):
    """Тепловая карта матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Отрицательный', 'Положительный'],
                yticklabels=['Отрицательный', 'Положительный'])
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Матрица ошибок сохранена как 'confusion_matrix.png'")

def plot_roc_curve(y_true, y_pred_prob):
    """ROC-кривая и площадь под кривой (AUC)"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC кривая (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("ROC-кривая сохранена как 'roc_curve.png'")

# Параметры генерации данных
NumSamples = 1000000  # 1,000,000 примеров
NumFeatures = 30       # 30 признаков
NoiseLevel = 0.2      # Уровень шума в данных

# Генерация синтетических данных
np.random.seed(12)
print("Генерация данных...")
start_time = time.time()

# 1. Генерируем случайные веса (истинные параметры модели)
true_weights = np.random.randn(NumFeatures)
true_bias = 0.5

# 2. Создаем матрицу признаков
X = np.random.randn(NumSamples, NumFeatures)

# 3. Генерируем метки с добавлением шума
z = np.dot(X, true_weights) + true_bias
probabilities = 1 / (1 + np.exp(-z))
y = (probabilities > 0.5).astype(int)

# Добавляем шум (инвертируем часть меток)
noise_mask = np.random.rand(NumSamples) < NoiseLevel
y[noise_mask] = 1 - y[noise_mask]

# Нормализация данных
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Разделение на train/test (80/20)
split_idx = int(0.8 * NumSamples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Данные сгенерированы за {time.time()-start_time:.2f}s")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Class balance: {np.mean(y_train)*100:.1f}% positive")

# Обучение модели
print("\nОбучение модели...")
model = LogisticRegression(input_size=NumFeatures)
model.fit(X_train, y_train, epochs=10, batch_size=10000)

# Оценка качества
y_pred = model.predict(X_test)
accuracy = np.mean(y_test == y_pred)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Матрица ошибок
def print_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    print(f"Confusion Matrix:")
    print(f"[[{tn:>5} {fp:>5}]")
    print(f" [{fn:>5} {tp:>5}]]")

print_confusion_matrix(y_test, y_pred)

# Генерация графиков
plot_loss_history(model.loss_history)
plot_confusion_matrix(y_test, y_pred)

y_pred_prob = model.predict_proba(X_test)
plot_roc_curve(y_test, y_pred_prob)
