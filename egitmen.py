import os
import subprocess
from sklearn.model_selection import cross_val_score

# Hedef doğruluk değeri
target_accuracy = 0.95

# Çapraz doğrulama sayısı
num_folds = 5

# Veri dosyası yolu
data_path = "veriler.txt"

# Takatuka.py dosyasının yolu
takatuka_path = "takatuka.py"

# Çapraz doğrulama işlemini gerçekleştir
current_accuracy = 0
while current_accuracy < target_accuracy:
    # Veri dosyasını yükle
    with open(data_path, "r") as f:
        data = f.read()
    
    # Takatuka.py dosyasını çalıştır
    result = subprocess.run(["python", takatuka_path], input=data.encode(), capture_output=True)
    
    # Çapraz doğrulama işlemini gerçekleştir
    scores = cross_val_score(result.stdout.decode(), cv=num_folds)
    current_accuracy = scores.mean()
    
    # Takatuka.py dosyasını yeniden çalıştır
    subprocess.run(["python", takatuka_path, "--retrain"])
    
print(f"Target accuracy ({target_accuracy}) reached!")
