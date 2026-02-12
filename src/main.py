import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 1. VERİYİ ÇEK VE DÜZELT
gold = yf.download("GC=F", start="2018-01-01") # Daha taze veri (2018 sonrası)
if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.droplevel(1)

# 2. GÖSTERGELERİ HESAPLA (Feature Engineering)

# A) RSI Hesaplama (Manuel Formül)
delta = gold["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
gold["RSI"] = 100 - (100 / (1 + rs))

# B) Hareketli Ortalamalar (Trend)
gold["SMA_10"] = gold["Close"].rolling(10).mean() # Kısa vade
gold["SMA_50"] = gold["Close"].rolling(50).mean() # Orta vade

# C) Sinyal: Kısa vade, uzun vadeyi kesti mi?
gold["Trend"] = np.where(gold["SMA_10"] > gold["SMA_50"], 1, -1)

# D) Oynaklık ve Getiri
gold["return"] = gold["Close"].pct_change()
gold["volatility"] = gold["return"].rolling(10).std()

gold.dropna(inplace=True)

# 3. YENİ HEDEF (Target) BELİRLEME
# Sadece yükselmesi yetmez! %0.1'den fazla yükselirse "1" (AL), yoksa "0" (BEKLE)
# Bu, "gürültüyü" eler.
gold["Target"] = (gold["Close"].shift(-1) > gold["Close"] * 1.001).astype(int)

# 4. MAKİNE ÖĞRENMESİ (XGBoost)
features = ["RSI", "volatility", "return", "Trend", "SMA_10"]
X = gold[features]
y = gold["Target"]

# Son %20'yi test için ayır
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Modeli Eğit (Daha güçlü ayarlar)
model = xgb.XGBClassifier(
    n_estimators=200, 
    learning_rate=0.03, 
    max_depth=4, 
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# 5. SONUÇLAR
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Yeni Model Doğruluk Oranı: %{acc * 100:.2f}")

# Özellik Önemini Göster
print("\nModelin En Sevdiği Göstergeler:")
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(importance)