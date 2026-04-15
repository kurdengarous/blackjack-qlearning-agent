# Blackjack Pekiştirmeli Öğrenme (Reinforcement Learning) Projesi

Bu proje, Blackjack (21) oyununu simüle eden bir ortam ve bu ortamda kendi kendine oynamayı öğrenen bir Q-Learning ajanını içerir. Mevcut yapı, klasik Monte Carlo ES yöntemine alternatif olarak modern bir Q-Learning yaklaşımı sunar.

## Özellikler

- **Modüler Yapı:** Ortam (`environment.py`), Ajan (`agent.py`) ve Ana Kontrol (`main.py`) olarak ayrılmıştır.
- **Q-Learning Algoritması:** Epsilon-greedy stratejisi ve epsilon decay (keşif oranının zamanla azalması) mekanizması kullanır.
- **İnteraktif Mod:** Kullanıcı oyun oynarken AI'dan gerçek zamanlı hamle önerisi (Hit/Stand) alabilir.
- **Eğitim Modu:** Binlerce oyun üzerinden ajanı eğitme ve win rate (kazanma oranı) takibi.
- **Görselleştirme:** Eğitim süreci ve öğrenilen politikanın (policy) grafiksel gösterimi.

## Dosya Yapısı

- `environment.py`: Blackjack oyun kuralları ve state/action tanımları.
- `agent.py`: Q-Learning ajanı, Q-table yönetimi ve model kaydetme/yükleme.
- `main.py`: Eğitim, interaktif oyun ve görselleştirme fonksiyonlarını içeren ana giriş noktası.
- `blackjack_agent.pkl`: Eğitilmiş modelin (Q-table) saklandığı dosya.

## Nasıl Çalıştırılır?

### 1. Eğitim (Training)
Ajanı eğitmek için aşağıdaki komutu kullanın:
```bash
python main.py --train --episodes 100000
```
Bu komut, ajanı 100,000 el boyunca eğitir ve eğitim sonunda `win_rate_graph.png` grafiğini oluşturur.

### 2. AI Önerisi ile Oynamak (Human + AI)
Eğitilmiş bir model ile kendiniz oynamak ve AI'dan tavsiye almak için:
```bash
python main.py --play
```
Her turda AI size mevcut Q değerlerine göre "KART ÇEK" veya "DUR" önerisi verecektir.

### 3. Politikayı Görselleştirmek
AI'nın hangi durumda ne karar verdiğini (Hit/Stand) görmek için:
```bash
python main.py --plot
```

## Ödül Sistemi
- **Kazanma:** +1
- **Kaybetme:** -1
- **Beraberlik:** 0
- **Blackjack:** +1.5 (Opsiyonel olarak ödüllendirilir)

## Gereksinimler
- Python 3.x
- NumPy
- Matplotlib

Eğitim sırasında win rate takibi yapılarak ajanın gelişimi gözlemlenebilir. Genellikle 50,000 - 100,000 episode sonrası ajan optimal politikaya yakın bir performans sergilemeye başlar.
