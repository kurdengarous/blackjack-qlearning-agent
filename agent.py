import numpy as np
import random
import pickle

class QLearningAgent:
    """
    Q-Learning ajanı. Epsilon-greedy stratejisi ve Q-table güncellemeleri içerir.
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
        self.q_table = {}  # (state, action) -> value
        self.alpha = alpha  # Öğrenme oranı (learning rate)
        self.gamma = gamma  # Gelecek ödül indirgeme (discount factor)
        self.epsilon = epsilon  # Keşif oranı (exploration rate)
        self.epsilon_decay = epsilon_decay  # Epsilon azalma hızı
        self.min_epsilon = min_epsilon  # Minimum epsilon değeri
        self.actions = [0, 1]  # 0: Hit, 1: Stand

    def get_q(self, state, action):
        """Belirli bir durum-aksiyon çifti için Q değerini döner."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, training=True):
        """Epsilon-greedy stratejisine göre bir aksiyon seçer."""
        if training and random.random() < self.epsilon:
            # Keşif (Exploration): Rastgele aksiyon seç
            return random.choice(self.actions)
        else:
            # Sömürü (Exploitation): En iyi Q değerine sahip aksiyonu seç
            q_values = [self.get_q(state, a) for a in self.actions]
            # Eğer Q değerleri eşitse rastgele birini seçmek daha iyidir
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Q-Learning güncelleme kuralı: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))"""
        old_q = self.get_q(state, action)
        
        if done:
            # Eğer oyun bittiyse bir sonraki durumun Q değeri yoktur
            target = reward
        else:
            next_max_q = max([self.get_q(next_state, a) for a in self.actions])
            target = reward + self.gamma * next_max_q
        
        # Güncelleme
        self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        """Her bölüm sonrası epsilon değerini azaltır."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, filename='blackjack_agent.pkl'):
        """Modeli (Q-table) kaydeder."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename='blackjack_agent.pkl'):
        """Kaydedilmiş modeli yükler."""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print(f"Model dosyası {filename} bulunamadı.")
