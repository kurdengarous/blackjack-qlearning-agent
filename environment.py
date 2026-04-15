import numpy as np

class BlackjackEnv:
    """
    Blackjack ortamını simüle eden sınıf.
    OpenAI Gym benzeri bir yapıdadır (reset, step).
    """
    
    def __init__(self):
        # Aksiyonlar: 0 = Hit (Kart çek), 1 = Stand (Dur)
        self.action_space = [0, 1]
        self.reset()

    def _generate_card(self):
        """1-13 arası rastgele kart üretir, 10 ve üzeri kartlar 10 değerindedir."""
        card = np.random.randint(1, 14)
        return min(card, 10)

    def _get_value(self, cards):
        """Eldeki kartların toplam değerini hesaplar, asları (Ace) optimize eder."""
        current_sum = 0
        ace_count = 0
        for card in cards:
            if card == 1:
                ace_count += 1
            else:
                current_sum += card
        
        # Asları 11 olarak ekle, eğer 21'i geçerse 1'e düşür
        for _ in range(ace_count):
            current_sum += 11
            if current_sum > 21:
                current_sum -= 10
        
        return current_sum

    def _has_usable_ace(self, cards):
        """Elde 11 olarak kullanılan bir as olup olmadığını kontrol eder."""
        current_sum = 0
        ace_count = 0
        for card in cards:
            if card == 1:
                ace_count += 1
            else:
                current_sum += card
        
        usable_ace = False
        for _ in range(ace_count):
            current_sum += 11
            if current_sum <= 21:
                usable_ace = True
            else:
                current_sum -= 10
        return usable_ace

    def reset(self):
        """Yeni bir oyun başlatır ve başlangıç state'ini döner."""
        self.player_cards = [self._generate_card(), self._generate_card()]
        self.dealer_cards = [self._generate_card(), self._generate_card()]
        
        # Eğer oyuncu direkt 21 ile başlarsa (Blackjack), bunu halletmeliyiz
        # Ancak standart state takibi için sadece durumu dönüyoruz.
        return self._get_state()

    def _get_state(self):
        """Mevcut durumu (state) döner: (oyuncu_toplamı, dealer_açık_kartı, usable_ace)"""
        return (self._get_value(self.player_cards), 
                self.dealer_cards[0], 
                self._has_usable_ace(self.player_cards))

    def step(self, action):
        """
        Bir aksiyon gerçekleştirir.
        0: Hit (Kart çek)
        1: Stand (Dur)
        
        Ödül Sistemi:
        - Kazanma: +1
        - Kaybetme: -1
        - Beraberlik: 0
        - Blackjack: +1.5
        """
        done = False
        reward = 0

        if action == 0:  # Hit (Kart Çek)
            self.player_cards.append(self._generate_card())
            if self._get_value(self.player_cards) > 21:
                done = True
                reward = -1  # Bust (21'i geçti, kaybetti)
        else:  # Stand (Dur)
            done = True
            # Dealer'ın sırası (Dealer 17'ye ulaşana kadar kart çeker)
            while self._get_value(self.dealer_cards) < 17:
                self.dealer_cards.append(self._generate_card())
            
            player_val = self._get_value(self.player_cards)
            dealer_val = self._get_value(self.dealer_cards)
            
            if dealer_val > 21:
                reward = 1  # Dealer bust (Oyuncu kazandı)
            elif player_val > dealer_val:
                reward = 1  # Oyuncu skoru daha yüksek (Kazandı)
            elif player_val < dealer_val:
                reward = -1  # Dealer skoru daha yüksek (Kaybetti)
            else:
                reward = 0  # Beraberlik (Push)
            
            # Blackjack (İlk iki kartla 21) kontrolü
            if player_val == 21 and len(self.player_cards) == 2:
                reward = 1.5 

        return self._get_state(), reward, done
