import numpy as np
import matplotlib.pyplot as plt
from environment import BlackjackEnv
from agent import QLearningAgent
import argparse
import os

def train_agent(episodes=100000):
    """Ajanı belirtilen sayıda bölüm boyunca eğitir."""
    env = BlackjackEnv()
    agent = QLearningAgent()
    
    win_rates = []
    wins = 0
    total_reward = 0
    rewards_history = []
    
    print(f"{episodes} bölümlük eğitim başlatılıyor...")
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            
        agent.decay_epsilon()
        
        if reward > 0:
            wins += 1
        total_reward += reward
        
        # Her 1000 bölümde bir istatistikleri kaydet
        if episode % 1000 == 0:
            win_rate = wins / 1000
            win_rates.append(win_rate)
            rewards_history.append(total_reward / 1000)
            wins = 0
            total_reward = 0
            if episode % 10000 == 0:
                print(f"Episode: {episode}, Epsilon: {agent.epsilon:.4f}, Win Rate: {win_rate:.2f}")

    # Modeli kaydet
    agent.save_model()
    print("Eğitim tamamlandı ve model kaydedildi.")
    
    return win_rates, rewards_history

def plot_training(win_rates):
    """Eğitim istatistiklerini görselleştirir."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1000, len(win_rates) * 1000 + 1, 1000), win_rates)
    plt.title('Win Rate vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (Last 1000 episodes)')
    plt.grid(True)
    plt.savefig('win_rate_graph.png')
    print("Win rate grafiği 'win_rate_graph.png' olarak kaydedildi.")
    plt.show()

def human_play_with_ai():
    """İnsan oyuncunun AI tavsiyesiyle oynamasını sağlar."""
    env = BlackjackEnv()
    agent = QLearningAgent()
    
    if os.path.exists('blackjack_agent.pkl'):
        agent.load_model()
        print("Eğitilmiş model yüklendi.")
    else:
        print("Uyarı: Eğitilmiş model bulunamadı, AI rastgele öneri verebilir.")
    
    while True:
        state = env.reset()
        done = False
        print("\n--- YENİ OYUN ---")
        
        while not done:
            player_sum, dealer_card, usable_ace = state
            print(f"\nSenin Toplamın: {player_sum}, Dealer Açık Kartı: {dealer_card}, Usable Ace: {usable_ace}")
            
            # AI Önerisi
            ai_action = agent.choose_action(state, training=False)
            q_hit = agent.get_q(state, 0)
            q_stand = agent.get_q(state, 1)
            
            advice = "KART ÇEK (Hit)" if ai_action == 0 else "DUR (Stand)"
            print(f"AI Önerisi: {advice} (Q-Values: Hit={q_hit:.2f}, Stand={q_stand:.2f})")
            
            user_input = input("Senin Kararın (h: Hit, s: Stand, q: Quit): ").lower()
            
            if user_input == 'q':
                return
            elif user_input == 'h':
                action = 0
            elif user_input == 's':
                action = 1
            else:
                print("Geçersiz giriş! 'h' veya 's' kullanın.")
                continue
            
            state, reward, done = env.step(action)
            
            if done:
                print(f"\nOyun Bitti! Senin Kartların: {env.player_cards} (Toplam: {env._get_value(env.player_cards)})")
                print(f"Dealer Kartları: {env.dealer_cards} (Toplam: {env._get_value(env.dealer_cards)})")
                
                if reward > 0:
                    print("TEBRİKLER! KAZANDIN!")
                elif reward < 0:
                    print("MAALESEF! KAYBETTİN!")
                else:
                    print("BERABERE!")
        
        again = input("\nTekrar oynamak ister misin? (y/n): ").lower()
        if again != 'y':
            break

def visualize_policy(agent):
    """AI'nın politikasını görselleştirir."""
    # Usable Ace durumu için politika haritası
    for usable_ace in [True, False]:
        policy_grid = np.zeros((11, 10)) # Rows: Player sum (11-21), Cols: Dealer card (1-10)
        for player_sum in range(11, 22):
            for dealer_card in range(1, 11):
                state = (player_sum, dealer_card, usable_ace)
                action = agent.choose_action(state, training=False)
                policy_grid[player_sum-11, dealer_card-1] = action
        
        plt.figure(figsize=(10, 6))
        title = "AI Policy - With Usable Ace" if usable_ace else "AI Policy - Without Usable Ace"
        plt.imshow(policy_grid, origin='lower', extent=[0.5, 10.5, 10.5, 21.5], cmap='RdBu', alpha=0.6)
        plt.title(title)
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        plt.xticks(range(1, 11), ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        plt.yticks(range(11, 22))
        plt.colorbar(label='0: Hit (Red), 1: Stand (Blue)')
        plt.savefig(f"policy_{'ace' if usable_ace else 'no_ace'}.png")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackjack RL Agent")
    parser.add_argument('--train', action='store_true', help='Ajanı eğit')
    parser.add_argument('--episodes', type=int, default=100000, help='Eğitim için bölüm sayısı')
    parser.add_argument('--play', action='store_true', help='AI ile oyna')
    parser.add_argument('--plot', action='store_true', help='Mevcut politikayı görselleştir')
    
    args = parser.parse_args()
    
    if args.train:
        win_rates, rewards = train_agent(args.episodes)
        plot_training(win_rates)
    elif args.play:
        human_play_with_ai()
    elif args.plot:
        agent = QLearningAgent()
        agent.load_model()
        visualize_policy(agent)
    else:
        parser.print_help()
