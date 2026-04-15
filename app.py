from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from environment import BlackjackEnv
from agent import QLearningAgent
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Global oyun ve ajan değişkenleri
env = BlackjackEnv()
agent = QLearningAgent()

if os.path.exists('blackjack_agent.pkl'):
    agent.load_model()
else:
    print("Uyarı: Model dosyası bulunamadı, AI rastgele öneri verebilir.")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/reset', methods=['POST'])
def reset_game():
    state = env.reset()
    return jsonify(get_game_data(state, False, 0))

@app.route('/api/step', methods=['POST'])
def step_game():
    data = request.json
    action = data.get('action') # 0: Hit, 1: Stand
    
    state, reward, done = env.step(action)
    return jsonify(get_game_data(state, done, reward))

def get_game_data(state, done, reward):
    player_sum, dealer_card, usable_ace = state
    
    # AI Önerisi
    ai_action = agent.choose_action(state, training=False)
    q_hit = agent.get_q(state, 0)
    q_stand = agent.get_q(state, 1)
    
    advice = "KART ÇEK (Hit)" if ai_action == 0 else "DUR (Stand)"
    
    return {
        'player_cards': env.player_cards,
        'player_sum': player_sum,
        'dealer_cards': env.dealer_cards if done else [dealer_card, '?'],
        'dealer_sum': env._get_value(env.dealer_cards) if done else '?',
        'usable_ace': usable_ace,
        'done': done,
        'reward': reward,
        'ai_advice': advice,
        'q_values': {
            'hit': round(q_hit, 3),
            'stand': round(q_stand, 3)
        }
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)
