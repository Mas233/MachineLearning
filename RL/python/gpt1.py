import numpy as np
import tensorflow as tf
import gym
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 定义经验回放缓冲区
buffer_size = 10000
buffer = []

# 定义训练参数
batch_size = 32
gamma = 0.99  # 折扣因子

# 定义ε-greedy策略
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# 训练模型
num_episodes = 1000

# 设置动画保存路径
animation_path = 'cartpole_animation.gif'
model_save_path = 'cartpole_model.h5'

def _train_cart_pole():
    # 创建CartPole环境
    env = gym.make('CartPole-v1')

    # 定义神经网络模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(env.action_space.n, activation='linear')
    ])

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')

    # 创建动画绘制函数
    def update(frame):
        plt.clf()
        state = frames[frame]
        plt.imshow(env.render(mode='rgb_array'))
        plt.title(f"Test Episode Frame: {frame}")

    # 训练模型
    for episode in range(num_episodes):
        state = env.reset()
        state = np.array(state).reshape(1, -1)
        total_reward = 0

        frames = []

        while True:
            # 选择动作
            action = np.argmax(model.predict(state)[0])

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)

            # 保存帧
            frames.append(env.render(mode='rgb_array'))

            # 更新状态
            state = next_state
            total_reward += reward

            # 满足结束条件
            if done:
                break

        # 绘制和保存 GIF
        fig = plt.figure()
        ani = FuncAnimation(fig, update, frames=len(frames), interval=50)
        ani.save(f'episode_{episode}_test.gif', writer='imagemagick', fps=30)

        # 打印每个episode的总奖励
        print(f"Training - Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # 保存训练好的模型到硬盘
    model.save(model_save_path)

    # 关闭环境
    env.close()

# ------------------------ 测试 ------------------------
def _test_cart_pole():
    if not os.path.exists(model_save_path):
        _train_cart_pole()
    # 创建新的环境用于测试
    test_env = gym.make('CartPole-v1')

    # 加载训练好的模型
    loaded_model = tf.keras.models.load_model(model_save_path)

    # 测试模型
    test_episodes = 10

    for episode in range(test_episodes):
        state = test_env.reset()
        state = np.array(state).reshape(1, -1)
        total_reward = 0

        while True:
            # 选择动作
            action = np.argmax(loaded_model.predict(state)[0])

            # 执行动作
            next_state, reward, done, _ = test_env.step(action)
            next_state = np.array(next_state).reshape(1, -1)
            state = next_state
            total_reward += reward

            # 满足结束条件
            if done:
                break

        print(f"Testing - Episode {episode + 1}/{test_episodes}, Total Reward: {total_reward}")

    # 关闭测试环境
    test_env.close()

if __name__ == "__main__":
    _train_cart_pole()
    _test_cart_pole()
