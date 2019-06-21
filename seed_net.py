from Agent import Agent
from SeedNetEnv import SeedNetEnv

input_size = (4, 84, 84)

epochs = int(1e6)
batch_size = 32
learning_rate = 1e-4
initial_epsilon = 1
min_epsilon = 0.2
epsilon_decrease_rate = 0.8/1e4
target_network_update_frequency = 512
min_buffer_size = int(10e3)
max_buffer_size = int(50e3)
actions = 800

discount_factor = 0.9

k = 5

agent = Agent(actions=action,
            network_input_shape=input_size,
            replay_memory_size=max_buffer_size,
            minibatch_size=batch_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=initial_epsilon,
            epsilon_decrease_rate=epsilon_decrease_rate,
            min_epsilon=min_epsilon)

env = SeedNetEnv()

total_steps = 0

for epoch in range(epochs):
    current_state = env.reset()
    total_reward = 0
    for i in range(10):
        action = agent.get_action(current_state)

        next_state, reward, done = env.step(action)

        agent.add_experience(current_state, action, reward, next_state, done)

        if len(agent.experiences) >= min_buffer_size:
            agent.train()

            if (agent.training_count % target_network_update_frequency == 0) and (agent.training_count > 0):
                agent.reset_target_network()

            agent.update_epsilon()

        current_state = next_state
        total_steps += 1
        total_reward += reward
    print(f'Epoch: {epoch}\tTotal Reward: {total_reward}\tBuffer Size: {len(agent.experiences)}')