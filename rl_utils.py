import cv2
import numpy as np
import matplotlib.pyplot as plt

### DISPLAY UTILITIES

def plot_stats(n_iterations, scores, epsilons):
    # Show stats
    fig, ax1 = plt.subplots()

    # Cumulative rewards
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cumulative Reward')
    ax1.plot(range(0, n_iterations), scores,  color)

    # Avg score line
    mean_win = 1000
    means = np.append(np.zeros(mean_win), scores)
    means = [np.mean(scores[i-mean_win:i]) for i in range(mean_win, n_iterations+mean_win)]
    ax1.plot(range(0, n_iterations), means, 'tab:red')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()

    color = 'tab:purple'
    ax2.set_ylabel('Randomness (%)')
    ax2.plot(range(0, n_iterations), epsilons, color)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.show()

def render(env):
    img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    cv2.imshow("Iteration " + str(iter), img)
    cv2.waitKey(50)

### LEARNING UTILITIES

def Learn(env, algo, policy, iteration):
    # Reset environment and choose first action of next iteration
    obs, info = env.reset()

    if (iteration > 1):
        action = policy.next_action(algo._q_table, obs)
    else:
        action = policy.random_action()

    term = False
    while not term:   
                 
        curr_obs, reward, term, trunc, info = env.step(action)
        # print("New observation: " + str(curr_obs))
        
        new_q_value = algo.update_q_value(prev_state=obs, action=action, curr_state=curr_obs, reward=reward)
        # print("New q value for " + str(obs + (action,)) + " = " + str(new_q_value))
        
        obs = curr_obs
        action = policy.next_action(algo._q_table, curr_obs)
        # print("Next action will be: " + ("L" if action==0 else "R"))

    # print("Episode terminated!")

def Test(env, algo, policy, iter, force_render = False):
    # Reset environment and choose first action of next iteration
    obs, info = env.reset()
    action = policy.next_action(algo._q_table, obs)

    score = 0
    term = False
    while not term:
        if (iter % 1000 == 0 or force_render == True):
            render(env=env)

        curr_obs, reward, term, trunc, info = env.step(action)
        #print("New observation: " + str(curr_obs))
        
        score += reward
        
        obs = curr_obs
        action = policy.next_action(algo._q_table, curr_obs)
        #print("Next action will be: " + ("L" if action==0 else "R"))

    return score