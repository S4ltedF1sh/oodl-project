def make_env(scenario_name, num_landmarks, num_foods, seed=None, benchmark=False):
    '''
    Creates a MultiAgentEnv object as multiagent-particle-envs-master. This can be used similar to a gym
    environment by calling multiagent-particle-envs-master.reset() and multiagent-particle-envs-master.step().
    Use multiagent-particle-envs-master.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful multiagent-particle-envs-master properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from new_env import NewEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    done_callback = scenario.done if scenario_name is "deep_learning_scenario" else None
    world = scenario.make_world(num_landmarks, num_foods, seed)
    # create multiagent environment
    if benchmark:
        env = NewEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = NewEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, done_callback)
    return env