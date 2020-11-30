"""
Code for creating a multiagent-particle-envs-master environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    multiagent-particle-envs-master = make_env('simple_speaker_listener')
After producing the multiagent-particle-envs-master object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (multiagent-particle-envs-master.world.dim_p + multiagent-particle-envs-master.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
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
    from env.environment import MultiAgentEnv
    import env.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent-particle-envs-master environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
