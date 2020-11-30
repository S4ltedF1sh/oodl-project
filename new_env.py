import numpy as np
from gym import spaces
from multiagent.multi_discrete import MultiDiscrete
from multiagent.core import World
from multiagent.environment import MultiAgentEnv

REWARD_NORMAL_STEP = -0.01
REWARD_HIT_OBSTACLE = 0
REWARD_FOOD_COLLECT = 1

class NewWorld(World):
    # rewrite the step function for the new world
    def __init__(self):
        World.__init__(self)
        self.damping = 0
        self.speed_limit = 1
        self.time = 750
        self.round_reward = REWARD_NORMAL_STEP

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def apply_push_back(self, agent, obst):
        dist = agent.state.p_pos - obst.state.p_pos
        push_back = np.array([(agent.size + obst.size) / np.sqrt(2)] * self.dim_p) + 0.01
        for i, d in enumerate(dist):
            push_back[i] *= 1 if d > 0 else -1
        agent.state.p_pos = obst.state.p_pos + push_back

    @property
    def available_landmarks(self):
        return [landmark for landmark in self.landmarks if landmark.available]

    @property
    def obstacles(self):
        return [landmark for landmark in self.landmarks if landmark.is_obstacle]

    @property
    def available_foods(self):
        return [landmark for landmark in self.landmarks if (not landmark.is_obstacle and landmark.available)]

    def step(self):
        self.round_reward = REWARD_NORMAL_STEP
        self.time = self.time - 1
        p_force = [None] * len(self.entities)
        p_force = self.apply_action_force(p_force)
        self.integrate_state(p_force)

        for i, landmark in enumerate(self.landmarks):
            for agent in self.agents:
                if self.is_collision(agent, landmark):
                    if landmark.is_obstacle:
                        self.round_reward += REWARD_HIT_OBSTACLE
                        self.apply_push_back(agent, landmark)
                    elif landmark.available:
                        self.round_reward += REWARD_FOOD_COLLECT
                        agent.collected += 1
                        landmark.color = np.array([0.85, 0.35, 1])
                        landmark.available = False

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt
            for i, pos in enumerate(entity.state.p_pos):
                if pos > 1.0 or pos < -1.0:
                    entity.state.p_pos[i] = max(-1.0, min(1.0, pos))
                    entity.state.p_vel[i] *= -1


class NewEnv(MultiAgentEnv):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        #MultiAgentEnv.__init__(self,world)
        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym multiagent-particle-envs-master property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # environment parameters
        self.discrete_action_space = False  # True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces

        self.action_space = spaces.Box(np.array([-1, -1]), np.array([+1, +1]))
        # self.action_space = spaces.Discrete(5)

        obs_dim = len(observation_callback(self.agents[0], self.world))
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)

        for agent in self.agents:
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            # self._set_action(action_n[i], agent, self.action_space[i])
            self._set_action(action_n, agent, self.action_space)
        # advance world state
        self.world.step()
        obs_n = self._get_obs(self.agents[0])
        reward_n = self._get_reward(self.agents[0])
        done_n = self._get_done(self.agents[0])

        info_n['n'].append(self._get_info(self.agents[0]))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)

        return np.array(obs_n), float(reward), done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = self._get_obs(self.agents[0])
        return np.array(obs_n)

    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    # speed_limit = self.world.speed_limit
                    # for i,act_p in enumerate(action[0]):
                    #  action[0][i] = max(-1.0*speed_limit, min(speed_limit,act_p))
                    # print(speed_limit, action)
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                #----set new color when the food is eaten----- Minh Vu
                self.render_geoms[e].set_color(*entity.color)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results
