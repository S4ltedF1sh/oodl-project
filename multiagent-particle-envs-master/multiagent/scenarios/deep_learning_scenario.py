import numpy as np
from new_env import NewWorld
from multiagent.core import Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, num_landmarks, num_foods, seed=None):
        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
        self.num_landmarks = num_landmarks
        self.num_foods = num_foods

        world = NewWorld()
        world.discrete_action_input = True
        # world.discrete_action = True # set world to be continuous
        world.agents = [Agent()]
        self.move_objects = [Landmark() for i in range(num_landmarks)]
        self.foods = [Landmark() for i in range(num_foods)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.06
            agent.max_speed = 0.4
            # agent.available = True

        for i, obj in enumerate(self.move_objects):
            obj.name = 'object %d' % i
            obj.collide = True
            obj.movable = False #True
            obj.is_obstacle = True
            obj.size = 0.12
            obj.max_speed = 0.05
            obj.available = True

        for i, food in enumerate(self.foods):
            food.name = 'food %d' % i
            food.collide = True
            food.movable = False
            food.is_obstacle = False
            food.size = 0.03
            food.available = True

        # world.landmarks.extend(move_objects)
        # world.landmarks.extend(foods)

        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.speed_limit = 1
        world.time = 750
        world.landmarks = []
        world.landmarks.extend(self.move_objects)
        world.landmarks.extend(self.foods)

        # set color, initial state
        for i, agent in enumerate(world.agents):
            # agent.dead = False
            agent.collected = 0
            agent.accel = 1.0
            agent.color = np.array([0.35, 0.85, 0.35])
        for i, landmark in enumerate(world.landmarks):
            landmark.available = True
            if not landmark.is_obstacle:
                landmark.color = np.array([0.85, 0.35, 0.35])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])

        # set coordination and velocity
        for i, agent in enumerate(world.agents):
            pos = [-0.9] * world.dim_p
            agent.state.p_pos = np.array(pos)
            agent.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            #if not landmark.is_obstacle:
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            #else:
            #    landmark.state.p_pos = np.zeros(world.dim_p)
            #    landmark.state.p_vel = np.zeros(world.dim_p)
            #    landmark.state.p_vel[i % 2] = landmark.max_speed
        for obst in world.obstacles:
            for landmark in world.landmarks:
                while self.is_collision(obst, landmark) and obst is not landmark:
                    obst.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def done(self, agent, world):
        if world.time <= 0:
            #print("time's up")
            return True
        # if agent.dead:
        #    print("dead")
        #    return True
        if agent.collected == self.num_foods:
            print("finished")
            return True
        return False

    def reward(self, agent, world):
        return world.round_reward

    def get_nearest_food(self, agent, world):
        shortest = 100
        pos = np.array([100,100])
        for food in world.available_foods:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - food.state.p_pos)))
            if dist < shortest:
                shortest = dist
                pos = food.state.p_pos
        return pos

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # distances = []

        # for entity in world.landmarks:
        # for entity in world.available_landmarks:
        #    delta_pos = agent.state.p_pos - entity.state.p_pos
        #    dist = np.sqrt(np.sum(np.square(delta_pos))) if entity.available else -1
        #    distances.append(dist)

        positions = []
        positions.extend(agent.state.p_pos)
        #positions.extend(self.get_nearest_food(agent, world))
        for food in world.available_foods:
            positions.extend(food.state.p_pos)
        for i in range(0, self.num_foods - len(world.available_foods)):
            positions.extend([0, 0])
        for obst in world.obstacles:
            positions.extend(obst.state.p_pos)
        #for entity in world.landmarks:
        #    # for entity in world.available_landmarks:
        #    pos = entity.state.p_pos
        #    positions.extend(pos)

        return np.array(positions)