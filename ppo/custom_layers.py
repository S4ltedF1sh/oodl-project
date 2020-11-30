import torch
import torch.nn as nn
import numpy as np


class CustomAttention(nn.Module):
    def __init__(self, attention, embedding, linear):
        super(CustomAttention, self).__init__()
        self.attention = attention
        self.embedding = embedding
        self.flatten = nn.Flatten()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = linear

    def forward(self, x):
        embeded = self.embedding(x)
        attn, weights = self.attention(embeded, embeded)

        avg = self.avg_pool(attn)
        flat = self.flatten(avg)
        w = self.flatten(weights.permute(1, 2, 0))
        self.linear.weight = nn.Parameter(w)
        out = self.linear(flat)

        return out


class CustomInputProcessingLayer(nn.Module):
    def get_objects(self, x, pos, obj_type):
        objects = []
        num_obj = x[pos]
        pos += 1
        for i in range(num_obj):
            obj = [obj_type]
            obj.append(x[pos])
            pos += 1
            obj.append(x[pos])
            pos += 1
            objects.append(obj)
        return objects, pos

    def get_context_vector(self, objects):
        pos_x = [obj[1] for obj in objects]
        pos_y = [obj[2] for obj in objects]
        return [sum(pos_x) / len(pos_x), sum(pos_y) / len(pos_y)]

    def get_objects_context(self, objects, context):
        context_objects = []
        for obj in objects:
            context_objects.append([obj[0], obj[1] - context[0], obj[2] - context[1]] + context)
        return context_objects

    def forward(self, x):
        objects = []
        i = 0
        agents, i = self.get_objects(x, i, 1)
        foods, i = self.get_objects(x, i, 2)
        obstacles, i = self.get_objects(x, i, 3)

        objects.extend(agents)
        objects.extend(foods)
        objects.extend(obstacles)
        context = self.get_context_vector(objects)
        context_objects = self.get_objects_context(objects, context)
        temp = np.array(context_objects) * 1000 + 2000
        out = []
        for t in temp.tolist():
            t[0] = t[0] / 1000 - 2
            out.extend(t)
        return torch.from_numpy(np.array(out).astype(int))


class CustomAvgPooling(nn.Module):
    def forward(self, objects):
        output = []
        for i in range(len(objects[0])):
            col = [obj[i] for obj in objects.tolist()]
            output.append(sum(col) / len(col))
        return torch.from_numpy(np.array(output)).float()
