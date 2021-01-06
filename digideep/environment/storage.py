import torch
import numpy as np


class Storage:

    def __init__(self, obs_cluster_num, action_cluster_num, num_steps=10000):
        self.input_batch = torch.zeros(num_steps, 24)
        self.obs_cluster_batch = torch.zeros(num_steps, obs_cluster_num)
        self.action_cluster_batch = torch.zeros(num_steps, action_cluster_num)
        self.action_batch = torch.zeros(num_steps, 6)
        self.hidden_action_batch = torch.zeros(num_steps, 64)
        #self.image_batch = np.zeros((num_steps, 240, 320, 3))
        self.image_step = 0
        self.action_idx = -1

        self.step = 0
        self.num_steps = num_steps
        self.flag = 0

    def store(self, input, obs_cluster, action_cluster, action, hidden_action):
        """
        Store the given SARS transition objective
        :return:
        """
        self.input_batch[self.step] = input.cpu()
        self.action_batch[self.step] = action.cpu()
        self.obs_cluster_batch[self.step] = obs_cluster.cpu()
        self.action_cluster_batch[self.step] = action_cluster.cpu()
        self.hidden_action_batch[self.step] = hidden_action.cpu()
        self.action = action

        """
        if self.step > 50:
            if self.action_idx != action_cluster.argmax():
                be = self.action_idx
                af = action_cluster.argmax().item()
                self.action_idx = action_cluster.argmax().item()
                idx = obs_cluster.argmax()
                print("="*20)
                print("AFTER: ", af, "  :::::: ", action)
                print("AFTER ACTION: ", action)
                import pdb
                pdb.set_trace()
        """
        if (self.step % (self.num_steps - 1) == 0) and (self.step != 0):
            self.flag = 1
            self.step = 0
        else:
            self.step += 1

    def dump(self):

        self.step = self.step - 1000