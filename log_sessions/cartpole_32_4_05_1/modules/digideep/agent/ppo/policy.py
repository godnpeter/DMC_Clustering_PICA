"""This module is highly inspired by `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>`__.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .policy_utils import Bernoulli, Categorical, DiagGaussian
from .policy_utils import init_easy, init_rnn
from .policy_utils import MLPBlock, RNNBlock, CNNBlock, ProjectionLayer, MLPBlock_ReLU

from digideep.agent.policy_base import PolicyBase

from digideep.utility.toolbox import get_class #, get_module
from digideep.utility.logging import logger

class Policy(PolicyBase):
    """The stochastic policy to be used with PPO algorithm. This policy supports three different action
    distributions:

    * ``Categorical``: For ``gym.spaces.Discrete`` action spaces.
    * ``DiagGaussian``: For ``gym.spaces.Box`` action spaces.
    * ``Bernoulli``: For ``gym.spaces.MultiBinary`` action spaces.

    Args:
        device: The device for the PyTorch computations. Either of CPU or GPU.
        obs_space: Observation space of the environment.
        act_space: Action space of the environment.
        modelname (str): The model to be used within the policy. CURRENTLY THIS OPTION IS NOT USED INSIDE THE CLASS
            AND MODEL IS DECIDED BY THE SHAPE OF OBSERVATION SPACE.
        modelargs (dict): A dictionary of arguments for the model.
    
    """
    def __init__(self, device, obs_space, act_space, modelname, modelargs):
        super(Policy, self).__init__(device)

        self.recurrent = modelargs["recurrent"]
        #######
        # modelclass = get_class(modelname)
        #######
        if len(obs_space["dim"]) == 3: # It means we have images as observations
            # obs_space["dim"][0] is the channels of the input
            self.model["base"] = CNNModel(num_inputs=obs_space["dim"][0], **modelargs)
        elif len(obs_space["dim"]) == 1: # It means we have vectors as observation
            self.model["base"] = MLPModel(num_inputs=obs_space["dim"][0], **modelargs)
        else:
            raise NotImplementedError

        # TODO: For discrete actions, `act_space["dim"][0]` works. It works for constinuous actions as well.
        #       Even for discrete actions `np.isscalar(act_space["dim"])` returns False.
        num_outputs = act_space["dim"] if np.isscalar(act_space["dim"]) else act_space["dim"][0]
        # num_outputs = act_space["dim"].item() if len(act_space["dim"].shape)==0 else act_space["dim"][0]
        if act_space["typ"] == "Discrete":
            print("Discrete is recognized and num_outputs=", num_outputs)
            self.model["dist"] = Categorical(num_inputs=modelargs["output_size"], num_outputs=num_outputs)
        elif act_space["typ"] == "Box":
            self.model["dist"] = DiagGaussian(num_inputs=modelargs["output_size"], num_outputs=num_outputs)
        elif act_space["typ"] == "MultiBinary":
            # TODO: Is the following necessary?
            num_outputs = act_space["dim"][0]
            self.model["dist"] = Bernoulli(num_inputs=modelargs["output_size"], num_outputs=num_outputs)
        else:
            raise NotImplementedError("The action_space of the environment is not supported!")
        
        self.model_to_gpu()
        logger("Number of parameters:\n>>>>>>", self.count_parameters())


    def generate_actions(self, inputs, hidden, masks, deterministic=False):
        """This function is used by :func:`~digideep.agent.ppo.PPO.action_generator` to generate the actions while simulating in the environments.
        
        Args:
            inputs: The observations.
            hidden: The hidden states of the policy models.
            masks: The masks indicates the status of the environment in the last state, either it was finished ``0``, or still continues ``1``.
            deterministic (bool): The flag indicationg whether to sample from the action distribution (when ``False``) or choose the best (when ``True``).
        """
        with torch.no_grad():
            self.model.eval()
            value, actor_features, hidden_, op, hp, hidden_action = self.model["base"](inputs, hidden, masks)
            dist = self.model["dist"](actor_features)
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            action_log_probs = dist.log_probs(action)
            # dist_entropy = dist.entropy().mean() # NOTE: Why do we calculate it here? Probably it's not a big deal!
            # return value.detach(), action.detach(), action_log_probs.detach(), hidden.detach()

            storage_item = (inputs, op, hp, action, hidden_action)
            self.model.train()
            return value, action, action_log_probs, hidden_, storage_item

    def evaluate_actions(self, inputs, hidden, masks, action):
        """Evaluates a given action in the PPO method.

        Args:
            inputs: The observations.
            hidden: The hidden states of the policy models.
            masks: The masks indicates the status of the environment in the last state, either it was finished ``0``,
              or still continues ``1``.
            action: The actions to be evaluated with the current policy model.
        """

        value, actor_features, hidden, op, hp, _ = self.model["base"](inputs, hidden, masks)
        dist = self.model["dist"](actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, hidden, op, hp

    @property
    def is_recurrent(self):
        """Indicates whether or not the policy uses a recurrent policy in its base method.
        """
        return self.recurrent


##############################################################
########################### MODELS ###########################
##############################################################
class MLPModel(nn.Module):
    """An MLP model of an actor-critic. It may use a recurrent unit or not.
    
    Args:
        num_inputs: The dimension of the input observation.
        output_size: The dimension of the output action feature vector.
        recurrent (bool): An indicator whether or not a base recurrent module.
    """

    def __init__(self, num_inputs, output_size, recurrent=False):
        super(MLPModel, self).__init__()

        o_num_clustering = 32
        h_num_clustering = 4
        #self.resultprint = False
        #self.t = 0
        self.recurrent = recurrent
        self.embedding = MLPBlock(num_inputs=num_inputs, output_size=output_size)
        self.obsprojection = ProjectionLayer(num_inputs = output_size, output_size = o_num_clustering)
        
        if recurrent:
            self.rnnblock = RNNBlock(num_inputs=o_num_clustering, hidden_size=output_size)
        
        self.hiddenprojection = ProjectionLayer(num_inputs = output_size, output_size = h_num_clustering)
        
        concat_size = num_inputs + h_num_clustering
        self.D2C_layer = MLPBlock_ReLU(num_inputs = concat_size, output_size = output_size)
        
        self.actor  = MLPBlock(num_inputs=output_size, output_size=output_size)
        self.critic = MLPBlock(num_inputs=output_size, output_size=output_size)

        
        init_ = init_easy(gain=np.sqrt(2), bias=0)
        #self.resize_layer = init_(nn.Linear(o_num_clustering, output_size))
        self.critic_linear = init_(nn.Linear(output_size, 1))
    
    def forward(self, inputs, hidden, masks):
        """
        Args:
            inputs (:obj:`torch.Tensor`): The input to the model that includes the observations.
            hidden (:obj:`torch.Tensor`): The hidden state of last step used in recurrent policies.
            masks (:obj:`torch.Tensor`): The mask indicator to be used with recurrent policies.
        
        Returns:
            tuple: A tuple of ``(values, feature_actor, hidden)``, which are action-value, the features that
            will form the action distribution probability, and the hidden state of the recurrent unit.
        """
        x = inputs
        x = self.embedding(x)
        ##pica loss 1
        op = self.obsprojection(x)
        
        x = op
        
        if self.recurrent:
            x, hidden = self.rnnblock(x, hidden, masks)
        ##pica loss 2
        hidden_action = x
        hp = self.hiddenprojection(x)
        
        #batch_size = hp.size(0)
        #rand = torch.rand(batch_size, 4)#.cuda()
        #rand_exp = torch.exp(rand)
        #rand_prop = rand_exp / torch.sum(rand_exp)
        #rand_prop = rand_prop.cuda()
        #hp = rand_prop
        
        ##concat
        ohp = torch.cat((inputs, hp), dim=1)
        ##d2c
        ohp = self.D2C_layer(ohp)
        #ohp = self.D2C_layer(x)
        
        hidden_critic = self.critic(ohp)
        feature_actor = self.actor(ohp)

        values = self.critic_linear(hidden_critic)
        
        #if self.resultprint == True:
            #print("obs_prop, obs_idx : {:.2f}, {} , hidden_prop, hidden_idx : {:.2f}, {}" .format(op.max(), op.argmax(), hp.max(), hp.argmax()))
            #if self.t == 950:
                #import pdb
                #pdb.set_trace()

            #self.t = self.t +1
        
        return values, feature_actor, hidden, op, hp, hidden_action

