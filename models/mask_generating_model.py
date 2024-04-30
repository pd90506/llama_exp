import torch 
import torch.nn as nn
from utils.img_utils import generate_mask
from utils import idx_to_selector
import torch.nn.functional as F
import numpy as np
from models import MLP
import math
from transformers import ViTModel
from typing import List


class SimilarityMeasure(nn.Module):
    def __init__(self, embed_size):
        super(SimilarityMeasure, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, q, k):
        """
        Forward pass of the model.

        Args:
            q (torch.Tensor): Query tensor of shape [N, d].
            k (torch.Tensor): Key tensor of shape [N, L, d].

        Returns:
            torch.Tensor: Similarity tensor of shape [N, L].
        """
        Q = F.normalize(self.query(q), p=2, dim=-1)  # [N, d]
        K = F.normalize(self.key(k), p=2, dim=-1)  # [N, L, d]
        logit_scale = self.logit_scale.exp()

        similarity = torch.matmul(K, Q.unsqueeze(-1)).squeeze(1).squeeze(-1) * logit_scale  # [N, L]

        return similarity  # [N, L]


class MaskGeneratingModel(nn.Module):
    def __init__(self, pred_model:nn.Module, hidden_size, num_classes):
        super().__init__()

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)
        self.pred_model = pred_model
        self.num_classes = num_classes

        self.similarity_measure = SimilarityMeasure(embed_size=hidden_size)

        self.bce_loss = nn.BCELoss(reduction='none')

        self.freeze_params()


    def freeze_params(self):
        """
        Freezes the parameters of the ViT and prediction model.

        This method sets the `requires_grad` attribute of all parameters in the ViT and prediction model to False,
        effectively freezing them and preventing them from being updated during training.

        Returns:
            int: Always returns 0.
        """
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.pred_model.parameters():
            param.requires_grad = False
        return 0


    def loss_func(self, sim:torch.Tensor, mask_list: List[torch.Tensor], prob_list: List[torch.Tensor]):
        """
        Calculates the loss function for the mask generating model.

        Args:
            sim (torch.Tensor): The similarity tensor of shape [N, L], where N is the batch size and L is the length of the sequence.
            mask_list (List[torch.Tensor]): A list of mask tensors of shape [N, L], representing the generated masks at each step.
            prob_list (List[torch.Tensor]): A list of probability tensors of shape [N, 1], representing the probabilities of the generated masks at each step.

        Returns:
            dict: A dictionary containing the loss, reward loss, and mask loss.
                - loss (torch.Tensor): The total loss.
                - reward_loss (torch.Tensor): The reward loss.
                - mask_loss (torch.Tensor): The mask loss.
        """
        n_steps = len(mask_list)
        mask_prob = torch.sigmoid(sim).unsqueeze(1).expand(-1, n_steps, -1) # [N, n_steps, L]
        mask_samples = torch.stack(mask_list, dim=1) # [N, n_steps, L]
        mask_sample_probs = torch.stack(prob_list, dim=1) # [N, n_steps, 1]
        
        # reward loss
        reward_loss = self.bce_loss(mask_prob, mask_samples) # [N, n_steps, L]
        reward_loss = (reward_loss * mask_sample_probs).mean(1) # [N, L]
        reward_loss = reward_loss.mean() # [1]

        # mask_loss
        mask_loss = mask_prob.mean()

        loss = reward_loss + 0.01 * mask_loss
        return {'loss': loss,
                'reward_loss': reward_loss,
                'mask_loss': mask_loss}
    
    def forward(self, x, n_steps=10):
        interpretable_features = self.vit(x)[0][:, 1:, :] # [N, L, d]

        logits = self.pred_model(x).logits # [N, n_classes]
        predicted_class_idx = logits.argmax(-1) # [N, 1]
        predicted_class_selector = idx_to_selector(predicted_class_idx, self.num_classes) # [N, n_classes]
    

        original_feature = self.pred_model.vit(x)[0][:, 0, :] # [N, d]
        
        sim = self.similarity_measure(q=original_feature, k=interpretable_features) # [N, L]

        if n_steps == 0:
            return {
                'sim': sim
            }

        probs_list = []
        mask_list = []

        with torch.no_grad():
            for i in range(n_steps):
                mask = self.generate_mask(sim)
                mask_size = int(math.sqrt(mask.shape[-1]))
                reshaped_mask = mask.reshape(-1, mask_size, mask_size).unsqueeze(1) # [N, 1, size, size]
                H, W = x.shape[-2:]
                unsampled_mask = F.interpolate(reshaped_mask, size=(H, W), mode='nearest') # [N, 1, H, W]
                masked_input = x * unsampled_mask # [N, C, H, W]

                probs = torch.softmax(self.pred_model(masked_input).logits, dim=-1) # [N, n_classes]
                probs = (probs * predicted_class_selector).sum(-1, keepdim=True) # [N, 1]

                probs_list.append(probs)
                mask_list.append(mask)
        
        loss = self.loss_func(sim, mask_list, probs_list)
        return {
            'loss': loss,
            'sim': sim,
            'mask_list': mask_list,
            'probs_list': probs_list
        }

    
    def generate_mask(self, sim):
        """Generate a mask based on the similarity tensor. [generate action based on policy]

        Args:
            sim (Tensor): The similarity tensor of shape [N, L].

        Returns:
            Tensor: The generated mask tensor of shape [N, L].
        """
        mask_prob = torch.sigmoid(sim)

        # sample a mask (action) based on the mask probability (policy)
        mask = torch.bernoulli(mask_prob) # [N, L]
        
        return mask # [N, L]


    def attribute_img(self, 
                      x, 
                      image_size=224, 
                      patch_size=16, 
                      baseline=None, 
                      seed=None):
        """
        Generate attribution heatmap for an input image.

        Args:
            x: An image tensor of shape [N, C, H, W], where H = W = image_size.
            image_size: The size of the input image (H = W = image_size).
            patch_size: The size of each patch. Can be used to calculate the number of tokens in each patch 
                        (image_size // patch_size) ** 2.
            baseline: The baseline tensor. If None, the baseline is set to the zero tensor.
            n_samples: The number of random masks to be generated.
            mask_prob: The probability of a token being masked, i.e., replaced by the baseline.
            seed: The seed value for random number generation.

        Returns:
            Attribution heatmap.
        """

        size = image_size // patch_size
        N, C, H, W = x.shape
        with torch.no_grad():
            outputs = self.forward(x, n_steps=0)

            probs = torch.sigmoid(outputs['sim']).reshape(N, size, size)

        
        return probs
    
    def attribute_text(self, x, baseline=None, n_samples=1000,):
        # TODO
        raise NotImplementedError("This function hasn't been developed.")

