import torch
import torch.nn.functional as F


def idx_to_selector(idx_tensor, selection_size):
    """
    Convert a labels indices tensor of shape [batch_size] 
    to one-hot encoded tensor [batch_size, selection_size]

    """
    batch_size = idx_tensor.shape[0]
    dummy = torch.arange(selection_size, device=idx_tensor.device).unsqueeze(0).expand(batch_size, -1)
    extended_idx_tensor = idx_tensor.unsqueeze(-1).expand(-1, selection_size)
    return (dummy == extended_idx_tensor).float()

def convert_mask_patch(pixel_values, mask, h_patch, w_patch):
    """
    given pixel values and mask, return masked pixel values
    """
    reshaped_mask = mask.reshape(-1, h_patch, w_patch).unsqueeze(1) # [N, 1, h_patch ,w_patch]
    image_size = pixel_values.shape[-2:]
    reshaped_mask = torch.nn.functional.interpolate(reshaped_mask, size=image_size, mode='nearest')
    return pixel_values * reshaped_mask + (1 - reshaped_mask) * pixel_values.mean(dim=(-1,-2), keepdim=True)