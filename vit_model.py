import torch
from torch import nn
from transformers import ViTModel, ViTConfig, ViTPreTrainedModel
from transformers.modeling_outputs import ModelOutput

class ViTForMultitask(ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # MIM head
        self.mim_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride)
        )

        self.init_weights()

    def forward(
        self,
        pixel_values,
        labels=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]  # assuming [CLS] token is at the first position

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        reconstructed_pixel_values = None
        mim_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

            if labels is not None:  # Assuming we also have true pixel values for calculating MIM loss
                mim_loss_fct = nn.MSELoss()
                mim_loss = mim_loss_fct(reconstructed_pixel_values, pixel_values)

        if not return_dict:
            output = (logits, reconstructed_pixel_values) + outputs[2:]
            return ((loss, mim_loss) + output) if loss is not None else output

        return ModelOutput(
            loss=loss,
            mim_loss=mim_loss,
            logits=logits,
            reconstructed_pixel_values=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Note: Config needs to be defined or loaded with the appropriate attributes.
