import torch
import torch.nn as nn
import kornia

from transformers import AutoProcessor, CLIPVisionModelWithProjection

from diffusers import DDIMScheduler, AutoencoderKL, DiffusionPipeline, StableDiffusionMixin, UNet2DConditionModel
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin



class CCProjection(ModelMixin, ConfigMixin):
    def __init__(self, in_channel=772, out_channel=768):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.projection = torch.nn.Linear(in_channel, out_channel)

    def forward(self, x):
        return self.projection(x)



class Zero1to3Wrapper(nn.Module):
    """Simple wrapper module for Stabel Diffusion that holds all the models together"""

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel,
        cc_projection: CCProjection,
        scheduler: DDIMScheduler,
        generator = None
        # clip_img_processor: AutoProcessor,
    ):
        super().__init__()
        
        self.vae = vae
        self.image_encoder = image_encoder
        self.unet = unet
        self.cc_projection = cc_projection
        self.scheduler = scheduler
        self.generator = generator
        self.clip_img_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        


    def CLIP_preprocess(self, x):
        if isinstance(x, torch.Tensor):
            if x.min() < -1.0 or x.max() > 1.0:
                raise ValueError("Expected input tensor to have values in the range [-1, 1]")
        x = (x + 1.0) / 2.0
        x = self.clip_img_processor(images=x, return_tensors="pt", do_rescale=False).pixel_values
        return x
      

    def _check_image(self, image):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"`image` has to be of type `torch.Tensor` but is {type(image)}")
        
        assert image.ndim == 4, "image must have 4 dimensions (b, c, h, w)"
        assert image.shape[1] == 3,  "image outside a batch should be of shape (3, h, w)"
        
        if image.min() < -1 or image.max() > 1:
            raise ValueError("image should be in [-1, 1] range")


    # (b, 3, 256, 256) -> (b, 4, 32, 32)
    def get_img_latents(self, image, do_classifier_free_guidance=False):
        self._check_image(image)
        
        latents = self.vae.encode(image).latent_dist.mode()
        latents = self.vae.config.scaling_factor * latents 

        latents = (
            torch.cat([torch.zeros_like(latents), latents]) if do_classifier_free_guidance else latents
        )   # TODO not sure

        return latents

    # def prepare_latents(self, bsz):
        

    # -> (b, 4, 32, 32)
    def get_noisy_latents(self, latents):
        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.size(), generator=self.generator, device=latents.device, dtype=latents.dtype)
        
        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), generator=self.generator, device=latents.device)
        timesteps = timesteps.long()    # 1000

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        return noisy_latents, noise, timesteps


    def _encode_image(self, image, num_images_per_prompt, do_classifier_free_guidance=False):
        self._check_image(image)
        device, dtype = image.device, image.dtype

        # image = (image + 1.0) / 2.0 
        # image = self.clip_img_processor(images=image, return_tensors="pt", do_rescale=False).pixel_values
        image = self.CLIP_preprocess(image)
        image = image.to(device=device, dtype=dtype)   # [b, 3, 224, 224]
        
        image_embeddings = self.image_encoder(image).image_embeds        # [b, 768]
        image_embeddings = image_embeddings.unsqueeze(1)            # [b, 1, 768]

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings
    

    def _encode_pose(self, pose, num_images_per_prompt, do_classifier_free_guidance=False):
        if not isinstance(pose, torch.Tensor):
            raise ValueError(f"`pose` has to be of type `torch.Tensor` but is {type(pose)}")
        
        # pose_embeddings = pose.unsqueeze(1)  # [b, 1, 4]
        x, y, z = pose[:, 0].unsqueeze(1), pose[:, 1].unsqueeze(1), pose[:, 2].unsqueeze(1)
        pose_embeddings = (
            torch.cat([x, torch.sin(y), torch.cos(y), z], dim=-1)
            .unsqueeze(1)
        )  # B, 1, 4

        # duplicate pose embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = pose_embeddings.shape
        pose_embeddings = pose_embeddings.repeat(1, num_images_per_prompt, 1)
        pose_embeddings = pose_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(pose_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            pose_embeddings = torch.cat([negative_prompt_embeds, pose_embeddings])
        return pose_embeddings


    # prompt_embeds: get clip [img, pose] embedding -> (10, 1, 768)
    def encode_image_with_pose(self, image, pose, num_images_per_prompt, do_classifier_free_guidance=False):
        img_prompt_embeds = self._encode_image(image, num_images_per_prompt)    # cond img: [b, 1, 768]
        pose_prompt_embeds = self._encode_pose(pose, num_images_per_prompt)
        
        prompt_embeds = torch.cat([img_prompt_embeds, pose_prompt_embeds], dim=-1)
        prompt_embeds = self.cc_projection(prompt_embeds)

        if do_classifier_free_guidance:
            negative_prompt = torch.zeros_like(prompt_embeds)
            prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
            
        return prompt_embeds
    

    def freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad = False

    def freeze_unet(self):
        for param in self.unet.parameters():
            param.requires_grad = False

    def freeze_cc_projection(self):
        for param in self.cc_projection.parameters():
            param.requires_grad = False