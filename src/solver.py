"""
Zero123 train for pose conditioned projection-generation
"""

import argparse
import yaml

import logging
import os
from contextlib import nullcontext
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from src.projection import ProjectionData
from src.pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from src.zero1to3 import CCProjection, Zero1to3Wrapper
from utils import dict_to_namespace



logger = logging.getLogger(__name__)

components = ['vae', 'image_encoder', 'unet', 'cc_projection', 'scheduler']



class Solver(object):
    def __init__(self, args) -> None:

        logger.info("***** Initializing *****") 
        logger.info(f"  Resume models from = {args.pretrained_model_name_or_path}")
        logger.info(f"  Conditional image processor from = {args.cond_stage_model_name_or_path}") 


        self.args = args
        self.device = args.device
        self.dtype = torch.float32


        # ------------ Model: load processor, models and create wrapper ------------
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="vae"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="image_encoder"
        )

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="unet", 
            fast_load=False
        )
        cc_projection = CCProjection.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="cc_projection" 
        )
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )   # noise scheduler
        clip_img_processor = AutoProcessor.from_pretrained(
            args.cond_stage_model_name_or_path
        )

        self.model = Zero1to3Wrapper(vae=vae, image_encoder=image_encoder,\
                        unet=unet, cc_projection=cc_projection, \
                        scheduler=scheduler,
                    )
        self.model.freeze_image_encoder()
        self.model.freeze_vae()
        if args.gradient_checkpointing:     # True why??????
            self.model.unet.enable_gradient_checkpointing()
        self.model.to(device=self.device, dtype=self.dtype)
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=[0,1])


        # ------------ Optimizer ------------
        self.optimizer = torch.optim.AdamW([
            {"params":self.model.unet.parameters()},
            {"params":self.model.cc_projection.parameters(), "lr": 10. * float(args.learning_rate)}],
            lr=float(args.learning_rate),
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=float(args.adam_weight_decay),
            eps=float(args.adam_epsilon),
        )    


        # ---------------- Dataset ----------------
        dataset = ProjectionData(args.train_data_dir, type="train", size=args.image_size)
        self.train_dataloader = DataLoader(dataset,
                shuffle=True,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,)

        dataset = ProjectionData(args.train_data_dir, type="val", size=args.image_size)
        self.val_dataloader = DataLoader(dataset,
                shuffle=True,
                batch_size=args.val_batch_size,
                num_workers=args.dataloader_num_workers,)
        

        # ---------------- Scheduler ----------------
        # Scheduler and math around the number of training steps.
        num_steps_per_epoch = len(self.train_dataloader)     # 500
        if args.overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_steps_per_epoch           # 50*2=100

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        

        # ---------------- Checkpoint ----------------
        self.ckpt_dict = {
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'epoch': 0,
            'loss': float('inf')
        }
    


    def train(self):
        # TODO: log
        timestamp = str(int(datetime.now().timestamp()))
        writer = SummaryWriter(os.path.join(self.args.logging_dir, timestamp))

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")     # 5000
        logger.info(f"  Num examples (len data_loader) = {len(self.train_dataloader)}")   # 500
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}") # 5
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}") # 250


        global_step = 0
        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=global_step,
            desc="Steps",
        ) 


        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            self.model.unet.train()
            self.model.cc_projection.train()
            train_loss = 0.0

            for batch in self.train_dataloader:
                    
                # ------------- INPUT -------------
                # 1. prepare image latents: convert images to latent space
                # (b, 3, 256, 256) -> (b, 4, 32, 32) 
                target_image = batch["target_image"].to(device=self.device, dtype=self.dtype)
                target_latents = self.model.get_img_latents(target_image, do_classifier_free_guidance=False)
                # latents = latents * self.model.vae.config.scaling_factor   # TODO: why
                # # TODO: do_classifier_free_guidance=True?  
                # latents1 (b, 3, 256, 256) -> (3*b, 4, 32, 32)
                # latents1 = get_img_latents(batch["target_image"], device=device, dtype=dtype, do_classifier_free_guidance=True)

                # 2. prepare nosisy latents: (b, 4, 32, 32)
                noisy_latents, noise, timesteps = self.model.get_noisy_latents(target_latents)
                timesteps = timesteps.to(device=self.device, dtype=self.dtype)
                # # TODO: pipeline_zero1to3 line 750
                # latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
                
                # 3. prepare conditional latents -> (b, 4, 32, 32)
                cond_image = batch["cond_image"].to(device=self.device, dtype=self.dtype)
                cond_latents = self.model.get_img_latents(cond_image, do_classifier_free_guidance=False)
                # TODO: cat cond_image or cond_latents
                # 4. prepare input latents -> (b, 8, 32, 32)
                input_latents = torch.cat([noisy_latents, cond_latents], dim=1)
                
                # 5. preapare clip [img, pose] embedding -> (b, 1, 768)  
                pose = batch["T"].to(device=self.device, dtype=self.dtype)
                prompt_embeds = self.model.encode_image_with_pose(cond_image, pose, num_images_per_prompt=1)
                                                        # cond_image or target_image?????

                # ------------- UNET -------------
                # Predict the noise residual and compute loss -> (b, 4, 32, 32)
                noise_pred = self.model.unet(input_latents, timesteps, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

            
                # ------------- LOSS -------------
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                train_loss += loss.item() 


                # ------------- BACKFORWARD -------------
                self.optimizer.zero_grad()
                loss.backward()
                # gradient clipping, avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()

                global_step += 1
                progress_bar.update(1)
                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= self.args.max_train_steps:
                    break    



            # Evaluation and Inference for Validation dataset
            print("\n")
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():       # to eval unet
                for batch in tqdm(self.val_dataloader, desc='Validation: '):
                    target_image = batch["target_image"].to(device=self.device, dtype=self.dtype)
                    target_latents = self.model.get_img_latents(target_image, do_classifier_free_guidance=False)

                    noisy_latents, noise, timesteps = self.model.get_noisy_latents(target_latents)
                    timesteps = timesteps.to(device=self.device, dtype=self.dtype)
                    
                    cond_image = batch["cond_image"].to(device=self.device, dtype=self.dtype)
                    cond_latents = self.model.get_img_latents(cond_image, do_classifier_free_guidance=False)

                    input_latents = torch.cat([noisy_latents, cond_latents], dim=1)
                    
                    pose = batch["T"].to(device=self.device, dtype=self.dtype)
                    prompt_embeds = self.model.encode_image_with_pose(cond_image, pose, num_images_per_prompt=1)

                    noise_pred = self.model.unet(input_latents, timesteps, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    val_loss += loss.item() 


            # Train/Val epoch loss information 
            train_loss = train_loss / len(self.train_dataloader)  
            val_loss = val_loss / len(self.val_dataloader)   
            print(f'epoch: {epoch+1}/{self.args.num_train_epochs} | train_loss: {train_loss:.3e} | val_loss: {val_loss:.3e}\n')

            # # Inference val datasets
            # if (epoch+1) % self.args.validation_epochs == 0:    
            #     self.inference_validation(
            #         self.model.vae,
            #         self.model.image_encoder,
            #         self.model.unet,
            #         self.model.cc_projection,
            #         global_step
            #     )



        # Final: Create the pipeline using the trained modules and SAVE it.
        pipe = Zero1to3StableDiffusionPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            vae=self.model.vae,
            image_encoder=self.model.image_encoder,
            unet=self.model.unet,
            cc_projection=self.model.cc_projection,
            scheduler=self.model.scheduler
        )

        # update output_dir
        base = os.path.basename(self.args.pretrained_model_name_or_path).split("-")
        if len(base) == 4:
            initial_global_step = int(base[-1])
        else:
            initial_global_step = 0
        self.args.output_dir = "-".join([self.args.output_dir, str(initial_global_step+global_step)])
        os.makedirs(self.args.output_dir, exist_ok=True)
        # save model
        pipe.save_pretrained(self.args.output_dir)

 

    def inference_validation(self, vae, image_encoder, unet, cc_projection, step):
        logger.info("Running inference validation... ")

        # # TODO:
        # pipe = Zero1to3StableDiffusionPipeline.from_pretrained(
        #     self.args.pretrained_model_name_or_path,
        #     vae=self.model.vae,
        #     image_encoder=self.model.image_encoder,
        #     unet=self.model.unet,
        #     cc_projection=self.model.cc_projection,
        #     scheduler=self.model.scheduler
        # )
        # pipe.to(self.device)

        # del pipe
        # torch.cuda.empty_cache()