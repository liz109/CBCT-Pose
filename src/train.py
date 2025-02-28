"""
Zero123 train for pose conditioned projection-generation
"""

import logging
import os
import numpy as np
from time import time
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, ConcatDataset

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

from transformers import AutoProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from src.projection import ProjectionData
# from src.projection_aapm import ProjectionData
from src.pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from src.zero1to3 import CCProjection, Zero1to3Wrapper
from src.utils import wandb_utils

"""
Add logging and writer

"""



components = ['vae', 'image_encoder', 'unet', 'cc_projection', 'scheduler']


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        filename=f"{logging_dir}/log.txt",
        level=logging.INFO,
        # format='[\033[34m%(asctime)s\033[0m] %(message)s',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def rgb_to_gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class Trainer(object):
    def __init__(self, args) -> None:
        # Log 
        timestamp = str(int(datetime.now().timestamp()))

        experiment_name = args.exp_name+"-"+timestamp
        args.output_dir = os.path.join(args.output_dir, experiment_name)
        logging_dir = os.path.join(args.logging_dir, experiment_name)
        os.makedirs(logging_dir, exist_ok=True)

        self.writer = SummaryWriter(os.path.join(logging_dir, timestamp))

        if args.report_to == "wandb":
            wandb_utils.initialize(args, exp_name=experiment_name, project_name=args.exp_name)

        logger = create_logger(logging_dir)
        logger.info("***** Initializing *****") 
        logger.info(f"  Resume models from = {args.pretrained_model_name_or_path}")
        logger.info(f"  Conditional image processor from = {args.cond_stage_model_name_or_path}")


        self.logger = logger
        self.args = args
        self.device = args.device
        self.dtype = torch.float32
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed)
        # self.generator = torch.manual_seed(args.seed)   # cpu generator

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
        # clip_img_processor = AutoProcessor.from_pretrained(
        #     args.cond_stage_model_name_or_path
        # )

        self.model = Zero1to3Wrapper(vae=vae, image_encoder=image_encoder,\
                        unet=unet, cc_projection=cc_projection, \
                        scheduler=scheduler,
                        generator=self.generator
                    )
        self.model.freeze_image_encoder()
        self.model.freeze_vae()
        if args.gradient_checkpointing:     # True
            self.model.unet.enable_gradient_checkpointing()
        self.model.to(device=self.device, dtype=self.dtype)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0,1])


        # ------------ Optimizer ------------
        self.optimizer = torch.optim.AdamW([
            {"params":self.model.module.unet.parameters()},
            {"params":self.model.module.cc_projection.parameters(), "lr": 10. * float(args.learning_rate)}],
            lr=float(args.learning_rate),
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=float(args.adam_weight_decay),
            eps=float(args.adam_epsilon),
        )    


        # ---------------- Dataset ----------------
        train_patients_list = sorted([d for d in os.listdir(args.train_data_dir) if 'pickle' in d])
        train_dataset = []
        for patient in train_patients_list:
            dataset = ProjectionData(os.path.join(args.train_data_dir, patient), type="train", size=args.image_size)
            train_dataset.append(dataset)
        self.train_dataset = ConcatDataset(train_dataset)
        self.train_dataloader = DataLoader(self.train_dataset,
                shuffle=True,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,)

        val_patients_list = sorted([d for d in os.listdir(args.val_data_dir) if 'pickle' in d])
        val_dataset = []
        for patient in val_patients_list:
            dataset = ProjectionData(os.path.join(args.val_data_dir, patient), type="val", size=args.image_size)
            val_dataset.append(dataset)
        self.val_dataset = ConcatDataset(val_dataset)
        # self.val_dataloader = DataLoader(self.val_dataset,
        #         shuffle=True,
        #         batch_size=args.val_batch_size,
        #         num_workers=args.dataloader_num_workers,)
        

        # ---------------- Scheduler ----------------
        # Scheduler and math around the number of training steps.
        num_steps_per_epoch = len(self.train_dataloader)     # 500
        if args.overrode_max_train_steps:
            args.max_train_stepss = args.num_train_epochs * num_steps_per_epoch           # 50*2=100

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_stepss,
        )
        




    def train(self):

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")     # 5000
        self.logger.info(f"  Num examples (len data_loader) = {len(self.train_dataloader)}")   # 500
        self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}") # 5
        self.logger.info(f"  Total optimization steps = {self.args.max_train_stepss}") # 250

        # variables for log
        start_time = time()
        log_loss = 0.0
        log_steps = 0
        train_steps = 0     # global    
        progress_bar = tqdm(
            range(0, self.args.max_train_stepss),
            initial=train_steps,
            desc="Steps",
        ) 


        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            # self.model.unet.train()
            # self.model.cc_projection.train()

            for batch in self.train_dataloader:
                    
                # ------------- INPUT -------------
                # 1. prepare image latents: convert images to latent space
                # (b, 3, 256, 256) -> (b, 4, 32, 32) 
                target_image = batch["target_image"].to(device=self.device, dtype=self.dtype)
                target_latents = self.model.module.get_img_latents(target_image, do_classifier_free_guidance=False)
                # # TODO: do_classifier_free_guidance=True?  
                # latents1 (b, 3, 256, 256) -> (3*b, 4, 32, 32)
                # latents1 = get_img_latents(batch["target_image"], device=device, dtype=dtype, do_classifier_free_guidance=True)

                # 2. prepare nosisy latents: (b, 4, 32, 32)
                noisy_latents, noise, timesteps = self.model.module.get_noisy_latents(target_latents)
                timesteps = timesteps.to(device=self.device, dtype=self.dtype)
                
                # 3. prepare conditional/extra latents -> (b, 4, 32, 32)
                cond_image = batch["cond_image"].to(device=self.device, dtype=self.dtype)
                target_edge = batch["target_edge"].to(device=self.device, dtype=self.dtype)
                extra_latents = self.model.module.get_img_latents(target_edge, do_classifier_free_guidance=False)

                # 4. prepare input latents -> (b, 8, 32, 32)
                # TODO: INPUT cat extra_latent: cond_image  / target_edge 
                input_latents = torch.cat([noisy_latents, extra_latents], dim=1)
                
                # 5. preapare clip [img, pose] embedding -> (b, 1, 768)  (visual token)
                pose = batch["T"].to(device=self.device, dtype=self.dtype)
                prompt_embeds = self.model.module.encode_image_with_pose(cond_image, pose, num_images_per_prompt=1)

                # ------------- UNET -------------
                # Predict the noise residual and compute loss -> (b, 4, 32, 32)
                noise_pred = self.model.module.unet(input_latents, timesteps, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

            
                # ------------- LOSS -------------
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")


                # ------------- BACKFORWARD -------------
                self.optimizer.zero_grad()
                loss.backward()
                # gradient clipping, avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()

                # ------------- LOG -------------
                log_loss += loss.detach().item()
                log_steps += 1
                train_steps += 1
                progress_bar.update(1)
                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)


                if train_steps % self.args.log_every_steps == 0 and train_steps > 0:
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    avg_loss = torch.tensor(log_loss / log_steps)
                    self.logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                    if self.args.report_to == "wandb":
                        wandb_utils.log(
                            {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                            step=train_steps
                        )
                    self.writer.add_scalar('Loss/train', avg_loss, train_steps)
                    

                    log_loss = 0.0
                    log_steps = 0
                    start_time = time()


                # if train_steps % self.args.sample_every_steps == 0 and train_steps > 0:
                #     self.logger.info("Generating samples...")
                #     self.validation_sample(
                #         train_steps
                #     )


                if train_steps >= self.args.max_train_stepss:
                        break 
                

            # # Evaluation and Inference for Validation dataset
            # print("\n")
            # self.model.eval()
            # val_loss = 0.0
            # with torch.no_grad():       # to eval unet
            #     for batch in tqdm(self.val_dataloader, desc='Validation: '):
            #         target_image = batch["target_image"].to(device=self.device, dtype=self.dtype)
            #         target_latents = self.model.get_img_latents(target_image, do_classifier_free_guidance=False)

            #         noisy_latents, noise, timesteps = self.model.get_noisy_latents(target_latents)
            #         timesteps = timesteps.to(device=self.device, dtype=self.dtype)
                    
            #         cond_image = batch["cond_image"].to(device=self.device, dtype=self.dtype)
            #         cond_latents = self.model.get_img_latents(cond_image, do_classifier_free_guidance=False)

            #         input_latents = torch.cat([noisy_latents, cond_latents], dim=1)
                    
            #         pose = batch["T"].to(device=self.device, dtype=self.dtype)
            #         prompt_embeds = self.model.encode_image_with_pose(cond_image, pose, num_images_per_prompt=1)

            #         noise_pred = self.model.unet(input_latents, timesteps, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

            #         loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            #         val_loss += loss.item() 


            # # Train/Val epoch loss information 
            # train_loss = train_loss / len(self.train_dataloader)  
            # val_loss = val_loss / len(self.val_dataloader)   
            # print(f'epoch: {epoch+1}/{self.args.num_train_epochs} | train_loss: {train_loss:.3e} | val_loss: {val_loss:.3e}\n')





        # SAVE ckpt: Create the pipeline using the trained modules and SAVE it.
        pipe = Zero1to3StableDiffusionPipeline(
            vae=self.model.module.vae,
            image_encoder=self.model.module.image_encoder,
            unet=self.model.module.unet,
            scheduler=self.model.module.scheduler,
            cc_projection=self.model.module.cc_projection,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )


        # update output_dir
        base = os.path.basename(self.args.pretrained_model_name_or_path).split("-")
        if len(base) == 2:
            initial_train_steps = 0
        else:
            initial_train_steps = int(base[-1])

        self.args.output_dir = "-".join([self.args.output_dir, str(initial_train_steps+train_steps)])
        os.makedirs(self.args.output_dir, exist_ok=True)
        pipe.save_pretrained(self.args.output_dir)

        self.logger.info("Done!")
        wandb.finish()
        self.writer.close()
 


    def validation_sample(self, step):

        pipe = Zero1to3StableDiffusionPipeline(
            vae=self.model.module.vae,
            image_encoder=self.model.module.image_encoder,
            unet=self.model.module.unet,
            cc_projection=self.model.module.cc_projection,
            scheduler=self.model.module.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe.to(self.device)

        images = []
        tests = []
        for i in range(len(self.val_dataset)):
            sample = self.val_dataset[i]
            T = sample['T'].unsqueeze(0)
            cond_image = sample['cond_image'].unsqueeze(0)
            input_image = torch.zeros_like(cond_image)

            with torch.autocast('cuda'):
                image = pipe(input_image, cond_image, T,\
                    num_inference_steps=20, guidance_scale=1.0,\
                    output_type=None,\
                    generator=self.generator).images[0]  # np (h,w,3) in [0,1]

                # image = torch.Tensor(rgb_to_gray(image))  # (h,w)
                tests.append(image)
                if self.args.report_to == 'wandb':
                    images.append(wandb.Image(image, caption=f"sample {i}"))


        np.save('foo.npy', np.array(tests))

        if self.args.report_to == 'wandb':
            wandb_utils.log({f"samples": images, "train_step": step})


        del pipe
        torch.cuda.empty_cache()