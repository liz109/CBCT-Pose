import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import yaml
import pickle
import matplotlib.pyplot as plt 
import numpy as np
import tqdm
import torch 
import collections

from src.projection import ProjectionData 
from src.pipeline_zero1to3 import Zero1to3StableDiffusionPipeline, CCProjection
from src.utils import dict_to_namespace, compute_SSIM, compute_PSNR
from src.utils import *

from data_generate.generateData import ConeGeometry_special

from catalog import CoordsCatalog

"""
    # --------------- model --------------
    # model_id = "ckpts/naf-1738773130-16000" 
    # model_id = "ckpts/zero123-165000-chest-5500" 
    # model_id = "ckpts/chest_50-1717001594-5000"
    # model_id = "ckpts/chest_50-1717451818-6680"
    # model_id = "ckpts/chest_50-1717772892-7180"   # chest only
    # model_id = "ckpts/naf-1738773130-16000"           

    # --------- option2 --------
    # model_id = 'ckpts/naf-1739409654-12500'       
    model_id = "ckpts/naf-1739473573-15000" 
"""
 

ckpts = {
    # 'chest': 'ckpts/naf-1739473573-15000',
    'chest': 'ckpts/chest_100-1740106713-30000',     # 150
    'jaw': 'ckpts/jaw_100-1739573955-30000',      
    'foot': 'ckpts/foot_100-1739812909-75000',     
    'abdomen': 'ckpts/abdomen_100-1739912100-45000',
}



def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/naf.yaml", type=str,
                        help="Configuration file")
    parser.add_argument("--data", default="data/naf_50/chest_50.pickle", type=str,
                        help="Name of data")
    parser.add_argument("--type", default="val", type=str,
                        help="Sampling type: train/val")
    parser.add_argument("--coords", default="coords2", type=str,
                        help="Coordinates for sampling")
    parser.add_argument("--model_id", default=None, type=str,
                        help="Name of model")
    parser.add_argument("--anatomy", default='chest', type=str,
                        help="Name of model")
    parser.add_argument("--step", default=300, type=int,
                        help="Number of tnference steps")
    parser.add_argument("--n_projs", default="50", type=int,
                        help="Number of projections for sampling (see catalog.py)")
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    with open(args.config, "r") as f:
        cfgs = yaml.load(f, Loader=yaml.Loader)
    cfgs.update(args_dict)

    args = dict_to_namespace(cfgs)
    args.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    args.dtype = torch.float32
    args.seed = 37

    
    # --------------- generator --------------
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # --------------- model --------------
    if args.model_id is None:
        args.model_id = ckpts[args.anatomy]
    model_name = args.model_id.split('/')[-1]
    model = Zero1to3StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=args.dtype)
    model.to(args.device)

    # --------------- coordinates --------------
    # for sampling (add new case in catelog.py)
    Catalog = CoordsCatalog(n_projs=args.n_projs)
    coords = getattr(Catalog, args.coords)

    # --------------- data --------------
    data_name = args.data.split('/')[-1].split('.')[0]
    with open(args.data, "rb") as handle:
        data = pickle.load(handle)

    geo = ConeGeometry_special(data)
    # global_min, global_max = data['val']['projections'].min(), data['val']['projections'].max()

    dataset = ProjectionData(args.data, type=args.type, \
                            size=args.image_size, \
                            coords=coords)
    
    samples = collections.defaultdict(list)
    samples['geo'] = geo



    # --------------- sampling --------------
    samples = collections.defaultdict(list)
    samples['geo'] = geo
    samples['target_idx'] = coords[:, 0].numpy()
    samples['cond_idx'] = coords[:, 1].numpy()

    for idx in tqdm.tqdm(range(len(dataset)), desc=f'Sampling...'):
        sample = dataset.__getitem__(idx)
        
        T = torch.Tensor(sample['T']).unsqueeze(0)
        cond_image = torch.Tensor(sample['cond_image']).unsqueeze(0)    # (b,c=3,h,w)
        
        # input_image = torch.zeros_like(cond_image)        # option1
        input_image = torch.Tensor(sample["target_edge"]).unsqueeze(0)  # option2


        # model(input_imgs, prompt_imgs, poses, ...)
        pred_image = model(input_image, cond_image, T,\
                        num_inference_steps=args.step, \
                        guidance_scale=1.0, \
                        output_type=None,\
                        generator=generator).images[0]
        
        samples['pred_image'].append(pred_image)    # (h,w,c)

        cond_image = cond_image.squeeze().numpy()   # (c,h,w) in [-1,1]
        cond_image = (cond_image[0]+1.0)/2.0        # (h,w) in [0,1]
        samples['cond_image'].append(cond_image)  

        target_image = sample['target_image'].numpy()   # (c,h,w) in [-1,1]
        target_image = (target_image[0]+1.0)/2.0        # (h,w) in [0,1]
        samples['target_image'].append(target_image)  

        samples['cond_angle'].append(sample['cond_angle'])
        samples['target_angle'].append(sample['target_angle'])


    # save
    file = f'{model_name}-{data_name}-{str(args.step)}steps-{args.coords}-{args.type}.pickle'
    print("Saving...", file)
    path = os.path.join('samples', file)
    with open(path, 'wb') as handle:
        pickle.dump(samples, handle) 


if __name__ == '__main__':
    main()


""" 
RUN: change parameters in naf.yaml

python sample.py --config config/naf.yaml \
    --data data/naf_50/chest_50.pickle \
    --type val \
    --coords coords1 \
    --anatomy chest



Bash:
chmod +x run_training.sh
./run_training.sh
"""