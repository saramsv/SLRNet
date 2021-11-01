import yaml
import argparse
import torch.nn as nn
from semseg.models import ModelBuilder, SegmentationModule
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
    
def parse_model_config(path):
    with open(path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    encoder_path = None
    decoder_path = None

    for p in os.listdir(data['DIR']):
        if "encoder" in p.lower():
            encoder_path = "{}/{}".format(data['DIR'], p)
            continue
        if "decoder" in p.lower():
            decoder_path = "{}/{}".format(data['DIR'], p)
            continue

    if encoder_path==None or decoder_path==None:
        raise("model weights not found")
        
    return data, encoder_path, decoder_path

def load_model_from_cfg(cfg):
    model_config, encoder_path, decoder_path = parse_model_config(cfg)
    net_encoder = ModelBuilder.build_encoder(
        arch = model_config["MODEL"]['arch_encoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        weights = encoder_path)
    net_decoder = ModelBuilder.build_decoder(
        arch = model_config["MODEL"]['arch_decoder'],
        fc_dim = model_config['MODEL']['fc_dim'],
        num_class = model_config['DATASET']['num_class'],
        weights = decoder_path,
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, batch_size=model_config["TRAIN"]["batch_size_per_gpu"])
    return segmentation_module, model_config["MODEL"]['arch_encoder'], model_config["MODEL"]['arch_decoder']


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument("-c", "--cfg", default="config/test_hidden_stage4_V1.yaml",
        metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--gpu", default=0, type=int, metavar='', help="gpu id for evaluation")
    
    args = parser.parse_args()
    
    # Network Builders
    print("parsing {}".format(args.cfg))
    segmentation_module, encoder_type, decoder_type = load_model_from_cfg(args.cfg)
    segmentation_module.eval()
    
    '''
    try: 
        segmentation_module.cuda()
    except:
        pass
    '''
    # if torch.cuda.is_available(): segmentation_module.cuda()
    
    print("\nEncoder: {}".format(encoder_type))
    for name, m in segmentation_module.encoder.named_children(): print("{}- {}".format(" "*3, name))
    
    print("\nDecoder: {}".format(decoder_type))
    for name, m in segmentation_module.decoder.named_children(): print("{}- {}".format(" "*3, name))
    print()