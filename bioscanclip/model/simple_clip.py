import os

import torch.nn.functional as F
import timm
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from bioscanclip.model.mlp import MLPEncoder
from bioscanclip.model.image_encoder import CLIBDImageEncoder
from bioscanclip.model.dna_encoder import load_pre_trained_bioscan_bert, CLIBDDNAEncoder
from bioscanclip.model.language_encoder import load_pre_trained_bert, CLIBDLanguageEncoder
from bioscanclip.util.util import add_lora_layer_to_open_clip, update_checkpoint_param_names, handle_local_ckpt_path
import numpy as np
from typing import Optional
import torch
import open_clip
from bioscanclip.util.util import remove_module_from_state_dict


class SimpleCLIP(nn.Module):
    def __init__(self, image_encoder, dna_encoder, language_encoder, open_clip_model=None, init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None, for_bio_clip=False):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.dna_encoder = dna_encoder
        self.language_encoder = language_encoder
        self.open_clip_model = open_clip_model
        self.tokenizer_for_open_clip = open_clip.get_tokenizer('ViT-B-32') if open_clip_model is not None else None
        if for_bio_clip:
            self.tokenizer_for_open_clip = open_clip.get_tokenizer('hf-hub:imageomics/bioclip') if open_clip_model is not None else None
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        if self.open_clip_model is not None:
            # TODO: find a better way to get the output dimension for different encoders.
            self.dna_projection = nn.Linear(768, 512)

    def forward(self, image_input, dna_input, language_input):
        image_output = None
        dna_output = None
        language_output = None


        if self.dna_encoder is not None:
            dna_output = self.dna_encoder(dna_input)
            if self.open_clip_model is not None:
                dna_output = self.dna_projection(dna_output)
            dna_output = F.normalize(dna_output, p=2, dim=-1)

        if self.open_clip_model is not None:
            if image_input is not None:
                image_features = self.open_clip_model.encode_image(image_input)
                image_output = F.normalize(image_features, p=2, dim=-1)
            if language_input is not None:
                language_input = self.tokenizer_for_open_clip(language_input, context_length=77)
                language_input = language_input.to(image_input.device)
                text_features = self.open_clip_model.encode_text(language_input)
                language_output = F.normalize(text_features, p=2, dim=-1)
        else:
            if self.image_encoder is not None:
                image_output = F.normalize(self.image_encoder(image_input), p=2, dim=-1)
            if self.language_encoder is not None:
                language_output = F.normalize(self.language_encoder(language_input), p=2, dim=-1)
        return image_output, dna_output, language_output, self.logit_scale.exp(), self.logit_bias


def load_vit_for_simclr_training(args, device=None):
    torch.cuda.empty_cache()
    image_model_name = 'vit_base_patch16_224'
    if hasattr(args.model_config.image, 'pre_train_model'):
        image_model_name = args.model_config.image.pre_train_model
    pre_trained_timm_vit = timm.create_model(image_model_name, pretrained=True)
    for param in pre_trained_timm_vit.parameters():
        param.requires_grad = True
    return pre_trained_timm_vit


def wrap_vit_into_simple_clip(args, vit, device=None):
    torch.cuda.empty_cache()
    # Either we have the small models
    image_encoder = None
    dna_encoder = None
    language_encoder = None
    open_clip_model = None

    disable_lora = True

    image_encoder = vit

    model = SimpleCLIP(image_encoder=image_encoder, dna_encoder=dna_encoder,
                       language_encoder=language_encoder, open_clip_model=open_clip_model)

    if device is not None:
        model.to(device)

    if disable_lora:
        for param in model.parameters():
            param.requires_grad = True

    return model


def load_clip_model(args, device=None):
    torch.cuda.empty_cache()
    # Either we have the small models
    image_encoder = None
    dna_encoder = None
    language_encoder = None

    # Or when we use the OpenCLIP model
    open_clip_model = None

    disable_lora = False
    if hasattr(args.model_config, 'disable_lora'):
        disable_lora = args.model_config.disable_lora

    using_open_clip = False
    if hasattr(args.model_config, 'using_open_clip'):
        disable_lora = args.model_config.using_open_clip

    image_model = None
    if hasattr(args.model_config, 'image'):
        if hasattr(args.model_config.image, 'model'):
            image_model = args.model_config.image.model

    language_model = None
    if hasattr(args.model_config, 'language'):
        if hasattr(args.model_config.language, 'model'):
            language_model = args.model_config.language.model

    dna_model = "barcode_bert"
    if hasattr(args.model_config, 'dna'):
        if hasattr(args.model_config.dna, 'model'):
            dna_model = args.model_config.dna.model

    for_bio_clip = False
    if hasattr(args.model_config, 'for_bio_clip'):
        for_bio_clip = args.model_config.for_bio_clip

    if (using_open_clip or (image_model == "clip_image" and language_model == "clip_text")) and not for_bio_clip:
        open_clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L/14', pretrained='commonpool_xl_laion_s13b_b90k')
        open_clip_model.to(device)
        if not disable_lora:
            open_clip_model = add_lora_layer_to_open_clip(open_clip_model, r=4, num_classes=args.model_config.output_dim)
    elif for_bio_clip:
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        open_clip_model = model
        print("Using BioCLIP model")
    else:
        # For image part
        if args.model_config.image.input_type == "image":
            image_model_name = 'vit_base_patch16_224'
            if hasattr(args.model_config.image, 'pre_train_model'):
                image_model_name = args.model_config.image.pre_train_model
            pre_trained_timm_vit = timm.create_model(image_model_name, pretrained=True)
            if hasattr(args.model_config.image, 'image_encoder_trained_with_simclr_style_ckpt_path'):
                image_encoder_trained_with_simclr_style_ckpt_path = args.model_config.image.image_encoder_trained_with_simclr_style_ckpt_path
                checkpoint = torch.load(image_encoder_trained_with_simclr_style_ckpt_path, map_location='cpu')
                state_dict = checkpoint['state_dict']
                state_dict = remove_module_from_state_dict(state_dict)
                pre_trained_timm_vit.load_state_dict(state_dict)

                # Free cuda memory for state_dict
                del state_dict
                del checkpoint
                torch.cuda.empty_cache()
                print("Loaded image encoder from %s" % image_encoder_trained_with_simclr_style_ckpt_path)
            if disable_lora:
                image_encoder = CLIBDImageEncoder(vit_model=pre_trained_timm_vit, r=4,
                                                  num_classes=args.model_config.output_dim, lora_layer=[])
            else:
                image_encoder = CLIBDImageEncoder(vit_model=pre_trained_timm_vit, r=4,
                                                  num_classes=args.model_config.output_dim)
        else:
            image_encoder = MLPEncoder(input_dim=args.model_config.image.input_dim,
                                       hidden_dim=args.model_config.image.hidden_dim,
                                       output_dim=args.model_config.output_dim)

        # For language
        if hasattr(args.model_config, 'language'):
            if args.model_config.language.input_type == "sequence":
                language_model_name = 'prajjwal1/bert-small'
                if hasattr(args.model_config.language, 'pre_train_model'):
                    language_model_name = args.model_config.language.pre_train_model
                if language_model_name == "bert-small" or language_model_name == "bert_small":
                    language_model_name = 'prajjwal1/bert-small'
                _, pre_trained_bert = load_pre_trained_bert(language_model_name)
                if disable_lora:
                    language_encoder = CLIBDLanguageEncoder(model=pre_trained_bert, r=4, num_classes=args.model_config.output_dim,
                                                            lora_layer=[])
                else:
                    language_encoder = CLIBDLanguageEncoder(model=pre_trained_bert, r=4, num_classes=args.model_config.output_dim)
            else:
                raise TypeError(f"Using {args.model_config.language.input_type} as language input is not support yet.")


    # For DNA part
    if hasattr(args.model_config, 'dna'):
        if args.model_config.dna.input_type == "sequence":
            if dna_model == "barcode_bert" or dna_model == "lora_barcode_bert":
                barcode_bert_ckpt = args.bioscan_bert_checkpoint
                if hasattr(args.model_config, 'pre_train_for_barcode_bert') and args.model_config.pre_train_for_barcode_bert == "BIOSCAN-5M":
                    barcode_bert_ckpt = args.bioscan_bert_checkpoint_trained_with_bioscan_5_m
                elif hasattr(args.model_config, 'pre_train_for_barcode_bert') and args.model_config.pre_train_for_barcode_bert == "CANADA-1-5M":
                    barcode_bert_ckpt = args.bioscan_bert_checkpoint_trained_with_canada_1_5_m

                pre_trained_barcode_bert = load_pre_trained_bioscan_bert(
                    bioscan_bert_checkpoint=barcode_bert_ckpt)
                if disable_lora:
                    dna_encoder = CLIBDDNAEncoder(model=pre_trained_barcode_bert, r=4,
                                                  num_classes=args.model_config.output_dim, lora_layer=[])
                else:
                    dna_encoder = CLIBDDNAEncoder(model=pre_trained_barcode_bert, r=4,
                                                  num_classes=args.model_config.output_dim)
        else:
            dna_encoder = MLPEncoder(input_dim=args.model_config.dna.input_dim,
                                     hidden_dim=args.model_config.dna.hidden_dim,
                                     output_dim=args.model_config.output_dim)


    model = SimpleCLIP(image_encoder=image_encoder, dna_encoder=dna_encoder,
                       language_encoder=language_encoder, open_clip_model=open_clip_model, for_bio_clip=for_bio_clip)

    if device is not None:
        model.to(device)

    if disable_lora:
        for param in model.parameters():
            param.requires_grad = True

    # Freeze based on requirements
    if hasattr(args.model_config, 'image') and hasattr(args.model_config.image, 'freeze') and args.model_config.image.freeze:
        if model.image_encoder is not None:
            for param in model.image_encoder.parameters():
                param.requires_grad = False
        elif model.open_clip_model is not None:
            for param in model.open_clip_model.visual.parameters():
                param.requires_grad = False
    if hasattr(args.model_config, 'dna') and hasattr(args.model_config.dna, 'freeze') and args.model_config.dna.freeze:
        if model.dna_encoder is not None:
            for param in model.dna_encoder.parameters():
                param.requires_grad = False
    if hasattr(args.model_config, 'language') and hasattr(args.model_config.language, 'freeze') and args.model_config.language.freeze:
        if model.language_encoder is not None:
            for param in model.language_encoder.parameters():
                param.requires_grad = False
        elif model.open_clip_model is not None:
            for param in model.open_clip_model.transformer.parameters():
                param.requires_grad = False
    return model

def initialize_model_and_load_from_checkpoint(args, device=None):
    print("Initialize model...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_clip_model(args, device)

    local_ckpt_path = handle_local_ckpt_path(args)

    if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
        pass
    elif os.path.exists(local_ckpt_path):
        checkpoint = torch.load(local_ckpt_path, map_location="cuda:0")
        checkpoint = update_checkpoint_param_names(checkpoint)
        model.load_state_dict(checkpoint)
        print(f"Loaded from local: {local_ckpt_path}")
    else:
        try:
            hf_model_name = f"ckpt/bioscan_clip/{args.version}/{args.model_config.dataset}/{args.model_config.model_output_name}/best.pth"
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=args.hf_repo_id,
                    filename=hf_model_name,
                )
            except (OSError, FileNotFoundError):
                raise ValueError(
                    "Checkpoint not found in Hugging Face Hub. Please check the config file"
                )
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            checkpoint = update_checkpoint_param_names(checkpoint)
            model.load_state_dict(checkpoint)
            print(f"Loaded from hf repo: {args.hf_repo_id}/{hf_model_name}")
        except (FileNotFoundError, ValueError, RuntimeError, HfHubHTTPError) as e:
            # No local checkpoint found and no checkpoint in the Hugging Face Hub
            raise ValueError(
                "Neither the local checkpoint nor the huggingface checkpoint was found. Please check the config file")
    return model