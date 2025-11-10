import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
import timm
import os

#support for resnet-18, resnet-34, alexnet. Others yet to be done
#support for resnext and wide resnet added
#removed resizing of images

class Img2Vec():
    def __init__(self, cuda = False, model = "resnet-18", layer = "default", layer_output_size = 512, fine_tuned_path=None):
        
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model
        
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer, fine_tuned_path)
        
        if fine_tuned_path and os.path.exists(fine_tuned_path):
            # Load checkpoint and try to extract a state_dict in common formats.
            # Try to load checkpoint using weights_only=True when supported by torch.load
            # This mitigates pickle-based code execution risks for untrusted checkpoints.
            try:
                ckpt = torch.load(fine_tuned_path, map_location=self.device, weights_only=True)
            except Exception as e:
                # weights_only load failed (unsupported objects or older torch)
                print(f"Warning: weights_only load failed: {e}\nFalling back to standard torch.load (this may execute pickled code).")
                ckpt = torch.load(fine_tuned_path, map_location=self.device)

            # If checkpoint is a dict, try common keys first
            state_dict = None
            if isinstance(ckpt, dict):
                # Try common wrapper keys used when saving checkpoints
                wrapper_keys = ('state_dict', 'model', 'backbone', 'net', 'state_dicts', 'model_state_dict', 'state_dict_ema', 'checkpoint')
                for key in wrapper_keys:
                    if key in ckpt and isinstance(ckpt[key], dict):
                        candidate = ckpt[key]
                        # Some wrappers nest another 'state_dict' inside
                        if 'state_dict' in candidate and isinstance(candidate['state_dict'], dict):
                            state_dict = candidate['state_dict']
                        else:
                            state_dict = candidate
                        break
                # if not found, maybe the dict itself is a state_dict-like mapping
                if state_dict is None:
                    # Heuristic: if most values are tensors, treat as state_dict
                    try:
                        sample_vals = list(ckpt.values())[:5]
                        if all(hasattr(v, 'shape') for v in sample_vals):
                            state_dict = ckpt
                    except Exception:
                        state_dict = None
            else:
                # ckpt is likely already a state_dict
                state_dict = ckpt

            if state_dict is None:
                raise RuntimeError(f"Could not find a state_dict inside checkpoint: {fine_tuned_path}")

            # Try loading directly; if keys are prefixed (module., backbone.) strip them
            def _strip_prefix(sd, prefixes=('module.', 'backbone.', 'net.')):
                new_sd = {}
                for k, v in sd.items():
                    new_k = k
                    for p in prefixes:
                        if new_k.startswith(p):
                            new_k = new_k[len(p):]
                    new_sd[new_k] = v
                return new_sd

            # First, try an exact load (strict)
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e_exact:
                # Prepare a cleaned state_dict that matches the model keys:
                # - strip common prefixes like 'module.', 'backbone.', 'net.'
                # - discard keys that don't exist in the model (e.g., projection heads)
                stripped = _strip_prefix(state_dict)

                # Also attempt to remove a single leading 'backbone.' or 'module.backbone.'
                def _normalize_key(k):
                    if k.startswith('module.backbone.'):
                        return k[len('module.backbone.'):]
                    if k.startswith('backbone.'):
                        return k[len('backbone.'):]
                    if k.startswith('module.'):
                        return k[len('module.'):]
                    return k

                model_keys = set(self.model.state_dict().keys())
                compatible = {}
                for k, v in stripped.items():
                    norm_k = _normalize_key(k)
                    # If normalized key matches a model key, include it
                    if norm_k in model_keys:
                        compatible[norm_k] = v

                if not compatible:
                    # Nothing matched — surface the original error
                    raise RuntimeError(f"Could not find any matching parameter keys for model when loading {fine_tuned_path}: {e_exact}") from e_exact

                # Load filtered dict with strict=False so missing keys in the checkpoint or
                # extra keys in the checkpoint are ignored. This lets us ignore projection
                # heads or other training-only modules that aren't present in the base model.
                try:
                    load_res = self.model.load_state_dict(compatible, strict=False)
                    # load_state_dict returns a NamedTuple with missing_keys/unexpected_keys when strict=False
                    missing = getattr(load_res, 'missing_keys', None)
                    unexpected = getattr(load_res, 'unexpected_keys', None)
                    if missing:
                        print(f"Warning: when loading weights from {fine_tuned_path}, the following keys were missing in the checkpoint for the model (will use model defaults): {missing}")
                    if unexpected:
                        # usually empty since we filtered to model keys, but log if present
                        print(f"Note: unexpected keys present when loading checkpoint: {unexpected}")
                except Exception as e2:
                    # If that still fails, give a helpful error message
                    raise RuntimeError(f"Failed to load compatible subset of weights from {fine_tuned_path}: {e2}") from e2
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
    def get_vec(self, img, tensor = False, normalize_vec: bool = False):
        """Return embedding for PIL image.

        Parameters
        - tensor: if True, return a torch tensor; otherwise a numpy array
        - normalize_vec: if True, return an L2-normalized embedding (unit length)
        """
        image = self.normalize(self.to_tensor(img)).unsqueeze(0).to(self.device)

        if self.model_name == 'alexnet':
            my_embedding = torch.zeros(1, self.layer_output_size)
        else:
            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        if self.model_name in ['resnet50d']:
            # For timm models with head=Identity, directly get embedding
            embedding = self.model(image)
            if tensor:
                out = embedding
            else:
                out = embedding.detach().cpu().numpy()[0, :, 0, 0] if len(embedding.shape) == 4 else embedding.detach().cpu().numpy()[0]

            if normalize_vec:
                if tensor:
                    # normalize torch tensor along feature dim
                    flat = out.view(out.size(0), -1)
                    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    out = out / norm.view(-1, 1, 1, 1) if out.dim() == 4 else out / norm.view(-1)
                else:
                    import numpy as _np
                    v = _np.asarray(out)
                    nrm = _np.linalg.norm(v)
                    if nrm > 0:
                        out = v / nrm
            return out
        else:
            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(image)
            h.remove()

            if tensor:
                out = my_embedding
            else:
                if self.model_name == 'alexnet':
                    out = my_embedding.detach().numpy()[0, :]
                else:
                    out = my_embedding.detach().numpy()[0, :, 0, 0]

            if normalize_vec:
                if tensor:
                    flat = out.view(out.size(0), -1)
                    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    out = out / norm.view(-1, 1, 1, 1) if out.dim() == 4 else out / norm.view(-1)
                else:
                    import numpy as _np
                    v = _np.asarray(out)
                    nrm = _np.linalg.norm(v)
                    if nrm > 0:
                        out = v / nrm

            return out

    def _get_model_and_layer(self, model_name, layer, fine_tuned_path=None):
        
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            if fine_tuned_path:
                model.fc = nn.Identity()
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            if fine_tuned_path:
                model.fc = nn.Identity()
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet50d':
            model = timm.create_model("resnet50d.ra2_in1k", pretrained=True)
            if hasattr(model, 'fc'):
                if fine_tuned_path:
                    model.fc = nn.Identity()
            elif hasattr(model, 'head'):
                if fine_tuned_path:
                    model.head = nn.Identity()
            if layer == 'default':
                layer = model._modules.get('global_pool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer
        
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer
        
        elif model_name == "wide_resnet":
            model = models.wide_resnet50_2(pretrained = True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)
            
            return model, layer
        
        elif model_name == "resnext":
            model = models.resnext50_32x4d(pretrained = True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)
            
            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)