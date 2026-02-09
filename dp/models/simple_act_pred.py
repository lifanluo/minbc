from dp.models.vision.resnet import ResnetEncoder 
import torch
import torch.nn as nn

class BCNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoders = nn.ModuleDict()
        total_feat_dim = 0
        
        # 1. Proprio (10D)
        self.encoders['proprio'] = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 128)
        )
        total_feat_dim += 128
        
        # 2. Nail Flow (12D)
        self.encoders['nail'] = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 128)
        )
        total_feat_dim += 128
        
        # 3. Pad Flow (Dense)
        # CHANGED: Use positional arguments (output_dim, input_channels)
        self.encoders['pad_idx'] = ResnetEncoder(128, 2)
        total_feat_dim += 128
        
        self.encoders['pad_thb'] = ResnetEncoder(128, 2)
        total_feat_dim += 128
        
        # 4. RGB Image (Optional)
        if 'rgb' in config.data.data_key:
            # CHANGED: Use positional arguments (output_dim, input_channels)
            self.encoders['rgb'] = ResnetEncoder(256, 3)
            total_feat_dim += 256
        
        # --- Decoder ---
        self.action_head = nn.Sequential(
            nn.Linear(total_feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10) # 10D Action
        )

    def forward(self, batch):
        feats = []
        # Handle batch size inference (proprio might have Time dim or just Batch dim depending on loader)
        # Assuming (B, T, D) for consistency with sequence loaders
        if len(batch['proprio'].shape) == 3:
            b, t = batch['proprio'].shape[:2]
            flatten = True
        else:
            flatten = False
        
        # Helper to flatten or keep 
        def flat(x): 
            return x.flatten(0, 1) if flatten else x

        # A. Proprio & Nail
        feats.append(self.encoders['proprio'](flat(batch['proprio'])))
        feats.append(self.encoders['nail'](flat(batch['nail'])))
        
        # B. Pad Flow (CNN)
        feats.append(self.encoders['pad_idx'](flat(batch['pad_idx'])))
        feats.append(self.encoders['pad_thb'](flat(batch['pad_thb'])))
        
        # C. RGB Image
        if 'rgb' in batch:
            feats.append(self.encoders['rgb'](flat(batch['rgb'])))
        
        # D. Concat & Decode
        global_feat = torch.cat(feats, dim=-1)
        pred_action = self.action_head(global_feat)
        
        if flatten:
            return pred_action.view(b, t, -1)
        return pred_action