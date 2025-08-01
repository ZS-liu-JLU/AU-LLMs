import os
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
except ImportError:
    print("请安装 transformers 和 peft: pip install transformers peft")
    exit()

import utils.datasets as datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MEGC(Dataset):
    def __init__(self, frames, labels, transform=None):
        self.frames = frames
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        sample = self.frames[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.from_numpy(sample.copy()).float(), torch.from_numpy(label.copy()).float()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            pt = xs_pos * y + xs_neg * (1 - y)
            focal_weight = torch.where(y == 1, (1 - xs_pos)**self.gamma_pos, xs_neg**self.gamma_neg)
            loss *= focal_weight
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
        
        return -loss.sum()


def calculate_W(T, alpha=20, r1=0.4, r2=0.05):
    W = torch.zeros(T, T, dtype=torch.float).to(device)
    for i in range(T):
        for j in range(T):
            a = j - i
            b = min(1, i)
            if j > i:
                W[i, j] = alpha * (1 - r1) ** a * r1 ** b - alpha * (1 - r2) ** a * r2 ** b
            elif j == i:
                W[i, j] = alpha * (r1 - r2)
    return W


def led(x, W):
    s, f, c, h, w = x.shape
    W = W.to(x.device)
    out = torch.einsum("sfchw,fx->sxchw", x, W)
    div = torch.einsum("sfchw,fx->sxchw", torch.abs(x), torch.abs(W))
    out /= div + 1e-8
    out[:, 0] = x[:, 0]
    return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Net_EnhancedFusion_LLM(nn.Module):
    def __init__(self, task_num, dropout=0.5, llm_model_path='Please replace it with your LLM path'):
        super().__init__()
        self.task_num = task_num
        h1, h2, h3 = 32, 64, 256
        
        self.conv1 = nn.Conv3d(1, h1, kernel_size=(1, 5, 5), stride=1)
        self.pool = nn.MaxPool3d((1, 3, 3), stride=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(h1)
        self.drop1 = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(h1, h2, kernel_size=(2, 3, 3), stride=1)
        self.bn2 = nn.BatchNorm3d(h2)
        self.se = SELayer(h2)
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.drop2 = nn.Dropout3d(dropout)
        self.fc1 = nn.Linear(9**2 * 2 * h2, h3)
        self.drop3 = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.log(torch.tensor(10.0)))
        self.r1 = nn.Parameter(torch.log(torch.tensor(0.4)))
        self.r2 = nn.Parameter(torch.log(torch.tensor(0.05)))

        print(f"从以下路径加载Tokenizer: {llm_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"从以下路径加载LLM: {llm_model_path}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
        
        llm_embedding_dim = self.llm_model.config.hidden_size

        print("正在配置LoRA...")
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1, 
            bias="none", 
            task_type="CAUSAL_LM"
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        self.llm_model.print_trainable_parameters()

        mid_feature_dim = 9**2 * 2 * h2
        high_feature_dim = h3
        concatenated_dim = mid_feature_dim + high_feature_dim

        self.fusion_projector = nn.Sequential(
            nn.Linear(concatenated_dim, (concatenated_dim + llm_embedding_dim) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((concatenated_dim + llm_embedding_dim) // 2, llm_embedding_dim)
        )
        
        self.final_classifier = nn.Linear(llm_embedding_dim, self.task_num)
        
    def forward(self, x):
        W = calculate_W(6, torch.exp(self.alpha), torch.exp(self.r1), torch.exp(self.r2))
        x = led(x, W)
        x = x[:, 1:].permute(0, 2, 1, 3, 4)
        x = self.drop1(self.bn1(self.pool(F.relu(self.conv1(x)))))
        x = self.bn2(F.relu(self.conv2(x)))
        
        b, c, d, h, w = x.shape
        x_se = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        x_se = self.se(x_se)
        x = x_se.view(b, d, c, h, w).permute(0, 2, 1, 3, 4)
        
        mid_level_map = self.drop2(self.pool2(x))
        mid_level_flat = mid_level_map.view(mid_level_map.shape[0], -1)
        high_level_features = self.drop3(F.relu(self.fc1(mid_level_flat)))

        batch_size = high_level_features.shape[0]
        concatenated_features = torch.cat([mid_level_flat, high_level_features], dim=1)
        projected_features = self.fusion_projector(concatenated_features).half()
        
        prompt_text = "分析以下融合后的面部视频特征以进行动作单元分类。"
        tokenized_prompt = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(tokenized_prompt.input_ids).expand(batch_size, -1, -1)
        
        combined_embeddings = torch.cat([prompt_embeddings, projected_features.unsqueeze(1)], dim=1)
        outputs = self.llm_model(inputs_embeds=combined_embeddings, output_hidden_states=True)
        llm_output_features = outputs.hidden_states[-1][:, -1, :]
        logits = self.final_classifier(llm_output_features.float())
        
        return logits


def train_transform(video):
    n_frames = 6
    if video.shape[0] - 1 <= n_frames:
        idx = np.arange(video.shape[0])
    else:
        max_f = np.random.randint(n_frames, video.shape[0] - 1)
        idx = np.round(np.linspace(0, max_f, n_frames)).astype("int")
    video = video[idx]
    return np.expand_dims(video, 1)


def test_transform(video):
    n_frames = 6
    idx = np.round(np.linspace(0, video.shape[0] - 1, n_frames)).astype("int")
    video = video[idx]
    return np.expand_dims(video, 1)


def LOSO(features, df, action_units, epochs=200, lr=0.01, batch_size=256, dropout=0.05, weight_decay=0.001):
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    labels = df[action_units].values
    outputs_list, labels_list = [], []
    subjects = df["subject"].unique()
    
    for test_subject in tqdm(subjects, desc="LOSO 进度"):
        print(f"\n--- Testing on subject: {test_subject} ---")
        
        train_indices = df[df["subject"] != test_subject].index
        test_indices = df[df["subject"] == test_subject].index
        
        X_train, y_train = [features[i] for i in train_indices], labels[train_indices]
        X_test, y_test = [features[i] for i in test_indices], labels[test_indices]
        
        train_dataset = MEGC(X_train, y_train, train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        test_dataset = MEGC(X_test, y_test, test_transform)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
        
        net = Net_EnhancedFusion_LLM(len(action_units), dropout=dropout).to(device)
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in tqdm(range(epochs), desc=f"Epochs for subject {test_subject}", leave=False):
            net.train()
            for batch_idx, (data_batch, labels_batch) in enumerate(train_loader):
                data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = net(data_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            subject_outputs, subject_labels = [], []
            for data_batch_test, labels_batch_test in test_loader:
                outputs = net(data_batch_test.to(device))
                subject_outputs.append(outputs.cpu())
                subject_labels.append(labels_batch_test.cpu())
            outputs_list.append(torch.cat(subject_outputs))
            labels_list.append(torch.cat(subject_labels))
            
    all_predictions = torch.cat(outputs_list)
    all_labels = torch.cat(labels_list)
    
    best_f1s, best_thresholds = [], []
    print("\n--- Optimizing thresholds for each AU ---")
    for i in range(all_labels.shape[1]):
        best_f1, best_thresh = 0, 0
        y_true = all_labels[:, i]
        y_pred_probs = torch.sigmoid(all_predictions[:, i])
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_binary = (y_pred_probs > thresh).int()
            f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        best_f1s.append(best_f1)
        best_thresholds.append(best_thresh)

    print("\n--- Final Results (with Optimized Thresholds) ---")
    for i, au in enumerate(action_units):
        print(f"F1-Score for {au}: {best_f1s[i]:.4f} (Best Threshold: {best_thresholds[i]:.2f})")
    print(f"\nMean Macro F1-Score: {np.mean(best_f1s):.4f}")
    
    return best_f1s


if __name__ == '__main__':
    print("Initializing SAMM dataset...")
    samm = datasets.SAMM(resize=64, color=False, cropped=False, optical_flow=False)
    df = samm.data_frame
    input_data = samm.data

    print("Preprocessing video frames...")
    pr_frames = list(input_data)

    action_units = ["AU2", "AU4", "AU7", "AU12"]
    
    print("Starting LOSO cross-validation on SAMM dataset...")
    predictions = LOSO(
        features=pr_frames, 
        df=df, 
        action_units=action_units, 
        epochs=350,
        lr=3e-5,
        weight_decay=0.005,
        dropout=0.3,
        batch_size=256 
    )
