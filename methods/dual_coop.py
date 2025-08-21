import time
from .utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MultiLabelDatasetBase, build_data_loader, preload_local_features, Cholec80Features, Cholec80FeaturesVal
from methods.utils import multilabel_metrics, search_hp_tip, ConstantWarmupScheduler
from methods.early_stopping import EarlyStopping
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer
from collections import OrderedDict
from methods.loss import AsymmetricLoss

class DualCOOP(nn.Module):
    '''
    Dual CoOp method
    '''
    
    def __init__(self, configs, model, preprocess, tokenizer):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.configs = configs

        # =====================================Training Settings======================================
        self.lr = configs.learning_rate
        self.epochs = configs.epochs
        self.batch_size = configs.batch_size
        self.accumulate_step = configs.accumulate_step
        self.patience = configs.patience
        self.checkpoint_path =  "checkpoints/best_dual_coop_" + str(self.configs.num_shots) + "shots.pt"
        self.early_stop = configs.early_stop
        self.early_stopping = EarlyStopping(patience = self.patience, path = self.checkpoint_path)
        self.annealling = configs.annealling

        self.feature_width = configs.model_config.backbone_img.num_classes # 768
        self.num_classes = configs.dataset_config.num_classes

        self.scale_logits = 100

        # ======================================Training Parameters=====================================
        # metrcis
        self.criterion = AsymmetricLoss(clip=0.1)

        # freeze image and text encoder
        for param in self.model.parameters():
            # print(param.requires_grad)
            param.requires_grad = False

        # ====================COOP Required Model========================
        class_names = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
        self.prompt_learner = BertPromptLearner(configs, model.dtype, class_names, self.model.backbone_text.model)

    def get_metrics(self, split):
        # total_loss = 0.0
        all_probs = []
        all_logits = []
        all_labels = []

        if split == "val":
            feature_loader = self.val_feature
        elif split == "test":
            feature_loader = self.test_feature
        else:
            assert(0, "get metrics split not valid")

        with torch.no_grad():
            for _ in range(len(feature_loader)):
                _, region_features, label = feature_loader[_]
                region_features = region_features[:64]
                label = label[:64]
                bs, height, width, dim = region_features.shape
                region_features = region_features.view(bs, height*width, dim).cuda()
                region_features = self.model.backbone_img.global_embedder(region_features)
                region_features = region_features / region_features.norm(dim = -1, keepdim = True)

                # 2. 获取正负文本特征
                pos_prompt_embeddings, neg_prompt_embeddings, attention_mask = self.prompt_learner()
                # (F_m_t)+ and (F_m_t)- in paper
                # shape: [num_classes, feature_dim]
                _, pos_text_features, _ = self.model.backbone_text(
                    input_embedding=pos_prompt_embeddings, 
                    attn_mask=attention_mask
                )

                _, neg_text_features, _ = self.model.backbone_text(
                    input_embedding=neg_prompt_embeddings, 
                    attn_mask=attention_mask
                )
                
                pos_text_features = pos_text_features / pos_text_features.norm(dim=-1, keepdim=True)
                neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)
                
                S_pos = region_features @ pos_text_features.t() # [B, N, Cls]
                S_pos = S_pos.permute(0, 2, 1) # [B, Cls, N]
                
                S_neg = region_features @ neg_text_features.t() # [B, N, Cls]
                S_neg = S_neg.permute(0, 2, 1) # [B, Cls, N]
                
                weights = F.softmax(S_pos, dim=-1) # softmax over regions
                S_pos_agg = (weights * S_pos).sum(dim=-1)
                S_neg_agg = (weights * S_neg).sum(dim=-1) # 使用相同的权重

                logits = self.scale_logits * (S_pos_agg - S_neg_agg)

                prob = logits.sigmoid()

                all_probs.append(prob)
                all_labels.append(label)

        final_labels = torch.cat(all_labels, dim=0)
        final_probs = torch.cat(all_probs, dim=0).to('cpu')
        metrics = multilabel_metrics(final_labels, final_probs)

        return metrics

    def forward(self,
                dataset: MultiLabelDatasetBase):

        wandb.init(
            project=f"cocoop-{self.configs.model_config.type}",
            name=f"batchsize{self.batch_size * self.accumulate_step}_lr{self.lr}_shot{self.configs.num_shots}_epoch{self.epochs}",
            config=self.configs,
            mode="offline",
        )

        # ===========================build dataloader=============================
        # Creating dataset loader
        test_loader = build_data_loader(data_source=dataset.test, batch_size = self.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)
        
        val_loader = build_data_loader(data_source=dataset.val, batch_size = self.batch_size, is_train = False, tfm = self.preprocess,
                                    num_classes = dataset.num_classes)

        if not self.configs.preload_local_features:
            preload_local_features(self.configs, "test", self.model, test_loader)
            preload_local_features(self.configs, "val", self.model, val_loader)
        
        self.test_feature = Cholec80Features(self.configs, "test")
        self.val_feature = Cholec80FeaturesVal(self.configs, "val")

        print(len(self.test_feature))
        print(len(self.val_feature))

        # Generate few shot data
        train_data = dataset.generate_fewshot_dataset_(self.configs.num_shots, split="train")

        train_loader = build_data_loader(data_source=train_data, batch_size = self.batch_size, tfm=self.preprocess, is_train=True, 
                                         num_classes = dataset.num_classes)
        

        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(coop_model.prompt_learner, self.cfg.OPTIM)
        self.optimizer = torch.optim.SGD(self.prompt_learner.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        self.scheduler = ConstantWarmupScheduler(self.optimizer, self.scheduler, 5, 1e-4)

        # Training Prodecure
        print("**** Start Training **** \n")
        best_val_map, best_epoch = 0.0, 0
        for epoch in range(self.epochs):
            # Train
            self.prompt_learner.train()

            epoch_loss = 0 
            batch_count = 0
            print('Train Epoch: {:} / {:}'.format(epoch, self.epochs))
            self.optimizer.zero_grad()

            for i, (images, target, _) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                # 1. 获取区域视觉特征
                # F_i_v in paper, shape: [batch_size, num_patches, feature_dim]
                with torch.no_grad():
                    _, region_features = self.model.extract_feat_img(images)
                    region_features = region_features.permute(0, 2, 3, 1)
                    bs, height, width, dim = region_features.shape
                    region_features = region_features.view(bs, height*width, dim)
                    region_features = self.model.backbone_img.global_embedder(region_features)
                    region_features = region_features / region_features.norm(dim = -1, keepdim = True)

                # 2. 获取正负文本特征
                pos_prompt_embeddings, neg_prompt_embeddings, attention_mask = self.prompt_learner()
                # (F_m_t)+ and (F_m_t)- in paper
                # shape: [num_classes, feature_dim]
                _, pos_text_features, _ = self.model.backbone_text(
                    input_embedding=pos_prompt_embeddings, 
                    attn_mask=attention_mask
                )

                _, neg_text_features, _ = self.model.backbone_text(
                    input_embedding=neg_prompt_embeddings, 
                    attn_mask=attention_mask
                )
                
                # 归一化
                pos_text_features = pos_text_features / pos_text_features.norm(dim=-1, keepdim=True)
                neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)
                
                # 3. 计算区域相似度
                # S+_i,m and S-_i,m in paper
                # region_features: [B, N, D], pos_text_features: [C, D]
                # S_pos: [B, C, N]
                S_pos = region_features @ pos_text_features.t() # [B, N, Cls]
                S_pos = S_pos.permute(0, 2, 1) # [B, Cls, N]
                
                S_neg = region_features @ neg_text_features.t() # [B, N, Cls]
                S_neg = S_neg.permute(0, 2, 1) # [B, Cls, N]
                
                # 4. 类别特定的区域聚合 (Eq. 5 & 6)
                # S_pos_agg (S+_m) and S_neg_agg (S-_m)
                weights = F.softmax(S_pos, dim=-1) # softmax over regions
                S_pos_agg = (weights * S_pos).sum(dim=-1)
                S_neg_agg = (weights * S_neg).sum(dim=-1) # 使用相同的权重

                # 5. 计算最终的二元分类概率 (Eq. 3)
                # 注意：论文的Eq.3是计算概率p，但ASL损失函数通常接收logits作为输入。
                # 我们计算logits，然后可以传递给 sigmoid + ASL loss。
                # logit = S_pos_agg - S_neg_agg 是一个合理的选择，因为它等价于 log(p / (1-p))
                # 其中 p = exp(S_pos_agg) / (exp(S_pos_agg) + exp(S_neg_agg))
                # 论文中没有明确给出最终logits的计算方式，但这种对比形式是标准的。
                
                # 我们返回聚合后的相似度分数，这可以被视为logits
                # 这里的logit_scale是CLIP预训练的温度系数，可以应用一下
                logits = self.scale_logits * (S_pos_agg - S_neg_agg)

                loss = self.criterion(logits, target) / self.accumulate_step
                epoch_loss += loss.item() * self.accumulate_step
                batch_count += 1

                loss.backward()

                if (i+1) % self.accumulate_step == 0 or (i + 1) == len(train_loader): # last iteration need to update grad
                    self.optimizer.step()
                    if self.annealling:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                
            avg_epoch_loss = epoch_loss / batch_count
            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch+1})
            current_lr = self.scheduler.get_last_lr()[0]
            print('Epoch {} LR: {:.6f}, Loss: {:.4f}'.format(epoch+1, current_lr, avg_epoch_loss))
            self.prompt_learner.eval()

            val_metrics = self.get_metrics("val")
            cur_val_map = val_metrics["mAP"]
            wandb.log({"val_map": cur_val_map, "epoch": epoch+1})
            wandb.log({"val_f1": val_metrics["f1"], "epoch": epoch+1})
            wandb.log({"val_precision": val_metrics["precision"], "epoch": epoch+1})
            wandb.log({"val_recall": val_metrics["recall"], "epoch": epoch+1})

            save_model = self.early_stopping(cur_val_map)
            if save_model:
                torch.save(self.state_dict(), self.checkpoint_path)
                # print(f'Validation map decreased ({self.val_map_max:.6f} --> {val_map:.6f}). Saving model...')
                print(f'Validation map increased. Saving model...')

            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            if cur_val_map > best_val_map:
                best_val_map = cur_val_map
                test_metrics = self.get_metrics("test")
                test_map = test_metrics["mAP"]
                print(f"test mAP from best clip adapter model {test_map}")
                wandb.log({"test_map": test_map, "epoch": epoch+1})
                wandb.log({"test_f1": test_metrics["f1"], "epoch": epoch+1})
                wandb.log({"test_precision": test_metrics["precision"], "epoch": epoch+1})
                wandb.log({"test_recall": test_metrics["recall"], "epoch": epoch+1})

        wandb.finish()
        return test_metrics

class BertPromptLearner(nn.Module):
    """
    Adapts the PromptLearner concept for a BERT-based text encoder.
    """
    def __init__(self, cfg, dtype, classnames, bert_encoder):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16  # Number of context tokens (default: 16)
        ctx_init = ""  # Context initialization string (default: "")
        self.class_token_position = "end"

        # 1. Get BERT-specific properties
        bert_model = bert_encoder
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        ctx_dim = 768 # e.g., 768 for bert-base
        
        print(f"Initializing BertPromptLearner with ctx_dim={ctx_dim}")

        # 2. Initialize Context Vectors (self.ctx)
        if ctx_init:
            # this branch is NOT USED
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenizer(ctx_init, return_tensors="pt", add_special_tokens=False)
            prompt_ids = prompt['input_ids'][0]
            n_ctx = len(prompt_ids) # Dynamically set n_ctx based on init string
            
            with torch.no_grad():
                # Get embeddings from the BERT model's embedding layer
                embedding_layer = bert_model.model.get_input_embeddings()
                ctx_vectors = embedding_layer(prompt_ids).type(dtype)
            
            prompt_prefix = ctx_init
        else:
            # Random initialization
            # UNIFIED PROMPT FOR ALL CLS
            
            print("Initializing a generic context")
            pos_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            neg_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            # normal distribution initialization
            nn.init.normal_(pos_ctx_vectors, std=0.02)
            nn.init.normal_(neg_ctx_vectors, std=0.02)
            
            # Placeholder "X" is just for visualization
            pos_prompt_prefix = " ".join(["[X]"] * n_ctx)
            neg_prompt_prefix = " ".join(["[X]"] * n_ctx)

        print(f'Initial context: positive prompt: "{pos_prompt_prefix}" negetive prompt: {neg_prompt_prefix}')
        print(f"Number of context words (tokens): {n_ctx}")

        self.pos_ctx = nn.Parameter(pos_ctx_vectors)  # This is the learnable part
        self.neg_ctx = nn.Parameter(neg_ctx_vectors)  # This is the learnable part

        # 3. Prepare fixed parts of the prompt (special tokens and class names)
        classnames = [name.replace("_", " ") for name in classnames]
        
        # Get special token IDs and their embeddings
        cls_token_id = torch.tensor(tokenizer.cls_token_id, device=device)
        sep_token_id = torch.tensor(tokenizer.sep_token_id, device=device)
        with torch.no_grad():
            embedding_layer = bert_model.get_input_embeddings()
            cls_emb = embedding_layer(cls_token_id).unsqueeze(0).unsqueeze(0).type(dtype) # Shape: (1, 1, dim)
            sep_emb = embedding_layer(sep_token_id).unsqueeze(0).unsqueeze(0).type(dtype) # Shape: (1, 1, dim)

        # Store these as buffers
        self.register_buffer("cls_token", cls_emb)
        self.register_buffer("sep_token", sep_emb)
        
        # Prepare class name embeddings
        self.tokenized_classnames = []
        self.classname_embeddings = []
        for name in classnames:
            tokens = tokenizer(name, return_tensors='pt', add_special_tokens=False)['input_ids'][0].cuda()
            self.tokenized_classnames.append(tokens)
            with torch.no_grad():
                class_emb = embedding_layer(tokens).type(dtype)
                aggregated_emb = torch.sum(class_emb, dim=0, keepdim=True)
            self.classname_embeddings.append(aggregated_emb)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.dtype = dtype

    def forward(self):
        """
        Constructs the full prompt embeddings by concatenating learnable
        context vectors with fixed class name and special token embeddings.
        """
        # Get the current context vectors
        pos_ctx = self.pos_ctx.cuda()
        neg_ctx = self.neg_ctx.cuda()
        if pos_ctx.dim() == 2:
            # Expand UNIFIED context for all classes
            pos_ctx = pos_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            neg_ctx = neg_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Get the fixed special token embeddings, expanded for the batch
        cls_token = self.cls_token.expand(self.n_cls, -1, -1)
        sep_token = self.sep_token.expand(self.n_cls, -1, -1)
        
        pos_prompts = []
        neg_prompts = []
        for i in range(self.n_cls):
            # Get embeddings for the current class name
            classname_emb = self.classname_embeddings[i].unsqueeze(0).to(device)
            pos_ctx_i = pos_ctx[i:i+1, :, :] # Shape: (1, n_ctx, dim)
            neg_ctx_i = neg_ctx[i:i+1, :, :] # Shape: (1, n_ctx, dim)

            # Standard BERT structure: [CLS] ... [SEP]
            if self.class_token_position == "end":
                # [CLS] [CTX] [CLASS] [SEP]
                pos_p = torch.cat([cls_token[i:i+1], pos_ctx_i, classname_emb, sep_token[i:i+1]], dim=1)
                neg_p = torch.cat([cls_token[i:i+1], neg_ctx_i, classname_emb, sep_token[i:i+1]], dim=1)
            
            else:
                raise NotImplementedError
            
            pos_prompts.append(pos_p)
            neg_prompts.append(neg_p)
        
        # At this point, `prompts` is a list of tensors with different lengths
        # We need to pad them to the same length to create a single batch tensor.
        
        # Find the max length in the batch
        # max_len = max(p.shape[1] for p in prompts)
        # pad to CLIP accepted text length, which is 77
        max_len = 77
        
        # Pad each prompt and create attention mask
        pos_padded_prompts = []
        neg_padded_prompts = []
        attention_masks = []
        for p_idx in range(len(pos_prompts)):
            pos_p = pos_prompts[p_idx]
            neg_p = neg_prompts[p_idx]
            padding_len = max_len - pos_p.shape[1]
            # Create padding tensor
            padding = torch.zeros(1, padding_len, pos_p.shape[2], dtype=self.dtype, device=pos_p.device)
            pos_padded_p = torch.cat([pos_p, padding], dim=1)
            neg_padded_p = torch.cat([neg_p, padding], dim=1)
            pos_padded_prompts.append(pos_padded_p)
            neg_padded_prompts.append(neg_padded_p)
            
            # Create attention mask: 1 for real tokens, 0 for padding
            attn_mask = torch.cat([
                torch.ones(1, pos_p.shape[1], device=pos_p.device),
                torch.zeros(1, padding_len, device=pos_p.device)
            ], dim=1)
            attention_masks.append(attn_mask)
            
        # Stack them into a single tensor
        final_pos_prompts = torch.cat(pos_padded_prompts, dim=0)
        final_neg_prompts = torch.cat(neg_padded_prompts, dim=0)
        final_attn_mask = torch.cat(attention_masks, dim=0)
        
        return final_pos_prompts, final_neg_prompts, final_attn_mask

