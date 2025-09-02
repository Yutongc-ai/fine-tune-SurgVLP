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

class COOP(nn.Module):
    '''
    CoOp method
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
        self.checkpoint_path =  "checkpoints/best_coop_" + str(self.configs.num_shots) + "shots.pt"
        self.early_stop = configs.early_stop
        self.early_stopping = EarlyStopping(patience = self.patience, path = self.checkpoint_path)
        self.annealling = configs.annealling

        self.feature_width = configs.model_config.backbone_img.num_classes # 768
        self.num_classes = configs.dataset_config.num_classes

        self.scale_logits = 10

        # ======================================Training Parameters=====================================
        # metrcis
        self.criterion = torch.nn.BCEWithLogitsLoss()

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
            prompt_embeddings, attention_mask = self.prompt_learner()
                
            _, text_features, _ = self.model.backbone_text(
                input_embedding=prompt_embeddings, 
                attn_mask=attention_mask
            )
            text_features /= text_features.norm(dim = -1, keepdim = True)

            for _ in range(len(feature_loader)):
                global_image_features, _, label, _ = feature_loader[_]
                global_image_features = global_image_features.to(device)
                global_image_features = global_image_features / global_image_features.norm(dim = -1, keepdim = True)

                logits = self.scale_logits * global_image_features @ text_features.t()
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
            project=f"coop-{self.configs.model_config.type}",
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

            for i, (images, target, _, _) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                # 1. Get image features
                with torch.no_grad():
                    image_features, _ = self.model.extract_feat_img(images)

                # 2. Get text features
                # a. Generate prompt embeddings and attention mask from the learner
                prompt_embeddings, attention_mask = self.prompt_learner()
                
                # b. Pass these directly to the BertEncoder
                # Note: We use the `input_embedding` argument of your BertEncoder's forward method
                # This bypasses the need for token IDs and the internal tokenizer
                _, text_features, _ = self.model.backbone_text(
                    input_embedding=prompt_embeddings, 
                    attn_mask=attention_mask
                )
                
                # 3. Normalize features (common practice in contrastive learning)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 4. Calculate similarity and loss
                logits = self.scale_logits * image_features @ text_features.t()

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

# NCTX: 16
# CTX_INIT: ""
# CLASS_TOKEN_POSITION: "end"
# CSC: False
# WARMUP_EPOCH: 1
# WARMUP_CONS_LR: 0.00001
# PREC: "fp16"

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
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            # normal distribution initialization
            nn.init.normal_(ctx_vectors, std=0.02)
            
            # Placeholder "X" is just for visualization
            prompt_prefix = " ".join(["[X]"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # This is the learnable part

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
        ctx = self.ctx.cuda()
        if ctx.dim() == 2:
            # Expand UNIFIED context for all classes
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Get the fixed special token embeddings, expanded for the batch
        cls_token = self.cls_token.expand(self.n_cls, -1, -1)
        sep_token = self.sep_token.expand(self.n_cls, -1, -1)
        
        prompts = []
        for i in range(self.n_cls):
            # Get embeddings for the current class name
            classname_emb = self.classname_embeddings[i].unsqueeze(0).to(ctx.device)
            ctx_i = ctx[i:i+1, :, :] # Shape: (1, n_ctx, dim)

            # Standard BERT structure: [CLS] ... [SEP]
            if self.class_token_position == "end":
                # [CLS] [CTX] [CLASS] [SEP]
                p = torch.cat([cls_token[i:i+1], ctx_i, classname_emb, sep_token[i:i+1]], dim=1)
            
            elif self.class_token_position == "middle":
                # [CLS] [CTX_part1] [CLASS] [CTX_part2] [SEP]
                half_n_ctx = self.n_ctx // 2
                ctx_half1 = ctx_i[:, :half_n_ctx, :]
                ctx_half2 = ctx_i[:, half_n_ctx:, :]
                p = torch.cat([cls_token[i:i+1], ctx_half1, classname_emb, ctx_half2, sep_token[i:i+1]], dim=1)
            
            elif self.class_token_position == "front":
                # [CLS] [CLASS] [CTX] [SEP]
                p = torch.cat([cls_token[i:i+1], classname_emb, ctx_i, sep_token[i:i+1]], dim=1)
            
            else:
                raise ValueError("Invalid class_token_position")
            
            prompts.append(p)
        
        # At this point, `prompts` is a list of tensors with different lengths
        # We need to pad them to the same length to create a single batch tensor.
        
        # Find the max length in the batch
        # max_len = max(p.shape[1] for p in prompts)
        # pad to CLIP accepted text length, which is 77
        max_len = 77
        
        # Pad each prompt and create attention mask
        padded_prompts = []
        attention_masks = []
        for p in prompts:
            padding_len = max_len - p.shape[1]
            # Create padding tensor
            padding = torch.zeros(1, padding_len, p.shape[2], dtype=self.dtype, device=p.device)
            padded_p = torch.cat([p, padding], dim=1)
            padded_prompts.append(padded_p)
            
            # Create attention mask: 1 for real tokens, 0 for padding
            attn_mask = torch.cat([
                torch.ones(1, p.shape[1], device=p.device),
                torch.zeros(1, padding_len, device=p.device)
            ], dim=1)
            attention_masks.append(attn_mask)
            
        # Stack them into a single tensor
        final_prompts = torch.cat(padded_prompts, dim=0)
        final_attn_mask = torch.cat(attention_masks, dim=0)
        
        return final_prompts, final_attn_mask