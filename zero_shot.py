import torch
import surgvlp
from PIL import Image
from mmengine.config import Config
from transformers import AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

configs = Config.fromfile('./tests/config_surgvlp.py')['config']
# Change the config file to load different models: config_surgvlp.py / config_hecvl.py / config_peskavlp.py

model, preprocess = surgvlp.load(configs.model_config, device=device, pretrain='./checkpoints/SurgVLP.pth')

vision_backbone = model.backbone_img.model
print(model.backbone_img.model)
print(vision_backbone.layer4[2].bn2)

tokenizer_clinical = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# 101, 102
cls_token_id = torch.tensor(tokenizer_clinical.cls_token_id, device="cuda")
sep_token_id = torch.tensor(tokenizer_clinical.sep_token_id, device="cuda")

embedding_layer = model.backbone_text.model.get_input_embeddings()
cls_emb = embedding_layer(cls_token_id).unsqueeze(0).unsqueeze(0).type(torch.float32) # Shape: (1, 1, dim)
sep_emb = embedding_layer(sep_token_id).unsqueeze(0).unsqueeze(0).type(torch.float32) # Shape: (1, 1, dim)


print(cls_emb.mean())
print(sep_emb.mean())

assert(0)


# def get_activation(self, layer_name):
#     def hook(module, input, output):
#         self.output_feature[layer_name] = output
#     return hook

# vision_backbone.layer4[2].bn2.register_forward_hook(get_activation("high_level_feature"))

# if basename == 'resnet50':
#     self.basemodel      = basemodels.resnet50(pretrained=True)
#     self.basemodel.layer1[2].bn2.register_forward_hook(self.get_activation('low_level_feature'))
#     self.basemodel.layer4[2].bn2.register_forward_hook(self.get_activation('high_level_feature'))

# template = {
#     "Grasper" : "I use grasper tor cautery forcep to grasp it",
#     "Bipolar" : "I use bipolar to coagulate and clean the bleeding",
#     "Hook" : "I use hook to dissect it",
#     "Scissors" : "I use scissors",
#     "Clipper" : "I use clipper to clip it",
#     "Irrigator" : "I use irrigator to suck it",
#     "SpecimenBag" : "I use specimenbag to wrap it",
# }

# image = preprocess(Image.open("./tests/video12_000176.png")).unsqueeze(0).to(device)
text = surgvlp.tokenize(["I use grasper to cautery forcep to grasp it", 'I use bipolar to coagulate and clean the bleeding'], device=device)
print(text['input_ids'].shape)
print(text["attention_mask"].shape)

assert(0)

with torch.no_grad():
    output_dict = model(image, text , mode='all')

    image_embeddings, _ = output_dict['img_emb']
    text_embeddings= output_dict['text_emb']

    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    print(image_embeddings.shape)
    print(text_embeddings.shape)
    
    logits_per_image = 100 * image_embeddings @ text_embeddings.T
    print(logits_per_image.shape)
    print(logits_per_image)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    probs = logits_per_image.sigmoid().cpu().numpy()

print("Label probs:", probs)