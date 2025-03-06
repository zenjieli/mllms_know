from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt

device = f'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True, attn_implementation="eager").eval().to(device, torch.bfloat16)

model.eval()

# Printed model layers
# QWenLMHeadModel(
#   (transformer): QWenModel(
#     (wte): Embedding(151936, 4096)
#     (drop): Dropout(p=0.0, inplace=False)
#     (rotary_emb): RotaryEmbedding()
#     (h): ModuleList(
#       (0-31): 32 x QWenBlock(
#         (ln_1): RMSNorm()
#         (attn): QWenAttention(
#           (c_attn): Linear(in_features=4096, out_features=12288, bias=True)
#           (c_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (attn_dropout): Dropout(p=0.0, inplace=False)
#         )
#         (ln_2): RMSNorm()
#         (mlp): QWenMLP(
#           (w1): Linear(in_features=4096, out_features=11008, bias=False)
#           (w2): Linear(in_features=4096, out_features=11008, bias=False)
#           (c_proj): Linear(in_features=11008, out_features=4096, bias=False)
#         )
#       )
#     )
#     (ln_f): RMSNorm()
#     (visual): VisionTransformer(
#       (conv1): Conv2d(3, 1664, kernel_size=(14, 14), stride=(14, 14), bias=False)
#       (ln_pre): LayerNorm((1664,), eps=1e-06, elementwise_affine=True)
#       (transformer): TransformerBlock(
#         (resblocks): ModuleList(
#           (0-47): 48 x VisualAttentionBlock(
#             (ln_1): LayerNorm((1664,), eps=1e-06, elementwise_affine=True)
#             (ln_2): LayerNorm((1664,), eps=1e-06, elementwise_affine=True)
#             (attn): VisualAttention(
#               (in_proj): Linear(in_features=1664, out_features=4992, bias=True)
#               (out_proj): Linear(in_features=1664, out_features=1664, bias=True)
#             )
#             (mlp): Sequential(
#               (c_fc): Linear(in_features=1664, out_features=8192, bias=True)
#               (gelu): GELU(approximate='none')
#               (c_proj): Linear(in_features=8192, out_features=1664, bias=True)
#             )
#           )
#         )
#       )
#       (attn_pool): Resampler(
#         (kv_proj): Linear(in_features=1664, out_features=4096, bias=False)
#         (attn): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=4096, out_features=4096, bias=True)
#         )
#         (ln_q): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
#         (ln_kv): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
#       )
#       (ln_post): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
#     )
#   )
#   (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
# )

image_path = './images/demo1.png'
question = 'what is the date of the photo?'

query = tokenizer.from_list_format([
    {'image': image_path},
    {'text': f'{question} Answer:'},
])

general_query = tokenizer.from_list_format([
    {'image': image_path},
    {'text': "Write a general description of the image. Answer:"},
])


with torch.no_grad():
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)

    pos = inputs['input_ids'].tolist()[0].index(151857) + 1 # 151857 is image_start_id

    general_inputs = tokenizer(general_query, return_tensors='pt')
    general_inputs = general_inputs.to(model.device)

    extract_attention_weights = None

    def attention_hook(module, input, output):
        global extract_attention_weights
        # output[1] = QK^T/\sqrt(d)
        # output[0] = softmax(output[1]) \cdot V
        # Ingoring the batch dimension, output[0] has a shape (num_img_tokens_in_LLM, hidden_dim) (256, 4096)
        # output[1] has a shape (num_img_tokens_in_LLM, num_img_tokens_in_vision) (256, 1024)
        extract_attention_weights = output[1]

    handle = model.transformer.visual.attn_pool.attn.register_forward_hook(attention_hook)

    output = model(**inputs, output_attentions=True)

    att_resample = extract_attention_weights[0]

    general_output = model(**general_inputs, output_attentions=True)

    general_att_resample = extract_attention_weights[0]

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))

    final_atts = []

    for i, ax in enumerate(axes.flatten()):

        att = output.attentions[i][0,:,-1,pos:pos+256].mean(dim=0)
        att = att @ att_resample
        att = att.to(torch.float32).detach().cpu().numpy()

        general_att = general_output.attentions[i][0,:,-1,pos:pos+256].mean(dim=0)
        general_att = general_att @ general_att_resample
        general_att = general_att.to(torch.float32).detach().cpu().numpy()

        att = att / general_att

        final_atts.append(att)
        ax.imshow(att.reshape(32,32), cmap='viridis', interpolation='nearest')
        ax.set_title(f'Layer {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('./output/qwen_attn_demo.png')