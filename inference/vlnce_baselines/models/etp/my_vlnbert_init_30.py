import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'bert_config/xlm-roberta-base'
    else:
        cfg_name = 'bert_config/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(config=None, dropout_rate=0.1):
    
    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp.my_vilmodel_cmt_30 import GlocalTextPathNavCMT

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    # new_ckpt_weights中都是以bert.开头的key，而GlocalTextPathNavCMT中对应项没有bert.。load的过程能自动保证上述对应关系，使得权重正确加载。
    keywords = ['graph_query_text', 'graph_attentioned_txt_embeds_transform', 'global_sap_head']
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        # print("keys\n", ckpt_weights.keys())
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            # if 'sap_head' in k or 'cls_transform' in k or 'stage_head' in k:
            if any(key in k for key in keywords):
                new_ckpt_weights['bert.' + k] = v
            else:
                new_ckpt_weights[k] = v
    
    # if config.task_type == 'r2r':
    #     cfg_name = 'bert_config/bert-base-uncased'
    # elif config.task_type == 'rxr':
    #     cfg_name = 'bert_config/xlm-roberta-base'
    cfg_name = 'bert_config/xlm-roberta-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    # if config.task_type == 'rxr':
    vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = 512
    vis_config.use_depth_embedding = config.use_depth_embedding
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = 12
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4
    vis_config.graph_sprels = config.use_sprels
    vis_config.glocal_fuse = 'global'

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_pano_embedding = config.fix_pano_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True

    vis_config.pred_head_dropout_prob = dropout_rate
    vis_config.hidden_dropout_prob = dropout_rate
    vis_config.attention_probs_dropout_prob = dropout_rate

    vis_config.use_lang2visn_attn = True

    vis_config.max_txt_task_embeddings = 4
    vis_config.max_gmap_task_embeddings = 3

    # print("new_ckpt_weights\n", new_ckpt_weights.keys())
    # print("Check model config: ", vis_config)
    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
    print("init end")
    # print("visual_model\n", visual_model)
    return visual_model
