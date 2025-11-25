import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'bert_config/xlm-roberta-base'
    else:
        cfg_name = 'bert_config/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(config=None):
    
    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp.vilmodel_cmt import GlocalTextPathNavCMT

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    # new_ckpt_weights中都是以bert.开头的key，而GlocalTextPathNavCMT中对应项没有bert.。load的过程能自动保证上述对应关系，使得权重正确加载。
    # 加载时命令行打印说：权重文件中有一些权重没被用到，都是GraphLXRTXLayer中if config.use_lang2visn_attn里的项，因为原版模型在线训练不再需要lang的自注意力了，因此在线模型中就不存在这三个网络层
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        # print("keys\n", ckpt_weights.keys())
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            if 'sap_head' in k:
                new_ckpt_weights['bert.' + k] = v
            else:
                 new_ckpt_weights[k] = v
    
    if config.task_type == 'r2r':
        cfg_name = 'bert_config/bert-base-uncased'
    elif config.task_type == 'rxr':
        cfg_name = 'bert_config/xlm-roberta-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if config.task_type == 'rxr':
        vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = 512
    vis_config.use_depth_embedding = config.use_depth_embedding
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = 9
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4
    vis_config.graph_sprels = config.use_sprels
    vis_config.glocal_fuse = 'global'

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_pano_embedding = config.fix_pano_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False
    # print("new_ckpt_weights\n", new_ckpt_weights.keys())
    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
    print("init end")
    # print("visual_model\n", visual_model)
    return visual_model
