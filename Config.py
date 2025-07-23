import ml_collections
def get_model_config():
    config = ml_collections.ConfigDict()
    config.extracted_feature = [1024, 14, 14]
    config.core = [256, 14, 14]
    config.rank = [64, 14, 14]
    config.C_scale = 1024
    config.scale_dim = 256
    config.num_heads = 4
    config.num_layers = 6
    config.expand_ratio = 4
    return config