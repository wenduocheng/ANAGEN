def get_config():
    config = {
        'lr': [1e-2, 1e-3, 1e-4],
        'epoch':20,
        'optimizer':["Adam", "AdamW", "SGD",],
        'dropout':[0, 0.05, 0.15],
        'weight_decay':[1e-4, 1e-5],
        'batch_size':32,
        'data_path':'/home/ec2-user/automation/deepsea_train/'
    }
    return config