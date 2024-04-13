def get_config():
    config = {
        'lr': [1e-2],
        'epoch':1,
        'optimizer':["Adam"],
        'dropout':[0.05],
        'weight_decay':[1e-5],
        'batch_size':32,
        'data_path':'/home/ec2-user/deepsea_filtered.npz'
    }
    return config