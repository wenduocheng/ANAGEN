def get_config():
    config = {
        'lr': [1e-2,1e-3],
        'epoch':20,
        'optimizer':["Adam"],
        'dropout':[0.05,0.01],
        'weight_decay':[1e-5],
        'batch_size':32,
        'data_path':'/home/ec2-user/deepsea_filtered.npz'
    }
    return config