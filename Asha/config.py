def configs():
    config = {
        'lr': [1e-2,1e-3,1e-4],
        # 'epoch':20,
        'optimizer':["Adam","AdamW","SGD"],
        'dropout':[0,0.2],
        'weight_decay':[1e-5],
        'batch_size':64,
        'data_path':'/home/wenduoc/automation/automation/data/deepsea_filtered.npz',
        'dash_res_path':"/home/wenduoc/automation/automation/Asha/final_res.npz"
    }
    return config