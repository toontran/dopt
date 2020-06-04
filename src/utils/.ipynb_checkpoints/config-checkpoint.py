r"""
Configurations for optimization process
"""
# TODO: Gather all seeds here?

CONFIG = {}

CONFIG["distribute"] = {
    "computer_list": [
        'tst008@acet116-lnx-10.bucknell.edu',
        'tst008@acet116-lnx-11.bucknell.edu',
        'tst008@acet116-lnx-12.bucknell.edu',
#         'localhost'
    ],
    "max_jobs": 1, # Num jobs per computer
    "min_gpu": 500, # TODO: Only use computers with GPU used lower than min_gpu (mbs)
}

CONFIG["optimizer"] = {
    "num_restarts": 10,  # ???
    "raw_samples": 500,  # Sample on GP using Sobel sequence
    "options": {
        "batch_limit": 5,
        "max_iter": 200,
        "seed": 0
    }
}
