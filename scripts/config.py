class Config:
    DATA_DIR = "data"
    CHECKPOINTS_DIR = "checkpoints"

    BATCH_SIZE = 4
    EPOCHS = 15
    IMG_LR = 1e-4
    TEXT_LR = 2e-5
    REGRESSOR_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENTCE = 5

    HIDDEN_DIM = 512
    DROPOUT = 0.1
    IMG_MODEL = "tf_efficientnetv2_m.in21k"
    IMG_SIZE = 384
    TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TEXT_LEN = 128

    VAL_FRACTION = 0.1
    SEED = 42
