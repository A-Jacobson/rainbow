
class AtariDefaults:
    # replay config
    MEMORY_CAPACITY = 500000
    INITIAL_MEMORY = 50000

    # Q network training config
    LEARNING_RATE = 1.5e-4
    BATCH_SIZE = 32

    # exploration vs exploitation
    EPSILON_MAX = 0.9
    EPSILON_MIN = 0.05
    DECAY_RATE = 1e-5

    # saving
    CHECKPOINT_INTERVAL = 50000
