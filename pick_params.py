def pick_params(manual_seed=None):
    from random import randint, seed
    if manual_seed is not None:
        seed(manual_seed)
    # Uniform sampling of parameters
    REWARD_GAME_WON = randint(50,120)
    REWARD_GAME_LOST = randint(-100,-20)
    REWARD_ZERO_FIELD = randint(-15, 15)
    REWARD_NUMBER_FIELD = randint(-15, 15)
    REWARD_ALREADY_SHOWN_FIELD = randint(-200,-80)
    out={"REWARD_GAME_WON":REWARD_GAME_WON, "REWARD_GAME_LOST":REWARD_GAME_LOST, "REWARD_ZERO_FIELD":REWARD_ZERO_FIELD, "REWARD_NUMBER_FIELD":REWARD_NUMBER_FIELD, "REWARD_ALREADY_SHOWN_FIELD":REWARD_ALREADY_SHOWN_FIELD}
    return(out)
