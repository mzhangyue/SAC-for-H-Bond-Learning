

# Base class for all environments
class EnvBase:
    # Initializer for the environment
    def __init__(self):
        self.env_name = ""
        self.started = False
        self.action_dim = 0
        self.state_dim = 0
        return None

    # Applies action to state and transtion to next state
    def apply_action(self):
        return None

    # Gets the current state
    def get_state(self):
        return None

    # Should return 
    def get_reward(self, state, action):
        return None
    
    # Gets the action space 
    def get_action_space(self):
        return None
