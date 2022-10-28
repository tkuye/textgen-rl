class BaseModel:
    def save(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def act(self, state):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

    def to_text(self, state):
        raise NotImplementedError