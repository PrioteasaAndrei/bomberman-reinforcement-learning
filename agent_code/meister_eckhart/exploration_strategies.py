import math

class EpsilonGreedyStrategy:
    def __init__(self, start_epsilon: float, min_epsilon: float):
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon

    def update_epsilon(self, step: int):
        """
        This method should be overridden by subclasses to define 
        specific epsilon update strategies.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class LinearDecayStrategy(EpsilonGreedyStrategy):
    def __init__(self, start_epsilon: float, min_epsilon: float, decay_steps: int):
        super().__init__(start_epsilon, min_epsilon)
        self.decay_steps = decay_steps

    def update_epsilon(self, step: int):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon - (self.epsilon - self.min_epsilon) / self.decay_steps
        )

class ExponentialDecayStrategy(EpsilonGreedyStrategy):
    def __init__(self, start_epsilon: float, min_epsilon: float, decay_rate: float):
        super().__init__(start_epsilon, min_epsilon)
        self.decay_rate = decay_rate

    def update_epsilon(self, step: int):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

class StepDecayStrategy(EpsilonGreedyStrategy):
    def __init__(self, start_epsilon: float, min_epsilon: float, drop_rate: float, steps_per_drop: int):
        super().__init__(start_epsilon, min_epsilon)
        self.drop_rate = drop_rate
        self.steps_per_drop = steps_per_drop

    def update_epsilon(self, step: int):
        if step % self.steps_per_drop == 0:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.drop_rate)

class InverseDecayStrategy(EpsilonGreedyStrategy):
    def __init__(self, start_epsilon: float, min_epsilon: float, alpha: float):
        super().__init__(start_epsilon, min_epsilon)
        self.alpha = alpha

    def update_epsilon(self, step: int):
        self.epsilon = max(self.min_epsilon, self.start_epsilon / (1 + self.alpha * step))

class SigmoidDecayStrategy(EpsilonGreedyStrategy):
    def __init__(self, start_epsilon: float, min_epsilon: float, max_epsilon: float, alpha: float, beta: float):
        super().__init__(start_epsilon, min_epsilon)
        self.max_epsilon = max_epsilon
        self.alpha = alpha
        self.beta = beta

    def update_epsilon(self, step: int):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * (1 / (1 + pow(2.71828, -self.alpha * (step - self.beta))))