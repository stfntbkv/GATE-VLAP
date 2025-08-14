from autogpt_p.planning.goal_validator import ValidationError


class ValidationErrorHandler:

    def __init__(self, validation_error: ValidationError, chat_gpt_interface):
        self.validation_error = validation_error
        self.chat_gpt_interface = chat_gpt_interface

    def correct_error(self):
        return self.chat_gpt_interface.correct_error(self.validation_error)
