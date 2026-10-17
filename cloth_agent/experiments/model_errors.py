"""Typed model-only failures; never used to retry a physical command."""
class ModelInvocationError(RuntimeError):
    def __init__(self,message,*,retryable=False,details=None):
        super().__init__(message)
        self.retryable=retryable
        self.details=details or {}
