import os
import sys
from src.logger import logger  # Importing logger from the logger module

class CustomException(Exception):
    """
    Custom exception class that captures system messages and error details.
    """
    def __init__(self, system_message: str, error: Exception):
        self.system_message = system_message
        _, _, exc_tb = sys.exc_info()
        self.file_name = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        self.line_number = exc_tb.tb_lineno
        self.error_type = type(error).__name__
        self.error_message = str(error)
        super().__init__(self.__str__())
        
        # Log the error
        logger.error(self.__str__())

    def __str__(self):
        return (f"\n[ERROR] {self.system_message}\n"
                f"File: {self.file_name}\n"
                f"Line: {self.line_number}\n"
                f"Type: {self.error_type}\n"
                f"Message: {self.error_message}\n")
