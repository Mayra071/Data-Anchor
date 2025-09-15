import sys
from src.logger import logging

# create a function to get detailed error message
def get_detailed_error_message(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    line_number = exc_tb.tb_lineno
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with error message: {str(error)}"
    return error_message



# Custom exception class for handling exceptions 
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_detailed_error_message(error_message, error_detail)
        
    def __str__(self):
        return self.error_message