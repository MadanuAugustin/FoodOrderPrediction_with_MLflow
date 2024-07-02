

import sys##The python sys module provides functions and variables which are used to manipulate different parts of--
##--the Python Runtime Environment
## Any exception that is getting controlled automatically the sys will have all that information.
import logging

# creating our own custom exception function

def error_message_detail(error,error_detail:sys):## error parameter is basically an exception message.
# error_detail is present inside the sys

    _,_,exc_tb=error_detail.exc_info()## exc_info() will give you the info on which file the error has occured and in which line.--
    ##it returns three output. all the error info is stored in exc_tb variable.
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message

    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error = error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message