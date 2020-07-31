from .new_test_main_host import the_entire_process
from celery import shared_task
import pandas as pd

@shared_task
def the_entire_process_function_approx(input_data_df, target, should_i_boost, test_data_df, context):
	return the_entire_process(input_data_df, target, should_i_boost, test_data_df, context)
