import os
import glob


def get_all_csv_files(directory):
    file_list = glob.glob(os.path.join(directory, '*.csv'))
    return file_list
