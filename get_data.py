import os
import requests

DOWNLOAD_URL = "https://raw.githubusercontent.com/chrisjmccormick/MinHash/master/data/"
CUR_DIR = __file__[0:-11]
DATA = [100, 1000, 2500, 10000]


# download and save file
def download_and_save(remote_url, local_file):
    data = requests.get(remote_url)
    # save to file if it isn't exist
    if not os.path.isfile(local_file):
        try:
            with open(local_file, "wb") as file:
                file.write(data.content)
        except IOError as error:
            print(error)

# download and save data to "data" directory
def fetch_data(download_url = DOWNLOAD_URL, cur_dir = CUR_DIR, data_size_arr = DATA):
    '''
        download data for project
        > input: 
            > download_url: url to the data used.
            > cur_dir: current diretory.
            > data_size_arr: array of data size used. 
        > output: none
    '''
    # data directory path
    data_dir = cur_dir + "data"
    # create data folder if it isn't exist
    try: 
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir) 
    except OSError as error:
        print(error)
    # get and save data
    for data_size in data_size_arr:
        remote_url = download_url + 'articles_' + str(data_size)
        local_file = data_dir + '/articles_' + str(data_size)
        download_and_save(remote_url + ".train", local_file + ".txt")
        download_and_save(remote_url + ".truth", local_file + ".truth.txt")
# ---------------------------------------------------------------------------------------

# main
if __name__ == "__main__":
    fetch_data()