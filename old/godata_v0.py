
from urllib.request import urlretrieve

from bs4 import BeautifulSoup
from os import path, makedirs, remove, environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

from numpy import deprecate


download_dir = path.join(path.dirname(path.realpath(__file__)), ".data")
kgs_archive_folder = "kgs"
db_file = "go_games.db"


@deprecate(new_name="SgfData.retrieve")
def retrieve(url: str, filename):
    start = time.perf_counter()
    download_filename = filename if path.isabs(filename) else path.join(download_dir, filename)
    makedirs(path.dirname(download_filename), exist_ok=True)

    def print_schedule(block_num: int, block_size: int, total_size: int):
        """
        block_num:当前已经下载的块
        block_size:每次传输的块大小
        total_size:网页文件总大小
        """
        block_size = block_size if block_size > 0 else 1
        total_size = total_size if total_size > 0 else 1
        total_blocks = min(30, total_size // block_size)
        percent = block_num * block_size / total_size
        percent = percent if percent <= 1.0 else 1.0
        progress_a = '█' * int(total_blocks * percent)
        progress_b = '_' * int(total_blocks * (1 - percent))
        during_time = time.perf_counter() - start
        print('''\r{0:^3.2f}%|{1}{2}|{3:.2}s'''.format(percent * 100.0, progress_a, progress_b, during_time),
              end="")

    if not path.exists(download_filename):
        print("Downloading %s from %s..." % (path.basename(download_filename), url))
        try:
            urlretrieve(url, download_filename, print_schedule)
        except:
            if path.exists(download_filename):
                remove(download_filename)
            print("Downloaded failed!")
            raise
        print("\tDownloaded successfully!")
    return download_filename


@deprecate(new_name="SgfData.retrieve_kgs_archive")
def retrieve_kgs_archive():
    kgs_index = retrieve('http://u-go.net/gamerecords/', "kgs_index.html")
    with open(kgs_index, "r") as f:
        soup = BeautifulSoup(f.read(), "lxml")
        links = soup.select('body>div>div:nth-child(2)>div.col-md-10>table>tr>td:nth-child(6)>a')
        for link in links:
            archive_url: str = link.get("href")
            if archive_url.endswith('.tar.gz'):
                retrieve(archive_url, path.join(kgs_archive_folder, path.basename(archive_url) + ".dataset"))
@deprecate()
def print_process_bar(percent: float, length: int, time: float):
    progress_a = '█' * int(length * percent)
    progress_b = '_' * int(length * (1 - percent))
    print('\r{0:^3.2f}%|{1}{2}|{3:.2}s'.format(percent * 100.0, progress_a, progress_b, time),
          end="")


