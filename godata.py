from urllib.request import urlretrieve
from urllib.error import URLError
from bs4 import BeautifulSoup
from os import path, getcwd, makedirs, remove, rename, environ
from collections import UserDict, deque
from gobot import GoBoardAI

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from keras.utils import to_categorical
from glob import glob
from sgfmill import sgf
from sgfmill.sgf import Sgf_game
import tarfile
from goboard import *
import numpy as np
from itertools import product, cycle
from typing import Callable, Union, Tuple, Iterator, List, Set, Mapping
import shelve
import random
import contextlib
from numpy import deprecate
import tqdm
import multiprocessing
import queue

__all__ = ["download_dir", "kgs_archive_folder", "db_file",
           "SgfFile", "SgfData", "SgfDataBase"]

download_dir = path.join(path.dirname(path.realpath(__file__)), ".data")
kgs_archive_folder = "kgs"
db_file = "go_games.db"
db_samples = "go_samples.db"


class SgfFile(NamedTuple):
    size: int
    winner: GoPlayer
    sequence: List[GoPoint]
    komi: float
    handicap: int
    setup_stones: Tuple[Optional[Set[GoPoint]], Optional[Set[GoPoint]], Optional[Set[GoPoint]]]
    first_player: GoPlayer

    def encode_policy(self, dtype=np.float16) -> Tuple[np.ndarray, np.ndarray]:
        b = GoBoardAI(self.size, self.first_player)
        b.setup_stones(*self.setup_stones)
        inputs = []
        outputs = []
        for pos in self.sequence:
            if pos is not None:
                inputs.append(b.encode_input(dtype=dtype))
                outputs.append(b.encode_policy_output(pos, dtype=dtype))
            b.play(pos)
        return np.array(inputs, dtype=dtype), np.array(outputs, dtype=dtype)

    def encode_value(self, dtype=np.float16) -> Tuple[np.ndarray, np.ndarray]:
        b = GoBoardAI(self.size, self.first_player)
        b.setup_stones(*self.setup_stones)
        inputs = []
        outputs = []
        index = random.randint(0, len(self.sequence) - 1)
        for _, pos in zip(range(index), self.sequence):
            b.play(pos)
            inputs.append(b.encode_input(dtype=np.float16))
            outputs.append(b.encode_value_output(self.winner, dtype=np.float16))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    @classmethod
    def from_sgf(cls, sgf_game: Sgf_game):
        size = sgf_game.get_size()
        winner = GoPlayer.to_player(sgf_game.get_winner())
        sequence: List[GoPoint] = []
        first_player = GoPlayer.none
        for node in sgf_game.get_main_sequence():
            player, point = node.get_move()
            if first_player is not None:
                if first_player == GoPlayer.none:
                    first_player = GoPlayer.to_player(player)
                sequence.append(point)
        komi = sgf_game.get_komi()
        handicap = sgf_game.get_handicap()
        setup_stones = sgf_game.get_root().get_setup_stones()
        return cls(size, winner, sequence, komi, handicap, setup_stones, first_player)


class SgfData:

    def download_begin(self, url: str, filename: str, downloading: bool = True):
        if downloading:
            self.bar = tqdm.tqdm(desc="Downloading %s from %s..." % (path.basename(filename), url)
                                 , unit="B", unit_scale=True)

    def downloading_callback(self, block_num: int, block_size: int, total_size: int):
        self.bar.reset(total_size)
        self.bar.update(block_size)

    def downloading_end(self, url: str, filename: str, downloading: bool):
        if downloading:
            self.bar.set_description_str("Downloaded %s successfully!" % path.basename(filename))
            self.bar.close()
            self.bar = None
            #   tqdm.tqdm.write("Downloaded %s successfully!" % path.basename(filename))

    def __init__(self):
        self.bar = None

    def retrieve(self, url: str, filename: str, force: bool = False) -> str:
        download_filename = filename if path.isabs(filename) else path.join(download_dir, filename)
        makedirs(path.dirname(download_filename), exist_ok=True)
        if force or not path.exists(download_filename):
            self.download_begin(url, download_filename, True)
            tmp_name = download_filename + ".download"
            urlretrieve(url, tmp_name, self.downloading_callback)
            rename(tmp_name, download_filename)
            self.downloading_end(url, download_filename, True)
        return download_filename

    def retrieve_sgf_archive(self, force: bool = False):
        kgs_index = self.retrieve('http://u-go.net/gamerecords/', "kgs_index.html", force)
        with open(kgs_index, "r") as f:
            soup = BeautifulSoup(f.read(), "lxml")
            links = soup.select('body>div>div:nth-child(2)>div.col-md-10>table>tr>td:nth-child(6)>a')
            for link in links:
                archive_url: str = link.get("href")
                if archive_url.endswith('.tar.gz'):
                    self.retrieve(archive_url, path.join(kgs_archive_folder, path.basename(archive_url)),
                                  force)

    def calculate_total(self):
        folder = path.join(download_dir, kgs_archive_folder)
        archives = glob(path.join(folder, "*.tar.gz"))
        number_of_files = 0
        for i_archive, archive_name in enumerate(archives):
            with tarfile.open(archive_name) as archive:
                for name in archive.getnames():
                    if name.endswith(".sgf"):
                        number_of_files += 1
        return number_of_files

    def iter(self):
        folder = path.join(download_dir, kgs_archive_folder)
        archives = glob(path.join(folder, "*.tar.gz"))
        for archive_name in archives:
            with tarfile.open(archive_name) as archive:
                files_name = filter(lambda name: name.endswith(".sgf"), archive.getnames())
                for file_name in files_name:
                    self.bar.set_description_str(
                        "Extracting %s from %s..." % (path.basename(file_name), path.basename(archive_name)))
                    with archive.extractfile(file_name) as file:
                        self.bar.update(1)
                        yield file_name, SgfFile.from_sgf(Sgf_game.from_bytes(file.read()))

    def extract_sgf_archive(self):
        self.bar = tqdm.tqdm(desc="Preparing to extract sgf files...", unit="files", unit_scale=False)
        self.bar.reset(self.calculate_total())
        with shelve.open(path.join(download_dir, db_file)) as db:
            for file_name, sgf_file in self.iter():
                if sgf_file.first_player == GoPlayer.none:
                    continue
                file = path.splitext(path.basename(file_name))[0]
                db[file] = sgf_file
        self.bar.set_description_str("Extract successfully")
        self.bar.close()
        self.bar = None

    def get(self, archive: str, file: str) -> SgfFile:
        if not path.isabs(archive):
            folder = path.join(download_dir, kgs_archive_folder)
            archive = path.join(folder, archive)
        with tarfile.open(archive) as archive:
            with archive.extractfile(file) as f:
                return SgfFile.from_sgf(sgf.Sgf_game.from_bytes(f.read()))


class SgfDataBase(UserDict):

    def __init__(self):
        super().__init__()
        self.data: shelve.Shelf = shelve.open(path.join(download_dir, db_file), "r")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data.close()

    def close(self):
        self.data.close()

    def sample(self, num: int, size: int = 19) -> List[SgfFile]:
        samples = random.sample(list(self.keys()), num)
        return list(filter(lambda sgf_game: sgf_game.size == size, (self[key] for key in samples)))

    @staticmethod
    def _policy_process_producer(works: multiprocessing.Queue, result: multiprocessing.Queue, dtype=np.float16):
        sgf_file: Optional[SgfFile] = works.get()
        while True:
            if sgf_file is not None:
                x, y = sgf_file.encode_policy(dtype=dtype)
                result.put((x, y))
            else:
                break
        # result.put(None)  # end process

    def extract_policy_network(self, numbers: Optional[int] = None, dtype=np.float16, size=19):
        bar = tqdm.tqdm(desc="Preparing to calculate sgf files...", unit="files", unit_scale=False)
        if numbers is None:
            numbers = len(self)
        bar.reset(numbers)
        bar.set_description("Calculating...")

        with shelve.open(path.join(download_dir, db_samples), writeback=True) as db:
            db["policy_network"] = []
            for sgf_file in self.sample(numbers, size):
                db["policy_network"].append(sgf_file.encode_policy(dtype=dtype))
                bar.update(1)
        bar.set_description_str("Done")
        bar.close()

    def policy_sample_generator(self, num: int, dtype=np.float16,
                                size: int = 19):
        samples = random.sample(list(self.keys()), num)
        for key in cycle(samples):
            sgf_game = self[key]
            if sgf_game.size == size:
                yield sgf_game.encode_policy(dtype)


class SgfSampleDataBase(UserDict):
    def __init__(self):
        super().__init__()
        self.data: shelve.Shelf = shelve.open(path.join(download_dir, db_samples), "r")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data.close()

    def close(self):
        self.data.close()

    def sample_generator(self, num: int, key="policy_network"):
        samples = random.sample(self[key], num)
        for sample in cycle(samples):
            yield sample

    def sample(self,num:int, key="policy_network"):
        samples = random.sample(self[key], num)
        X,Y=[],[]
        for x,y in samples:
            X.append(x)
            Y.append(y)
        return np.concatenate(X),np.concatenate(Y)
@deprecate()
def del_empty():
    with shelve.open(path.join(download_dir, db_file)) as db:
        for k, v in tqdm.tqdm(db.items()):
            if v.first_player is GoPlayer.none:
                del db[k]


@deprecate()
def check_all():
    with SgfDataBase() as data:
        # white_first,black_first,none = 0 ,0 ,0
        # c = Counter(sgf_file.first_player for sgf_file in tqdm.tqdm(data.values()))
        # print(c)
        for name, sgf_file in tqdm.tqdm(data.items()):
            b = GoBoard(19, sgf_file.first_player)
            try:
                b.setup_stones(*sgf_file.setup_stones)
                for pos in sgf_file.sequence:
                    b.play(pos)
            except GoIllegalActionError as e:
                print("Exception:%s" % name)
                print(sgf_file.sequence)
                print(e.details())
                raise e


if __name__ == "__main__":
    pass
    # data = SgfData()
    # data.retrieve_sgf_archive()
    # data.extract_sgf_archive()
    # del_empty()
    # check_all()
    with SgfDataBase() as data:
        data.extract_policy_network(2000)
