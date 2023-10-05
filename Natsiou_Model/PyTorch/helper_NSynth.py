from torchdata.datapipes.iter import IterableWrapper, FileOpener
import os


def get_name(path_and_stream):
    return os.path.basename(path_and_stream[0]), path_and_stream[1]


def parse_json_metadata():
    datapipe1 = IterableWrapper(["nsynth-train/examples.json"])
    datapipe2 = FileOpener(datapipe1, mode="b")
    datapipe3 = datapipe2.map(get_name)
    json_dp = datapipe3.parse_json_files()
    return list(json_dp)


if __name__ == '__main__':
    json_list = parse_json_metadata()
    for key in json_list[0][1]:
        print(key, "->", json_list[0][1][key])
