from email.policy import default
import re
import argparse
import sys

sys.path[0]+='/../'

from typing import List, Tuple

from translation_utils.data_utils import read_data, write_data, cjk_deseg


def valid_tgt_data(tgt_data: List[Tuple[int, str]]):
    if len(tgt_data) == 0:
        print("Empty tgt data")
        return
    assert tgt_data[0][0] == 0, tgt_data[0][0]
    for i in range(1, len(tgt_data)):
        if tgt_data[i][0] != tgt_data[i-1][0] + 1:
            print(tgt_data[i])
            print('error at line {}'.format(i))
            return False
    return True

def extract_data(args, translation_data, output_path, str='D'):
    tgt_data=[]
    for line in translation_data:
        if line.startswith(f"{str}-"):
            line = line[2:].split("\t")
            assert 2 <= len(line) <= 3
            line_num = int(line[0])
            tgt = line[2] if len(line) == 3 else line[1]
            tgt_data.append((line_num, tgt))
    tgt_data.sort(key=lambda item: item[0])
    if valid_tgt_data(tgt_data):
        tgt_data = [item[1] for item in tgt_data]
        if args.desegment_zh:
            tgt_data = [cjk_deseg(line) for line in tgt_data]
        print(output_path)
        write_data(tgt_data, output_path)
    else:
        write_data([""], output_path)
        print("Error in translation file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation_file_path", required=True)
    parser.add_argument("--output_hp_file_path", required=True)
    parser.add_argument("--output_ref_file_path", default='')
    parser.add_argument("--desegment_zh", action="store_true")

    args = parser.parse_args()

    translation_data = read_data(args.translation_file_path)
    if args.output_ref_file_path!='':
        extract_data(args, translation_data, args.output_ref_file_path, 'T')

    extract_data(args, translation_data, args.output_hp_file_path, 'D')


if __name__ == "__main__":
    main()
