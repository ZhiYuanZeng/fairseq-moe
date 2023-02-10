import sys

bleu_path1=sys.argv[1]
bleu_path2=sys.argv[2]


def read_bleu(fpath):
    with open(fpath, 'r') as fi:
        data=fi.read()
    lang_bleu=dict()
    for l in data.split('\n'):
        if not (len(l.strip())>0 and len(l.split())>1):
            continue
        lang, bleu=l.split()[0], float(l.split()[1])
        # if lang in lang_bleu and lang_bleu[lang]!=bleu:
        #     print(f'Warning: mulitple BLEU scores of language {lang}, overfit with the newest score')
        lang_bleu[lang]=bleu
    return lang_bleu

def get_win_rate(bleu1, bleu2):
    win_num=0.
    # assert len(bleu1)==len(bleu2), 'len of bleu1:{}, len of bleu2:{}'.format(len(bleu1), len(bleu2))
    for k,v in bleu1.items():
        assert k in bleu2, f'{k} not in file1'
        if bleu1[k]>bleu2[k]:
            win_num+=1
    return win_num/len(bleu1)

def avg_bleu(bleu):
    def avg_of_list(l):
        return sum(l)/len(l)
    all_bleu=bleu.values()
    en_any=[b for l,b in bleu.items() if l.startswith('en')]
    any_en=[b for l,b in bleu.items() if l.endswith('en:')]
    return avg_of_list(all_bleu), avg_of_list(en_any), avg_of_list(any_en)

if __name__=='__main__': 
    bleu1=read_bleu(bleu_path1)
    bleu2=read_bleu(bleu_path2)

    all_bleu1, en_any1, any_en1=avg_bleu(bleu1)
    all_bleu2, en_any2, any_en2=avg_bleu(bleu2)
    print('file1: avg {}, en-any {}, any-en {}'.format(all_bleu1, en_any1, any_en1))
    print('file2: avg {}, en-any {}, any-en {}'.format(all_bleu2, en_any2, any_en2))
    print('win-rate of file1 on file2:{}'.format(get_win_rate(bleu1, bleu2)))
