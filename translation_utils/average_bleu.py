import sys

if len(sys.argv)>1:
    fpath=sys.argv[1]
    with open(fpath, 'r') as fi:
        data=fi.read()
else:
    data=sys.stdin.read()

lang_bleu=dict()
for l in data.split('\n'):
    if not (len(l.strip())>0 and len(l.split())>1):
        continue
    lang, bleu=l.split()[0], float(l.split()[1])
    # if lang in lang_bleu and lang_bleu[lang]!=bleu:
        # print(f'Warning: mulitple BLEU scores of language {lang}, overfit with the newest score')
    lang_bleu[lang]=bleu

all_bleu=lang_bleu.values()
en_any=[b for l,b in lang_bleu.items() if l.startswith('en')]
any_en=[b for l,b in lang_bleu.items() if l.endswith('en:')]

def avg_of_list(l):
    return sum(l)/len(l)

print('avg bleu:-------------')
print('language number: ', len(all_bleu))
print('all:', avg_of_list(all_bleu))
print('en-any:', avg_of_list(en_any))
print('any-en:', avg_of_list(any_en))