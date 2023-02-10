
#!/bin/bash
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 9
#SBATCH --gpus-per-node 8
#SBATCH -w 16,17,18,19

export NCCL_DEBUG=WARN

best_or_last=best
enc_langtok=src
capacity_factor=0.5
ratio1=0.
ratio2=0.
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--save_dir)
            save_dir=$2
            shift
            shift
            ;;
        -s|--subset)
            subset=$2
            shift
            shift
            ;;
        -l|--last)
            best_or_last=last
            shift
            ;;
        -c|--capacity_factor )
            capacity_factor=$2
            shift
            shift
            ;;
        -r1|--ratio1)
            ratio1=$2
            shift
            shift
        -r2|--ratio2)
            ratio2=$2
            shift
            shift
        -*|--*)
            echo "unkown option $1"
            exit 1
            ;;
    esac
done
# save
echo save_dir=$save_dir
echo subset=$subset
echo best_or_last=$best_or_last
echo ratio1=$ratio1
echo ratio2=$ratio2

model_name=(${save_dir//// })
model_name=${model_name[-1]}
model_name=${model_name}${prefix}
translation_dir=./translation_data/opus-100/hir-moe/$model_name
score_path=bleu/opus-100/hir-moe/$model_name.bleu

echo "model_name:${model_name}"
echo "prefix:${prefix}"
echo "translation_dir:${translation_dir}"
echo "score_path:${score_path}"

#distributed
master_addr="127.0.0.3"
master_port=12345
n_process=8

# data
root_data_dir=/path/to/opus100
main_data_bin_dir=${root_data_dir}/many-many/main_data_bin
extra_data_bin_dir=${root_data_dir}/many-many/extra_data_bin

spm_data_dir=${root_data_dir}/many-many/spm_data
spm_corpus_dir=${root_data_dir}/many-many/spm_corpus

max_tokens=6000

all_lang_pairs="en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh"
lang_dict=${root_data_dir}/lang_dict.txt

python=/path/to/python
sacrebleu=/path/to/sacrebleu

echo "spm decode complete!"
checkpoint='checkpoint_best'
checkpoint_path="${save_dir}/${checkpoint}.pt"

mkdir -p ${translation_dir}

lang_pairs=${all_lang_pairs//,/ }
result_path=${translation_dir}
echo "write translation to ${translation_dir}"

# for generate_multiple.py, --source-lang and --target-lang does not work, it would iterate all languages in lang-pairs-to-generate
srun ${python} generate_multiple.py ${main_data_bin_dir} \
--distributed-port 12346 \
--task translation_multi_simple_epoch \
--user-dir ./user_dir \
--distributed-world-size ${n_process} \
--lang-pairs ${all_lang_pairs} \
--lang-dict ${lang_dict} \
--source-dict ${main_data_bin_dir}/dict.txt \
--target-dict ${main_data_bin_dir}/dict.txt \
--decoder-langtok \
--encoder-langtok src \
--source-lang en \
--target-lang fr \
--gen-subset ${subset} \
--path ${checkpoint_path} \
--max-tokens ${max_tokens} \
--beam 5 \
 --results-path ${result_path} \
--post-process sentencepiece \
--lang-pairs-to-generate $lang_pairs \
--is-moe \
--model-overrides "{'moe_eval_capacity_token_fraction':${capacity_factor}, 'ratio1':${ratio1}, 'ratio2':${ratio2} }" \
--ddp-backend fully_sharded \

for lang_pair in ${lang_pairs// / }; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}

    parallel_trans_dir=${translation_dir}/${lang_pair}
    echo "compute bleu for ${lang_pair}"
    ${python} -u ./translation_utils/extract_translation.py \
        --translation_file_path ${parallel_trans_dir}/generate-${subset}.txt \
        --output_hp_file_path ${parallel_trans_dir}/extract.${subset}.txt
    
    score=$(${python} ${sacrebleu} -l ${lang_pair} -w 6 ${spm_corpus_dir}/${lang_pair}/spm_decode.${subset}.${src_lang}-${tgt_lang}.${tgt_lang} < ${parallel_trans_dir}/extract.${subset}.txt)
    
    score=$(echo $score | grep -Po "=\s(\d+\.*\d*)" | head -n 1 | grep -Po "\d+\.*\d*")
    
    echo "${lang_pair}: ${score}" >> ${score_path}
    echo "${lang_pair}: ${score}"
done
echo "average bleu score:"
./translation_utils/average_bleu.py ${score_path}