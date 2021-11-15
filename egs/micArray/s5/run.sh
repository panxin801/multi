#!/bin/bash

start_stage=1
end_stage=1

if [ ${start_stage} -le 0 ]  && [ ${end_stage} -ge 0 ]; then
    bash prep_data.sh
fi

if [ ${start_stage} -le 1 ]  && [ ${end_stage} -ge 1 ]; then
    bash train.sh
fi

if [ ${start_stage} -le 2 ]  && [ ${end_stage} -ge 2 ]; then
    bash decode_test.sh
fi

if [ ${start_stage} -le 3 ]  && [ ${end_stage} -ge 3 ]; then
    bash avg.sh
fi

if [ ${start_stage} -le 4 ]  && [ ${end_stage} -ge 4 ]; then
    bash score.sh data/test/text exp/exp1/decode_test_avg-last10
fi
