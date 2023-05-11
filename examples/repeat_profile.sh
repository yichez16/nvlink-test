# for i in $(seq 1 100);
# do
#     for j in $(seq 1 10);
#     do
#         mkdir -p ./data/m1_$i
#         mkdir -p ./data/m2_$i
#         mkdir -p ./data/m3_$i
#         mkdir -p ./data/m4_$i
#         mkdir -p ./data/m5_$i
#         nvprof --profile-from-start off --devices 0 --log-file ./data/m1_$i/$j.csv --aggregate-mode off --csv --event-collection-mode kernel -m nvlink_total_data_transmitted ./p2p $i
#         nvprof --profile-from-start off --devices 0 --log-file ./data/m2_$i/$j.csv --aggregate-mode off --csv --event-collection-mode kernel -m nvlink_transmit_throughput ./p2p $i
#         nvprof --profile-from-start off --devices 0 --log-file ./data/m3_$i/$j.csv --aggregate-mode off --csv --event-collection-mode kernel -m nvlink_overhead_data_transmitted ./p2p $i
#         nvprof --profile-from-start off --devices 0 --log-file ./data/m4_$i/$j.csv --aggregate-mode off --csv --event-collection-mode kernel -m nvlink_user_data_transmitted ./p2p $i
#         nvprof --profile-from-start off --devices 0 --log-file ./data/m5_$i/$j.csv --aggregate-mode off --csv --event-collection-mode kernel -m nvlink_user_write_data_transmitted ./p2p $i

#         # nvprof --profile-from-start off --devices 0 --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_transmit_throughput ./p2p $i
#         # nvprof --profile-from-start off --devices 0 --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_overhead_data_transmitted ./p2p $i
#         # nvprof --profile-from-start off --devices 0 --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_total_write_data_transmitted ./p2p $i
#         # nvprof --profile-from-start off --devices 0 --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_user_data_transmitted ./p2p $i
#         # nvprof --profile-from-start off --devices 0 --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_user_write_data_transmitted ./p2p $i
#         # # nvprof --profile-from-start off --devices 0 --aggregate-mode off --csv --event-collection-mode continuous -m  ./p2p $i
#         # nvprof --profile-from-start off --aggregate-mode off --csv --log-file ./data/log_continuous_1_$i.csv --event-collection-mode continuous -m nvlink_overhead_data_transmitted,nvlink_overhead_data_received ./p2p
#         # nvprof --profile-from-start off --aggregate-mode off --csv --log-file ./data/log_continuous_2_$i.csv --event-collection-mode continuous -m nvlink_total_response_data_received,nvlink_user_response_data_received ./p2p
#         # nvprof --profile-from-start off --aggregate-mode off --csv --log-file ./data/log_continuous_3_$i.csv --event-collection-mode continuous -m nvlink_user_data_transmitted,nvlink_user_data_received ./p2p
#         # nvprof --profile-from-start off --aggregate-mode off --csv --log-file ./data/log_continuous_4_$i.csv --event-collection-mode continuous -m nvlink_total_write_data_transmitted ./p2p
#         # nvprof --profile-from-start off --aggregate-mode off --csv --log-file ./data/log_continuous_5_$i.csv --event-collection-mode continuous -m nvlink_user_write_data_transmitted ./p2p

#     done
# done
for i in $(seq 1 100);
do
    # nvprof --profile-from-start off --aggregate-mode off --devices 0 --csv -m nvlink_total_nratom_data_transmitted,nvlink_total_ratom_data_transmitted,nvlink_total_response_data_received,nvlink_user_nratom_data_transmitted,nvlink_user_ratom_data_transmitted,nvlink_user_response_data_received ./p2p_vecAdd $i &>> vecAdd.csv
    ./main

done



# nvprf  xxx ./p2p 100 > 1.csv

# nvlink_total_data_transmitted,nvlink_total_data_received,nvlink_transmit_throughput,nvlink_receive_throughput,nvlink_overhead_data_transmitted,nvlink_overhead_data_received,nvlink_total_response_data_received,nvlink_user_response_data_received,nvlink_total_write_data_transmitted,nvlink_user_data_transmitted,nvlink_user_data_received,nvlink_user_write_data_transmitted

# for i in $(seq 1 10);
# do
#     # nvprof --profile-from-start off --aggregate-mode off --csv --log-file ./data/log_kernel_$i.csv --event-collection-mode kernel -m nvlink_total_data_transmitted,nvlink_total_data_received,nvlink_transmit_throughput,nvlink_receive_throughput,nvlink_overhead_data_transmitted,nvlink_overhead_data_received,nvlink_total_response_data_received,nvlink_user_response_data_received,nvlink_total_write_data_transmitted,nvlink_user_data_transmitted,nvlink_user_data_received,nvlink_user_write_data_transmitted ./p2p
    

#     nvprof --profile-from-start off --aggregate-mode off --csv --event-collection-mode kernel -m nvlink_transmit_throughput ./p2p $i
#     # nvprof --profile-from-start on --aggregate-mode off --csv --event-collection-mode kernel -m nvlink_total_data_transmitted ./p2p


# done


