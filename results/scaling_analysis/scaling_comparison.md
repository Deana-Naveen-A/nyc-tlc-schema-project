# NYC TLC Scaling Experiment Results

| dataset                         | contract_status   |   requested_rows |   records_used |   batch_size |   execution_time_seconds |   peak_process_memory_mb |    mae |    rmse | source_file                 |
|:--------------------------------|:------------------|-----------------:|---------------:|-------------:|-------------------------:|-------------------------:|-------:|--------:|:----------------------------|
| yellow_tripdata_2024-01.parquet | VALID             |            10000 |          10000 |        50000 |                   0.5380 |                 189.9000 | 3.5825 |  7.1344 | scaling_results_10000.json  |
| yellow_tripdata_2024-01.parquet | VALID             |            50000 |          50000 |        50000 |                   0.5120 |                 196.3600 | 2.7495 |  6.3246 | scaling_results_50000.json  |
| yellow_tripdata_2024-01.parquet | VALID             |           100000 |         100000 |        50000 |                   0.5450 |                 205.5800 | 2.8534 | 12.5028 | scaling_results_100000.json |
| yellow_tripdata_2024-01.parquet | VALID             |           500000 |         500000 |        50000 |                   1.7190 |                 261.5400 | 2.4137 |  6.1987 | scaling_results_500000.json |
