import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq
MODE = 'label'
dir = f'/mnt/d/python/graduation_thesis/{MODE}_by_stock'

files = sorted(glob.glob(os.path.join(dir, '*.parquet')))
assert len(files) > 0, "没有找到 factor parquet 文件"

tables = []
base_cols = None

for fp in files:
    tb = pq.read_table(fp)

    if base_cols is None:
        base_cols = tb.column_names
    else:
        if tb.column_names != base_cols:
            raise ValueError(f'列不一致: {fp}')

    tables.append(tb)

data_all = pa.concat_tables(tables, promote_options="default")

print(data_all.num_rows, data_all.num_columns)

pq.write_table(data_all, f'/mnt/d/python/graduation_thesis/{MODE}_all.parquet')