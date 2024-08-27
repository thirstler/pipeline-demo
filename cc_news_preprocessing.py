#!/bin/env python

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, hash
from bs4 import BeautifulSoup
from pyspark.sql.types import StringType, LongType, TimestampType
from datetime import datetime
import sys
import os
import glob
import time
import re
import argparse
import logging

##
# Configuration
logging.basicConfig(level=logging.WARN)
SPARK_LOG_LEVEL="WARN"
SPARK_LOG_LOCATION="/home/vastdata/spark_logs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default="/tmphx-70/jason/llmcc/cc-news/2024/01", help="directory to find files in")
    parser.add_argument('--limit', default=56, type=int, help="how many parquet files to process")
    parser.add_argument('--db', default="db0", help="Database Name in VAST")
    parser.add_argument('--schema', default="common_crawl", help="vastdb schema name")
    parser.add_argument('--table', default="cc_news", help="vastdb table name")
    parser.add_argument('--access-key', default="KTYJE7EBRPXFA8LW40RT", help="VAST DB API access key")
    parser.add_argument('--secret-key', default="CrCK8xPdTNtUo+vXzXDukRFeDQYL7Q9XThEb3iQh", help="VAST DB API secret key")
    parser.add_argument('--endpoint', default="http://11.0.0.1:8070", help="vastdb endpoint")
    
    args = parser.parse_args()
    if not args.table:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.table = f"cc_table_{current_time}"
    return args
    
def remove_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s,.!?]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    
def remove_garbage_text(text):
    '''
    Thank you Simon!
    '''
    patterns = [
        r'^get\s.*\shttp', r'^useragent', r'accept', r'^host', r'^connection', r'^keepalive', 
        r'charset', r'xcrawler', r'http1\.1', r'contenttype', r'xpoweredby', r'setcookie', 
        r'pragma', r'vary', r'accesscontrol', r'expires', r'cachecontrol', r'cf-ray', r'cfnel', 
        r'nel', r'link', r'reportto', r'alt-svc', r'xcontenttypeoptions', r'xframeoptions', 
        r'xxssprotection', r'contentsecuritypolicy', r'^--\s', r'<!doctype', r'<!\[endif\]-->', 
        r'texthtml', r'contentlength', r'nocache', r'lastmodified', r'timeexpire', r'contentencoding', 
        r'xcachestatus', r'transferencoding', r'sameorigin', r'xoptions', r'xcachekey', r'relamphtml', 
        r'relcanonical', r'xnsfwcontent', r'xppapreloadlayer', r'xppapreloadlayer', r'xcachenext', 
        r'xtraceid', r'xdownloadoptions', r'unsafeinline', r'maxage3600', r'nosniff', r'scriptsrc', 
        r'httponly', r'allowcredentials', r'allowheaders', r'allowmethods', r'alloworigin'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text



def parse_html(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=' ')
        text = remove_urls(text)
        text = clean_text(text)
        text = remove_garbage_text(text)
        if sys.getsizeof(text) > filterSize:
            return None
        return text
    except Exception as e:
        print(f"Error parsing HTML: {e}", file=sys.stderr)
        return None
        
parse_html_udf = udf(parse_html, StringType())

def start_transaction(spark):
    spark.sql("SELECT ndb.create_tx()")
    
def commit_transaction(spark):
    spark.sql("SELECT ndb.commit_tx()")
    
def rollback_transaction(spark):
    spark.sql("SELECT ndb.rollback_tx()")
    
def process_batch(batch_files, spark, catalog_table_name, batch_index):

    try:
        batch_start_time = time.time()
        start_transaction(spark)
        df = spark.read.parquet(*batch_files).filter(col("content_length") >= 7000)

        # Parse HTML: remove URLS, clean formatting, remove junk text
        df = df.withColumn("date", df["date"].cast(TimestampType())) \
               .withColumn("body", parse_html_udf(df["body"].cast(StringType())))

        # Remove duplicates
        print("deduplicating...", end="", flush=True)
        df_filtered = df.filter(df["body"].isNotNull())
        df_hashed = df_filtered.withColumn("hash_value", hash(col("body")).cast(LongType()))
        deduplicated_df = df_hashed.dropDuplicates(["hash_value"])
        print("done")

        # Write to VAST DB table
        print("saving batch to table...")
        deduplicated_df.write.mode("append").saveAsTable(catalog_table_name)
        commit_transaction(spark)
        print("done")

        print(f"Batch {batch_index} processed in {time.time() - batch_start_time} seconds")

    except Exception as e:
        print(f"Error processing batch: {e}", file=sys.stderr)
        rollback_transaction(spark)
        exit()


def create_tables(args, spark):

    res = spark.sql("SHOW SCHEMAS FROM ndb")
    print(res.collect())

    sql = f"CREATE DATABASE IF NOT EXISTS `ndb`.`{args.db}`.`{args.schema}`"
    print(sql)
    spark.sql(sql)
    
    
    table_path = f"`ndb`.`{args.db}`.`{args.schema}`.`{args.table}`"
    print(f"Created Table in VastDB: {table_path}")

    return table_path


if __name__ == "__main__":
    
    filterSize = 126023
    print("Size Filter: ", filterSize)

    args = parse_args()
    spark = SparkSession.builder \
        .config("spark.driver.extraJavaOptions", "-DSPARK_LOGS_DIR={log_loc} -DSPARK_ROLE=app_driver -XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=12 -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/home/vastdata/spark_log/gc_driver.log".format(log_loc=SPARK_LOG_LOCATION)) \
        .config("spark.executor.extraJavaOptions", "-DSPARK_LOGS_DIR={log_loc} -DSPARK_ROLE=app_executor -XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=12 -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/home/vastdata/spark_logs/gc_executor.log".format(log_loc=SPARK_LOG_LOCATION)) \
        .appName("PySpark Pre-process Ingest & Normalize to VASTDB") \
        .config("spark.executor.userClassPathFirst", "True") \
        .config("spark.task.maxFailures", 1) \
        .config("spark.sql.shuffle.partitions", "128") \
        .config("spark.default.parallelism", "128") \
        .config("spark.shuffle.file.buffer", "1MB") \
        .config("spark.reducer.maxSizeInFlight", "96MB") \
        .config("spark.shuffle.io.retryWait", "60s") \
        .config("spark.shuffle.io.maxRetries", "10") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.memory.offHeap.enabled", "True") \
        .config("spark.memory.offHeap.size", "3g") \
        .config("spark.sql.files.maxPartitionBytes", "256m") \
        .config("spark.locality.wait", "2s") \
        .config("spark.network.timeout", "3600s") \
        .config("spark.port.maxRetries", "30") \
        .config("spark.shuffle.push.finalize.timeout", "1000s") \
        .config("spark.shuffle.service.removeShuffle", "True") \
        .config("spark.ui.showConsoleProgress", "true") \
        .config("spark.shuffle.push.finalize.timeout", "1000s") \
        .config("spark.sql.parquet.columnarReaderBatchSize", "4096") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.optimizer.runtime.bloomFilter.enabled", "True") \
        .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "True") \
        .config("spark.sql.optimizer.dynamicPartitionPruning.useStats", "true") \
        .config("spark.sql.optimizer.dynamicPartitionPruning.fallbackFilterRatio", "0.5") \
        .config("spark.sql.optimizer.runtimeFilter.semiJoinReduction.enabled", "true") \
        .config("spark.sql.cbo.enabled", "True") \
        .config("spark.sql.cbo.joinReorder.enabled", "True") \
        .config("spark.sql.cbo.planStats.enabled", "True") \
        .config("spark.sql.cbo.joinReorder.dp.star.filter", "True") \
        .config("spark.sql.cbo.starSchemaDetection", "True") \
        .config("spark.sql.join.preferSortMergeJoin", "False") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.sql.readSideCharPadding", "True") \
        .config("spark.sql.charAsVarchar", "False") \
        .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
        .config("spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold", "64MB") \
        .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "2") \
        .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "104857600") \
        .config("spark.sql.adaptive.enabled", "True") \
        .config("spark.sql.sources.fileCompressionFactor", "5.0") \
        .config("spark.sql.statistics.histogram.enabled", "False") \
        .config("spark.sql.cbo.joinReorder.dp.threshold", "12") \
        .config("spark.sql.exchange.reuse", "True") \
        .config("spark.sql.execution.reuseSubquery", "True") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.debug.maxToStringFields", "100") \
        .config("spark.ndb.endpoint", args.endpoint) \
        .config("spark.ndb.data_endpoints", args.endpoint) \
        .config("spark.ndb.access_key_id", args.access_key) \
        .config("spark.ndb.secret_access_key", args.secret_key) \
        .config("spark.ndb.num_of_splits", "128") \
        .config("spark.ndb.num_of_sub_splits", "10") \
        .config("spark.ndb.rowgroups_per_subsplit", "1") \
        .config("spark.ndb.query_data_rows_per_split", "4000000") \
        .config("spark.ndb.retry_max_count", "3") \
        .config("spark.ndb.retry_sleep_duration", "1") \
        .config("spark.ndb.parallel_import", "true") \
        .config("spark.ndb.dynamic_filter_compaction_threshold", "100") \
        .config("spark.ndb.dynamic_filtering_wait_timeout", "2") \
        .config("spark.sql.catalog.ndb", "spark.sql.catalog.ndb.VastCatalog") \
        .config("spark.sql.extensions", "ndb.NDBSparkSessionExtension") \
        .config("spark.python.authenticate.socketTimeout", "1m") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    
    total_start_time = time.time()

    # Create DB elements if they're not there
    table_path = create_tables(args, spark)
    
    # Get list of parquet flies
    all_files = glob.glob(os.path.join(args.directory, '**', '*.parquet'), recursive=True)
    total_files = min(len(all_files), args.limit)
    batch_size = min(args.limit, 56)
    print(f"Total files to process: {total_files}, Batch Size: {batch_size}")
    
    # Process batches of filese
    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i+batch_size]
        process_batch(batch_files, spark, table_path, i//batch_size + 1)
        

    print(f"Total ingestion time: {time.time() - total_start_time} seconds")
    
    spark.stop()