#!/bin/env python

import vastdb, re, argparse, logging
from sparknlp.base import DocumentAssembler, Pipeline, ReadAs
from sparknlp.annotator import Tokenizer, Lemmatizer, SentimentDetector
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower
from pyspark.sql.types import StringType
from langdetect import detect
from langcodes import Language
import pyarrow as pa
import pandas as pd

##
# Configuration
logging.basicConfig(level=logging.WARN)
SPARK_LOG_LEVEL="WARN"
SPARK_LOG_LOCATION="/home/vastdata/spark_logs"

def load_blocklist(filepath):
    try:
        with open(filepath, 'r') as file:
            keywords = file.read().strip().split('\n')
        pattern = '|'.join([r'\b' + re.escape(kw.strip()) + r'\b' for kw in keywords if kw.strip() != ''])
        return pattern
    except Exception as e:
        print(f"Error loading blocklist: {e}")
        return None


def find_matched_keywords(body, pattern):
    try:
        compiled_pattern = re.compile(pattern, flags=re.IGNORECASE)
        matches = compiled_pattern.findall(body)
        return ", ".join(set(matches))
    except Exception as e:
        print(f"Error finding matched keywords: {e}")
        return ""


def setup_archive_table(spark, schema, archive_table):
    try:
        spark.sql(f"CREATE TABLE IF NOT EXISTS `ndb`.`{args.db}`.`{args.schema}`.`{args.archive_table}` (id STRING, body STRING, restricted_words STRING)")
    except Exception as e:
        print(f"Error setting up archive table: {e}")


def detect_language(text):
    try:
        code = detect(text)
        language = Language.get(code).display_name()
        return language
    except Exception:
        return "Unknown"


def ensure_language_column(session, bucket_name, schema_name, table_name):
    try:
        with session.transaction() as tx:
            table = tx.bucket(bucket_name).schema(schema_name).table(table_name)
            existing_columns = [field.name for field in table.columns()]
            if 'language' not in existing_columns:
                new_column = pa.schema([('language', pa.string())])
                table.add_column(new_column)
                print("Language column added to the table.")
            else:
                print("Language column already exists.")
    except Exception as e:
        print(f"Error ensuring language column: {e}")


def execute_blocklist(args, spark_session, vdb_session):

    # list of ugly stuff
    blocklist_regex = load_blocklist(args.blocklist_file)
    
    if blocklist_regex:
        try:
            setup_archive_table(spark_session, args.schema, args.archive_table)
            find_keywords_udf = udf(lambda text: find_matched_keywords(text, blocklist_regex), StringType())
    
            total_rows_deleted = 0
            total_rows_archived = 0
            archive_table_name = f"`ndb`.`{args.db}`.`{args.schema}`.`{args.archive_table}`"
    
            with vdb_session.transaction() as tx:

                table = tx.bucket(args.db).schema(args.schema).table(args.table)
                record_batches = table.select(columns=["id", "body"], internal_row_id=True)
    
                pandas_df = pd.concat([batch.to_pandas() for batch in record_batches])
                spark_df = spark_session.createDataFrame(pandas_df)
                
                spark_df = spark_df.withColumn("restricted_words", find_keywords_udf(col("body")))
                filtered_df = spark_df.filter(lower(col("body")).rlike(blocklist_regex))
                
                rows_to_delete = filtered_df.count()
                total_rows_deleted += rows_to_delete
    
                filtered_df.select("id", "body", "restricted_words").write.mode("append").saveAsTable(archive_table_name)
                total_rows_archived += rows_to_delete
    
                if rows_to_delete > 0:
                    filtered_pandas_df = filtered_df.toPandas()
                    filtered_arrow = pa.Table.from_pandas(filtered_pandas_df, preserve_index=False)
                    with vdb_session.transaction() as tx:
                        table = tx.bucket(args.db).schema(args.schema).table(args.table)
                        table.delete(filtered_arrow)
                        print(f"Deleting: {rows_to_delete} Rows > From Table: {args.table} ")
    
            print(f"Total rows blocklisted and deleted: {total_rows_deleted}")
            print(f"Total rows archived: {total_rows_archived}")
        except Exception as e:
            print(f"Error during blocklisting process: {e}")


def execute_lang_detect(args, spark_session, vdb_session):

    try:
        ensure_language_column(vdb_session, args.db, args.schema, args.table)
    
        total_rows_updated = 0
    
        with vdb_session.transaction() as tx:
            
            # !! in the driver !!
            table = tx.bucket(args.db).schema(args.schema).table(args.table)
            record_batches = table.select(columns=["id", "body"], internal_row_id=True)
            pandas_df = pd.concat([batch.to_pandas() for batch in record_batches])

            # Spark context
            spark_df = spark_session.createDataFrame(pandas_df)
            spark_df = spark_df.withColumn("language", detect_language_udf(col("body")))
            update_df = spark_df.select("id", "language", "$row_id").toPandas()
            arrow_table = pa.Table.from_pandas(update_df, preserve_index=False)
            
        with vdb_session.transaction() as tx:

             # !! in the driver !!
            table = tx.bucket(args.db).schema(args.schema).table(args.table)
            table.update(arrow_table, ["language"])

            # Spark context
            updated_language_counts = spark_session.createDataFrame(update_df)
            language_count = updated_language_counts.filter(updated_language_counts.language.isNotNull()).count()
            total_rows_updated += language_count
                
        print(f"Total rows language updates: {total_rows_updated}")

    except Exception as e:
        print(f"Error during language detection process: {e}")


def execute_sentiment(args, spark_session):

    table_path = f"`ndb`.`{args.db}`.`{args.schema}`.`{args.table}`"

    documentAssembler = DocumentAssembler() \
        .setInputCol("body") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    lemmatizer = Lemmatizer() \
        .setInputCols(["token"]) \
        .setOutputCol("lemma") \
        .setDictionary("/tmphx-70/jason/llmcc/lemmas_small.txt", "->", "\t")

    sentimentDetector = SentimentDetector() \
        .setInputCols(["lemma", "document"]) \
        .setOutputCol("sentimentScore") \
        .setDictionary("/tmphx-70/jason/llmcc/default-sentiment.dict", ",", ReadAs.TEXT)

    pipeline = Pipeline().setStages([
        documentAssembler,
        tokenizer,
        lemmatizer,
        sentimentDetector,
    ])

    table_df = spark_session.read.table(table_path)

    record_batches = table_df.filter(table_df.language.contains("English")).select("id", "body")

    result = pipeline.fit(record_batches).transform(record_batches)
    #result.selectExpr("sentimentScore").show(100, truncate=False) 
    result.show(2, truncate=True) 



def parse_args():
    parser = argparse.ArgumentParser(description='Process VASTDB tables for language detection and blocklisting.')
    parser.add_argument('--db', default="db0", help='Database name in VAST')
    parser.add_argument('--schema', default="common_crawl", help='Schema name in VASTDB')
    parser.add_argument('--table', default="cc_news", help='Table name in VASTDB')
    parser.add_argument('--blocklist_file', default="/tmphx-70/jason/llmcc/bad.txt", help='Path to the blocklist file')
    parser.add_argument('--archive_table', help='Archive table name for storing blocklisted rows')
    parser.add_argument('--access-key', default="KTYJE7EBRPXFA8LW40RT", help="VAST DB API access key")
    parser.add_argument('--secret-key', default="CrCK8xPdTNtUo+vXzXDukRFeDQYL7Q9XThEb3iQh", help="VAST DB API secret key")
    parser.add_argument('--endpoint', default="http://11.0.0.1:8070", help="vastdb endpoint")
    args = parser.parse_args()
    if not args.archive_table:
        args.archive_table = f"{args.table}_blocklisted"
    return args

if __name__ == "__main__":
    args = parse_args()
    spark = SparkSession.builder \
        .config("spark.driver.extraJavaOptions", "-DSPARK_LOGS_DIR={log_loc} -DSPARK_ROLE=app_driver -XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=12 -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/home/vastdata/spark_log/gc_driver.log".format(log_loc=SPARK_LOG_LOCATION)) \
        .config("spark.executor.extraJavaOptions", "-DSPARK_LOGS_DIR={log_loc} -DSPARK_ROLE=app_executor -XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=12 -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:/home/vastdata/spark_logs/gc_executor.log".format(log_loc=SPARK_LOG_LOCATION)) \
        .appName("PySpark Pre-process Ingest & Normalize to VASTDB") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.4.0") \
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
        .config("spark.sql.debug.maxToStringFields", "1000") \
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
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    vdb_session = vastdb.connect(endpoint=args.endpoint,
                             access=args.access_key,
                             secret=args.secret_key)
    
    # Register language detection function
    detect_language_udf = udf(detect_language, StringType())
    
    print("language detection...")
    #execute_lang_detect(args, spark, vdb_session)

    print("blocklisting...")
    #execute_blocklist(args, spark, vdb_session)

    print("sentiment detection...")
    execute_sentiment(args, spark)

    spark.stop()
