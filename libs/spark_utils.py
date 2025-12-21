# libs/spark_utils.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession


def norm_win_path(p: str | Path) -> str:
    return str(Path(p).resolve()).replace("\\", "/")


def file_uri(p: str | Path) -> str:
    return "file:///" + norm_win_path(p).lstrip("/")


def ensure_dir(p: str | Path) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return norm_win_path(p)


def ensure_windows_hadoop(
    *,
    hadoop_home_default: str = r"C:\hadoop",
    spark_tmp_root_default: str = r"C:\spark-tmp",
) -> None:
    if os.name != "nt":
        return

    hadoop_home = norm_win_path(os.getenv("HADOOP_HOME", hadoop_home_default))
    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ["hadoop.home.dir"] = hadoop_home

    path = os.environ.get("PATH", "")
    hbin = f"{hadoop_home}/bin"
    if hbin not in path:
        os.environ["PATH"] = hbin + ";" + path

    spark_tmp_root = ensure_dir(os.getenv("SPARK_TMP_ROOT", spark_tmp_root_default))
    os.environ["SPARK_TMP_ROOT"] = spark_tmp_root

    run_dir = ensure_dir(Path(spark_tmp_root) / f"run-{os.getpid()}")
    os.environ["SPARK_LOCAL_DIRS"] = run_dir


def _find_ivy_jars_dir() -> Path:
    # prioridade: env var -> C:\spark-ivy\jars -> %USERPROFILE%\.ivy2\jars
    env = os.getenv("SPARK_IVY_JARS")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(r"C:\spark-ivy\jars"))
    candidates.append(Path(os.path.expandvars(r"%USERPROFILE%\.ivy2\jars")))

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    raise FileNotFoundError(
        "Não encontrei diretório de jars do Ivy. "
        "Define SPARK_IVY_JARS ou confirma se existe C:\\spark-ivy\\jars / %USERPROFILE%\\.ivy2\\jars."
    )


def _pick_latest(pattern: str, base: Path) -> Path:
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Não encontrei nenhum jar com pattern '{pattern}' em {base}")
    # escolhe o mais recente pelo nome (normalmente inclui versão)
    return matches[-1]


def build_spark(
    app_name: str,
    *,
    master: str = "local[*]",
    s3_endpoint: str = "s3.eu-north-1.amazonaws.com",
    aws_profile: str = "default",
) -> SparkSession:
    ensure_windows_hadoop()

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    os.environ.setdefault("AWS_PROFILE", aws_profile)
    os.environ.setdefault("AWS_SDK_LOAD_CONFIG", "1")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

    spark_local_dirs = os.environ.get("SPARK_LOCAL_DIRS", norm_win_path(r"C:\spark-tmp"))
    java_tmp = spark_local_dirs

    ivy_jars = _find_ivy_jars_dir()

    # ✅ os jars reais no ivy cache têm nomes tipo:
    # org.apache.hadoop_hadoop-aws-3.3.4.jar
    # com.amazonaws_aws-java-sdk-bundle-1.12.262.jar
    # org.wildfly.openssl_wildfly-openssl-1.0.7.Final.jar
    hadoop_aws = _pick_latest("org.apache.hadoop_hadoop-aws-*.jar", ivy_jars)
    aws_sdk = _pick_latest("com.amazonaws_aws-java-sdk-bundle-*.jar", ivy_jars)
    wildfly = _pick_latest("org.wildfly.openssl_wildfly-openssl-*.jar", ivy_jars)

    jars = ",".join([file_uri(hadoop_aws), file_uri(aws_sdk), file_uri(wildfly)])

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)

        # ✅ usar jars locais (evita Ivy em cada run e evita o erro "scheme C")
        .config("spark.jars", jars)

        # S3A
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.path.style.access", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.profile.ProfileCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.profile", aws_profile)

        # python executors
        .config("spark.executorEnv.PYSPARK_PYTHON", sys.executable)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)

        # temp dirs
        .config("spark.local.dir", spark_local_dirs)
        .config("spark.hadoop.hadoop.tmp.dir", java_tmp)
        .config("spark.hadoop.io.native.lib.available", "false")

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
