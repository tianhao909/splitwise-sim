import os  # 导入os模块，用于操作系统相关的功能

from collections import namedtuple  # 导入namedtuple，用于创建具名元组

import requests  # 导入requests模块，用于发送HTTP请求

import numpy as np  # 导入numpy库，用于科学计算
import pandas as pd  # 导入pandas库，用于数据处理和分析

from scipy import stats  # 导入scipy.stats模块，用于统计分布相关操作


Distributions = namedtuple('Distributions', ['application_id',
                                             'request_type',
                                             'arrival_process',
                                             'batch_size',
                                             'prompt_size',
                                             'token_size'])  # 定义一个具名元组Distributions，用于存储 各种分布信息
Distribution = namedtuple('Distribution', ['name', 'params'])  # 定义一个具名元组Distribution，用于存储 分布名称和参数


def generate_samples(distribution, params, size):
    """
    Generate random samples from the given distribution.
    """
    if distribution == "constant":  # 如果分布类型是常数分布
        return np.ones(size) * params["value"]  # 返回大小为size的数组，所有值为params["value"]
    elif distribution == "normal":  # 如果分布类型是正态分布
        return stats.norm(**params).rvs(size=size)  # 使用scipy生成正态分布样本
    elif distribution == "truncnorm":  # 如果分布类型是截断正态分布
        return stats.truncnorm(**params).rvs(size=size)  # 使用scipy生成截断正态分布样本
    elif distribution == "randint":  # 如果分布类型是随机整数分布
        return stats.uniform(**params).rvs(size=size)  # 使用scipy生成均匀分布样本（注意：这里实现可能有问题）
    elif distribution == "uniform":  # 如果分布类型是均匀分布
        return stats.uniform(**params).rvs(size=size)  # 使用scipy生成均匀分布样本
    elif distribution == "exponential":  # 如果分布类型是指数分布
        return stats.expon(**params).rvs(size=size)  # 使用scipy生成指数分布样本
    elif distribution == "poisson":  # 如果分布类型是泊松分布
        return stats.poisson(**params).rvs(size=size)  # 使用scipy生成泊松分布样本
    elif distribution == "trace":  # 如果分布类型是基于轨迹文件的分布
        df = pd.read_csv(params["filename"])  # 读取指定的CSV文件
        return df[params["column"]].sample(size, replace=True).values  # 从指定列中随机采样
    else:  # 如果分布类型无效
        raise ValueError(f"Invalid distribution: {distribution}")  # 抛出异常


def generate_trace(max_requests, distributions, end_time=None):
    """
    Generate a trace of requests based on the given distributions.
    """
    # Generate request IDs
    request_ids = np.arange(max_requests)  # 生成从0到max_requests-1的请求ID数组

    # Generate the distributions
    arrival_timestamps = generate_samples(distributions.arrival_process.name,
                                          distributions.arrival_process.params,
                                          max_requests)  # 根据到达过程分布生成时间戳样本
    arrival_timestamps = np.cumsum(arrival_timestamps)  # 计算累积和，得到实际到达时间
    application_ids = generate_samples(distributions.application_id.name,
                                       distributions.application_id.params,
                                       max_requests)  # 根据应用ID分布生成样本
    application_ids = map(int, application_ids)  # 将应用ID转换为整数
    batch_sizes = generate_samples(distributions.batch_size.name,
                                   distributions.batch_size.params,
                                   max_requests)  # 根据批处理大小分布��成样本
    batch_sizes = map(int, batch_sizes)  # 将批处理大小转换为整数
    prompt_sizes = generate_samples(distributions.prompt_size.name,
                                    distributions.prompt_size.params,
                                    max_requests)  # 根据提示大小分布生成样本
    prompt_sizes = map(int, prompt_sizes)  # 将提示大小转换为整数
    token_sizes = generate_samples(distributions.token_size.name,
                                   distributions.token_size.params,
                                   max_requests)  # 根据令牌大小分布生成样本
    token_sizes = map(int, token_sizes)  # 将令牌大小转换为整数
    request_type_ids = generate_samples(distributions.request_type.name,
                                        distributions.request_type.params,
                                        max_requests)  # 根据请求类型分布生成样本
    request_type_ids = map(int, request_type_ids)  # 将请求类型转换为整数

    # Combine the arrays into a DataFrame
    trace_df = pd.DataFrame({
        "request_id": request_ids,  # 请求ID
        "request_type": request_type_ids,  # 请求类型
        "application_id": application_ids,  # 应用ID
        "arrival_timestamp": arrival_timestamps,  # 到达时间戳
        "batch_size": batch_sizes,  # 批处理大小
        "prompt_size": prompt_sizes,  # 提示大小
        "token_size": token_sizes,  # 令牌大小
    })  # 将所有数据组合成一个DataFrame

    if end_time is not None:  # 如果指定了结束时间
        trace_df = trace_df[trace_df["arrival_timestamp"] < end_time]  # 过滤掉超过结束时间的请求

    return trace_df  # 返回生成的轨迹DataFrame


def get_exponential_scale(num_servers, utilization, request_duration):
    """
    assumes that request_duration is in seconds
    """
    interarrival_time = request_duration / (1.0 * utilization)  # 计算到达间隔时间
    exponential_scale = interarrival_time / num_servers  # 计算指数分布的尺度参数
    return exponential_scale  # 返回指数分布的尺度参数


def generate_trace_from_utilization(
    max_requests,
    end_time,
    num_servers,
    utilization,
    request_duration,
    pt_distributions_file):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    exponential_scale = get_exponential_scale(num_servers, utilization, request_duration)  # 获取指数分布的尺度参数
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),  # 应用ID为常数0
        request_type=Distribution("constant", {"value": 2}), # 2表示LLM推理请求
        arrival_process=Distribution("exponential", {"scale": exponential_scale}),  # 到达过程为指数分布
        prompt_size=Distribution("trace", {"filename": pt_distributions_file,
                                           "column": "ContextTokens"}),  # 提示大小基于轨迹文件中的ContextTokens列
        token_size=Distribution("trace", {"filename": pt_distributions_file,
                                          "column": "GeneratedTokens"}),  # 令牌大小基于轨迹文件中的GeneratedTokens列
        batch_size=Distribution("constant", {"value": 1}),  # 批处理大小为常数1
    )  # 定义分布配置

    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)  # 生成轨迹数据
    return trace_df  # 返回生成的轨迹DataFrame


def generate_trace_from_prompt_token_size_distributions(
    max_requests,
    end_time,
    request_rate,
    pt_distributions_filename):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),  # 应用ID为常数0
        request_type=Distribution("constant", {"value": 2}), # 2表示LLM推理请求
        arrival_process=Distribution("exponential", {"scale": 1.0 / request_rate}),  # 到达过程为指数分布
        prompt_size=Distribution("trace", {"filename": pt_distributions_filename,
                                           "column": "ContextTokens"}),  # 提示大小基于轨迹文件中的ContextTokens列
        #prompt_size=Distribution("truncnorm", {"a": (prompt_min-prompt_mean)/prompt_std,
        #                                       "b": (prompt_max-prompt_mean)/prompt_std,
        #                                       "loc": prompt_mean,
        #                                       "scale": prompt_std}),  # 提示大小为截断正态分布（注释掉）
        token_size=Distribution("trace", {"filename": pt_distributions_filename,
                                          "column": "GeneratedTokens"}),  # 令牌大小基于轨迹文件中的GeneratedTokens列
        #token_size=Distribution("truncnorm", {"a": (token_min-token_mean)/token_std,
        #                                      "b": (token_max-token_mean)/token_std,
        #                                      "loc": token_mean,
        #                                      "scale": token_std}),  # 令牌大小为截断正态分布（注释掉）
        batch_size=Distribution("constant", {"value": 1}),  # 批处理大小为常数1
    )  # 定义分布配置
    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)  # 生成轨迹数据
    return trace_df  # 返回生成的轨迹DataFrame


def generate_traces(max_requests,
                    end_time,
                    request_rates,
                    pt_distributions_file,
                    trace_filename_template):
    """
    Generate traces with prompt/token size distributions.
    """
    for request_rate in request_rates:  # 遍历每个请求率
        trace_df = generate_trace_from_prompt_token_size_distributions(
            max_requests,
            end_time,
            request_rate,
            pt_distributions_file)  # 生成轨迹数据
        trace_filename = trace_filename_template.format(request_rate)  # 根据模板生成文件名
        trace_df.to_csv(trace_filename, index=False)  # 将轨迹数据保存为CSV文件


def generate_code_traces(
    max_requests,
    end_time,
    request_rates,
    code_distributions_file,
    trace_filename_template="traces/rr_code_{}.csv"):
    """
    code traces distribution
    prompt_mean = 2048, prompt_std = 1973, prompt_min = 3, prompt_max = 7437
    token_mean = 28, token_std = 60, token_min = 6, token_max = 1899
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):  # 检查目录是否存在
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])  # 创建目录

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    code_distributions_file,
                    trace_filename_template)  # 生成代码轨迹


def generate_conv_traces(
    max_requests,
    end_time,
    request_rates,
    conv_distributions_file,
    trace_filename_template="traces/rr_conv_{}.csv"):
    """
    conv traces distribution
    prompt_mean = 1155, prompt_std = 1109, prompt_min = 2, prompt_max = 14050
    token_mean = 211, token_std = 163, token_min = 7, token_max = 1000
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):  # 检查目录是否存在
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])  # 创建目录

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    conv_distributions_file,
                    trace_filename_template)  # 生成对话轨迹


def download_file(url, filename):
    """
    Download a file from the given URL.
    """
    response = requests.get(url)  # 发送HTTP GET请求下载文件
    with open(filename, "wb") as f:  # 打开文件以二进制写模式
        f.write(response.content)  # 将下载的内容写入文件


def download_azure_llm_traces():
    """
    Download traces from the given URL.
    """
    if not os.path.exists("data"):  # 检查data目录是否存在
        os.makedirs("data")  # 创建data目录

    url_base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"  # Azure公共数据集的基础URL

    if not os.path.exists("data/code_distributions.csv"):  # 检查代码分布文件是否存在
        url = url_base + "AzureLLMInferenceTrace_code.csv"  # 构造代码分布文件的URL
        download_file(url, "data/code_distributions.csv")  # 下载代码分布文件
        print("Downloaded code traces")  # 打印下载完成信息

    if not os.path.exists("data/conv_distributions.csv"):  # 检查对话分布文件是否存在
        url = url_base + "AzureLLMInferenceTrace_conv.csv"  # 构造对话分布文件的URL
        download_file(url, "data/conv_distributions.csv")  # 下载对话分布文件
        print("Downloaded conv traces")  # 打印下载完成信息


if __name__ == "__main__":
    # download prompt and token size distributions
    download_azure_llm_traces()  # 下载Azure LLM轨迹数据

    # generate request traces
    generate_code_traces(
        max_requests=1000000,  # 最大请求数
        end_time=600,  # 结束时间（秒）
        request_rates=list(range(30, 251, 10)),  # 请求率范围
        code_distributions_file="data/code_distributions.csv")  # 代码分布文件路径
    print("Generated code traces")  # 打印生成完成信息

    generate_conv_traces(
        max_requests=1000000,  # 最大请求数
        end_time=600,  # 结束时间（秒）
        request_rates=list(range(30, 251, 10)),  # 请求率范围
        conv_distributions_file="data/conv_distributions.csv")  # 对话分布文件路径
    print("Generated conv traces")  # 打印生成完成信息

    # generate request traces for 2 min
    generate_code_traces(
        max_requests=1000000,  # 最大请求数
        end_time=120,  # 结束时间（秒）
        request_rates=list(range(30, 101, 10)),  # 请求率范围
        code_distributions_file="data/code_distributions.csv",  # 代码分布文件路径
        trace_filename_template="traces/rr_code_{}_2min.csv")  # 文件名模板
    print("Generated code 2min traces")  # 打印生成完成信息

    generate_conv_traces(
        max_requests=1000000,  # 最大请求数
        end_time=120,  # 结束时间（秒）
        request_rates=list(range(30, 101, 10)),  # 请求率范围
        conv_distributions_file="data/conv_distributions.csv",  # 对话分布文件路径
        trace_filename_template="traces/rr_conv_{}_2min.csv")  # 文件名模板
    print("Generated conv 2min traces")  # 打印生成完成信息
