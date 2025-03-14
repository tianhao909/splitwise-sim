"""
Utility functions for the notebooks.
"""
import os

import numpy as np
import pandas as pd


def baseline_a100_config(num_a100,
                         start_state="baseline",
                         scheduler="token_jsq",
                         h100_cost=4.76,
                         h100_power=44,
                         a100_cost=2.21,
                         a100_power=24.8):
    config = {
        "name": f"Baseline-A100 ({num_a100}P/T)",  # 系统名称，包含A100服务器数量
        "system": "Baseline-A100",  # 系统类型
        "scheduler": f"{scheduler}",  # 调度器类型
        "start_state": start_state,  # 初始状态
        "cluster": f"{num_a100}_0",  # 集群配置，仅使用A100服务器
        "num_servers": num_a100,  # 服务器总数
        "num_a100": num_a100,  # A100服务器数量
        "num_h100": 0,  # H100服务器数量为0
        "num_prompts": num_a100,  # 提示处理的服务器数量
        "num_tokens": num_a100,  # 生成token的服务器数量
        "cost": num_a100 * a100_cost,  # 总成本计算
        "power": num_a100 * a100_power,  # 总功耗计算
    }
    return config  # 返回配置字典


def baseline_h100_config(num_h100,
                         start_state="baseline",
                         scheduler="token_jsq",
                         h100_cost=4.76,
                         h100_power=44,
                         a100_cost=2.21,
                         a100_power=24.8):
    config = {
        "name": f"Baseline-H100 ({num_h100}P/T)",  # 系统名称，包含H100服务器数量
        "system": "Baseline-H100",  # 系统类型
        "scheduler": f"{scheduler}",  # 调度器类型
        "start_state": start_state,  # 初始状态
        "cluster": f"0_{num_h100}",  # 集群配置，仅使用H100服务器
        "num_servers": num_h100,  # 服务器总数
        "num_a100": 0,  # A100服务器数量为0
        "num_h100": num_h100,  # H100服务器数量
        "num_prompts": num_h100,  # 提示处理的服务器数量
        "num_tokens": num_h100,  # 生成token的服务器数量
        "cost": num_h100 * h100_cost,  # 总成本计算
        "power": num_h100 * h100_power,  # 总功耗计算
    }
    return config  # 返回配置字典


def splitwise_ha_config(num_prompt,
                        num_token,
                        start_state="splitwise",
                        scheduler="mixed_pool",
                        h100_cost=4.76,
                        h100_power=44,
                        a100_cost=2.21,
                        a100_power=24.8):
    num_h100 = num_prompt  # H100服务器用于提示处理
    num_a100 = num_token  # A100服务器用于生成token
    config = {
        "name": f"Splitwise-HA ({num_prompt}P, {num_token}T)",  # 系统名称，包含提示和token处理的服务器数量
        "system": "Splitwise-HA",  # 系统类型
        "scheduler": f"{scheduler}",  # 调度器类型
        "start_state": f"{start_state}_1_1",  # 初始状态
        "cluster": f"{num_token}_{num_prompt}",  # 集群配置，A100和H100混合使用
        "num_servers": num_token + num_prompt,  # 服务器总数
        "num_a100": num_token,  # A100服务器数量
        "num_h100": num_prompt,  # H100服务器数量
        "num_prompts": num_prompt,  # 提示处理的服务器数量
        "num_tokens": num_token,  # 生成token的服务器数量
        "cost": num_h100 * h100_cost + num_a100 * a100_cost,  # 总成本计算
        "power": num_h100 * h100_power + num_a100 * a100_power,  # 总功耗计算
    }
    return config  # 返回配置字典


def splitwise_aa_config(num_prompt,
                        num_token,
                        start_state="splitwise",
                        scheduler="mixed_pool",
                        h100_cost=4.76,
                        h100_power=44,
                        a100_cost=2.21,
                        a100_power=24.8):
    num_a100 = num_prompt + num_token  # 所有任务均由A100服务器处理
    config = {
        "name": f"Splitwise-AA ({num_prompt}P, {num_token}T)",  # 系统名称，包含提示和token处理的服务器数量
        "system": "Splitwise-AA",  # 系统类型
        "scheduler": f"{scheduler}",  # 调度器类型
        "start_state": f"{start_state}_{num_prompt}_{num_token}",  # 初始状态
        "cluster": f"{num_a100}_0",  # 集群配置，仅使用A100服务器
        "num_servers": num_a100,  # 服务器总数
        "num_a100": num_a100,  # A100服务器数量
        "num_h100": 0,  # H100服务器数量为0
        "num_prompts": num_prompt,  # 提示处理的服务器数量
        "num_tokens": num_token,  # 生成token的服务器数量
        "cost": num_a100 * a100_cost,  # 总成本计算
        "power": num_a100 * a100_power,  # 总功耗计算
    }
    return config  # 返回配置字典


def splitwise_hh_config(num_prompt,
                        num_token,
                        start_state="splitwise",
                        scheduler="mixed_pool",
                        h100_cost=4.76,
                        h100_power=44,
                        a100_cost=2.21,
                        a100_power=24.8):
    num_h100 = num_prompt + num_token  # 所有任务均由H100服务器处理
    config = {
        "name": f"Splitwise-HH ({num_prompt}P, {num_token}T)",  # 系统名称，包含提示和token处理的服务器数量
        "system": "Splitwise-HH",  # 系统类型
        "scheduler": f"{scheduler}",  # 调度器类型
        "start_state": f"{start_state}_{num_prompt}_{num_token}",  # 初始状态
        "cluster": f"0_{num_h100}",  # 集群配置，仅使用H100服务器
        "num_servers": num_h100,  # 服务器总数
        "num_a100": 0,  # A100服务器数量为0
        "num_h100": num_h100,  # H100服务器数量
        "num_prompts": num_prompt,  # 提示处理的服务器数量
        "num_tokens": num_token,  # 生成token的服务器数量
        "cost": num_h100 * h100_cost,  # 总成本计算
        "power": num_h100 * h100_power,  # 总功耗计算
    }
    return config  # 返回配置字典


def splitwise_hhcap_config(num_prompt,
                           num_token,
                           start_state="splitwisehhcap",
                           scheduler="mixed_pool",
                           h100_cost=4.76,
                           h100_power=44,
                           a100_cost=2.21,
                           a100_power=24.8,
                           power_cap_scaler=0.7):
    num_h100 = num_prompt + num_token  # 所有任务均由H100服务器处理
    config = {
        "name": f"Splitwise-HHcap ({num_prompt}P, {num_token}T)",  # 系统名称，包含提示和token处理的服务器数量
        "system": "Splitwise-HHcap",  # 系统类型
        "scheduler": f"{scheduler}",  # 调度器类型
        "start_state": f"{start_state}_1_1",  # 初始状态
        "cluster": f"{num_token}_{num_prompt}",  # 集群配置，A100和H100混合使用
        "num_servers": num_h100,  # 服务器总数
        "num_a100": 0,  # A100服务器数量为0
        "num_h100": num_h100,  # H100服务器数量
        "num_prompts": num_prompt,  # 提示处理的服务器数量
        "num_tokens": num_token,  # 生成token的服务器数量
        "cost": num_h100 * h100_cost,  # 总成本计算
        "power": num_prompt * h100_power + num_token * h100_power * power_cap_scaler,  # 总功耗计算，考虑功率限制
    }
    return config  # 返回配置字典


def get_summary_data(results_dir, scheduler, start_state, cluster, trace, seed, model=""):
    try:
        summary_df = pd.read_csv(f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/summary.csv")  # 尝试读取summary.csv文件
    except Exception as e:
        print(e)  # 打印异常信息
        print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/summary.csv")  # 打印失败信息
        return None  # 返回None表示读取失败
    return summary_df  # 返回读取的数据框


def get_request_data(results_dir, scheduler, start_state, cluster, trace, seed, model=""):
    try:
        request_df = pd.read_csv(f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/detailed/0.csv")  # 尝试读取详细请求数据
    except:
        print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/detailed/0.csv")  # 打印失败信息
        return None  # 返回None表示读取失败
    return request_df  # 返回读取的数据框


def get_request_nodes(results_dir, scheduler, start_state, cluster, trace, seed, model=""):
    try:
        request_nodes_df = pd.read_csv(f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/request_nodes.csv")  # 尝试读取请求节点数据
        request_nodes_df["start_timestamp_dt"] = pd.to_datetime(request_nodes_df["start_timestamp"], unit="s")  # 转换开始时间戳为日期时间格式
        request_nodes_df["completion_timestamp_dt"] = pd.to_datetime(request_nodes_df["completion_timestamp"], unit="s")  # 转换完成时间戳为日期时间格式
    except:
        print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/request_nodes.csv")  # 打印失败信息
        return None  # 返回None表示读取失败
    return request_nodes_df  # 返回读取的数据框


def get_instances_data(results_dir, scheduler, start_state, cluster, num_servers, trace, seed, model=""):
    try:
        instance_dfs = []  # 初始化实例数据框列表
        application_id = 0  # 应用ID
        for idx in range(num_servers):  # 遍历所有服务器
            filename = f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/instances/{application_id}/{idx}.csv"  # 构造文件路径
            filepath = os.path.join(results_dir, filename)  # 拼接完整路径
            df = pd.read_csv(filepath)  # 读取CSV文件
            df["iteration"] = range(len(df))  # 添加迭代次数列
            instance_dfs.append(df)  # 将数据框添加到列表
        instances_df = pd.concat(instance_dfs)  # 合并所有实例数据框
        instances_df["iteration_start_dt"] = pd.to_datetime(instances_df["iteration_start"], unit="s")  # 转换迭代开始时间戳为日期时间格式
        instances_df["iteration_end_dt"] = pd.to_datetime(instances_df["iteration_end"], unit="s")  # 转换迭代结束时间戳为日期时间格式
        instances_df["duration"] = (instances_df["iteration_end"] - instances_df["iteration_start"])  # 计算持续时间
        instances_df["memory"] /= 1024 * 1024 * 1024  # 将内存单位转换为GB
        return instances_df  # 返回合并后的数据框
    except:
        print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/instances/0/*.csv")  # 打印失败信息
        return None  # 返回None表示读取失败


def get_num_batch_tokens_baseline(instances_df):
    num_batch_tokens = []  # 初始化批量token列表
    for row in instances_df.iterrows():  # 遍历数据框的每一行
        num_batch_tokens.extend(int(row[1]["num_contiguous_iterations"]) * [row[1]["batch_tokens"]])  # 根据连续迭代次数扩展批量token列表
    return num_batch_tokens  # 返回批量token列表


def get_num_batch_tokens_splitwise(instances_df):
    num_prompt_batch_tokens = []  # 初始化提示批量token列表
    num_token_batch_tokens = []  # 初始化生成token批量token列表
    for row in instances_df.iterrows():  # 遍历数据框的每一行
        if row[1]["tag"] == "prompt":  # 如果标签为"prompt"
            num_prompt_batch_tokens.extend(int(row[1]["num_contiguous_iterations"]) * [row[1]["batch_tokens"]])  # 扩展提示批量token列表
        else:  # 否则
            num_token_batch_tokens.extend(int(row[1]["num_contiguous_iterations"]) * [row[1]["batch_tokens"]])  # 扩展生成token批量token列表
    return num_prompt_batch_tokens, num_token_batch_tokens  # 返回提示和生成token的批量token列表


def get_time_duration_batch_tokens(instances_df):
    instances_df = instances_df.copy()  # 复制数据框以避免修改原始数据
    return instances_df.groupby("batch_tokens").sum()["duration"]  # 按批量token分组并计算总持续时间


def count_token_on_prompt_servers(instances_df, request_nodes_df):
    prompt_nodes = instances_df[instances_df["tag"] == "prompt"]["name"].unique()  # 获取提示节点名称
    count = len(request_nodes_df[(request_nodes_df["node_type"] == "TOKEN") & 
                             (request_nodes_df["runner"].isin(prompt_nodes))])  # 统计在提示节点上运行的token数量
    num_requests = request_nodes_df["request_id"].nunique()  # 统计唯一请求ID的数量
    return count, num_requests, len(prompt_nodes)  # 返回统计结果


def get_summary_data_with_config(results_dir, config, trace, seed, model=""):
    scheduler = config["scheduler"]  # 获取调度器类型
    start_state = config["start_state"]  # 获取初始状态
    cluster = config["cluster"]  # 获取集群配置
    return get_summary_data(results_dir, scheduler, start_state, cluster, trace, seed, model)  # 调用get_summary_data函数


def get_request_data_with_config(results_dir, config, trace, seed, model=""):
    scheduler = config["scheduler"]  # 获取调度器类型
    start_state = config["start_state"]  # 获取初始状态
    cluster = config["cluster"]  # 获取集群配置
    return get_request_data(results_dir, scheduler, start_state, cluster, trace, seed, model)  # 调用get_request_data函数


def get_request_nodes_with_config(results_dir, config, trace, seed, model=""):
    scheduler = config["scheduler"]  # 获取调度器类型
    start_state = config["start_state"]  # 获取初始状态
    cluster = config["cluster"]  # 获取集群配置
    return get_request_nodes(results_dir, scheduler, start_state, cluster, trace, seed, model)  # 调用get_request_nodes函数


def get_instances_data_with_config(results_dir, config, trace, seed, model=""):
    scheduler = config["scheduler"]  # 获取调度器类型
    start_state = config["start_state"]  # 获取初始状态
    cluster = config["cluster"]  # 获取集群配置
    num_servers = config["num_servers"]  # 获取服务器数量
    return get_instances_data(results_dir, scheduler, start_state, cluster, num_servers, trace, seed, model)  # 调用get_instances_data函数


def find_within_slo(results_df, slos):
    configs_within_slo = []  # 初始化满足SLO的配置列表
    for system_name in results_df["system"].unique():  # 遍历每个系统
        system_df = results_df[results_df["system"] == system_name]  # 获取当前系统的数据
        for key, value in slos.items():  # 遍历SLO条件
            system_df = system_df[system_df[f"{key}"] < value]  # 筛选满足SLO的数据
        configs_within_slo.append(system_df)  # 将满足SLO的系统数据添加到列表
    return pd.concat(configs_within_slo)  # 返回合并后的数据框


def find_cheapest(results_df):
    configs = []  # 初始化最便宜配置列表
    for system_name in results_df["system"].unique():  # 遍历每个系统
        system_df = results_df[results_df["system"] == system_name]  # 获取当前系统的数据
        cheapest = system_df[system_df["cost"] == system_df["cost"].min()]  # 找到成本最低的配置
        configs.append(cheapest)  # 将最便宜配置添加到列表
    return pd.concat(configs)  # 返回合并后的数据框


def find_least_power(results_df):
    configs = []  # 初始化最低功耗配置列表
    for system_name in results_df["system"].unique():  # 遍历每个系统
        system_df = results_df[results_df["system"] == system_name]  # 获取当前系统的数据
        least_power = system_df[system_df["power"] == system_df["power"].min()]  # 找到功耗最低的配置
        configs.append(least_power)  # 将最低功耗配置添加到列表
    return pd.concat(configs)  # 返回合并后的数据框


def find_least_count(results_df):
    configs = []  # 初始化最少服务器数量配置列表
    for system_name in results_df["system"].unique():  # 遍历每个系统
        system_df = results_df[results_df["system"] == system_name]  # 获取当前系统的数据
        least_count = system_df[system_df["num_servers"] == system_df["num_servers"].min()]  # 找到服务器数量最少的配置
        configs.append(least_count)  # 将最少服务器数量配置添加到列表
    return pd.concat(configs)  # 返回合并后的数据框


def find_max_throughput(results_df):
    if "throughput" not in results_df.columns:  # 如果数据框中没有吞吐量列
        results_df["throughput"] = results_df["trace"].apply(lambda x: int(x.split("_")[2]))  # 根据trace字段计算吞吐量
    configs = []  # 初始化最大吞吐量配置列表
    for system_name in results_df["system"].unique():  # 遍历每个系统
        system_df = results_df[results_df["system"] == system_name]  # 获取当前系统的数据
        max_throughput = system_df[system_df["throughput"] == system_df["throughput"].max()]  # 找到吞吐量最大的配置
        configs.append(max_throughput)  # 将最大吞吐量配置添加到列表
    return pd.concat(configs)  # 返回合并后的数据框
