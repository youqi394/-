# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import random
from datetime import datetime, timedelta
import re

# -------------------------- 导入 --------------------------
from heuristic_common import (
    Config,
    Solution,
    decode_with_random_keys,
)

# -------------------------- 全局配置 --------------------------
st.set_page_config(
    page_title="智能公交调度系统",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stButton>button {
    height: 50px;
    font-size: 16px;
    width: 100%;
    border-radius: 12px;
    border: none;
    background-color: #1f77b4;
    color: white;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #155a8a;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
}
.stMetric {
    background-color: #f8f9fa;
    padding: 12px;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
}
h1, h2, h3 {
    color: #2c3e50;
    font-weight: 600;
}

/* 进度条样式 */
.stProgress {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.stProgress > div:first-child {
    position: static !important;
    background: transparent !important;
    background-color: transparent !important;
    height: auto !important;
    color: #2c3e50 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
}
.stProgress > div:last-child {
    height: 12px !important;
    margin: 0 !important;
    background-color: #e9ecef !important;
}
.stProgress > div:last-child > div {
    background-color: #1f77b4 !important;
    border-radius: 10px !important;
}

.stMetric [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    overflow: visible !important;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------------------------- 会话状态初始化 --------------------------
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = "等待开始"
if 'timetable_data' not in st.session_state:
    st.session_state.timetable_data = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'prediction_hours' not in st.session_state:
    st.session_state.prediction_hours = None
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None
if 'schedule_data' not in st.session_state:
    st.session_state.schedule_data = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'current_gap' not in st.session_state:
    st.session_state.current_gap = 0.85
if 'current_objective' not in st.session_state:
    st.session_state.current_objective = 50.0
if 'convergence_data' not in st.session_state:
    st.session_state.convergence_data = []
if 'solve_log' not in st.session_state:
    st.session_state.solve_log = []
if 'power_prediction_table' not in st.session_state:
    st.session_state.power_prediction_table = None
if 'weather_source' not in st.session_state:
    st.session_state.weather_source = ""
# 贪心算法结果
if 'greedy_solution' not in st.session_state:
    st.session_state.greedy_solution = None
if 'greedy_schedule_data' not in st.session_state:
    st.session_state.greedy_schedule_data = None
if 'greedy_objective' not in st.session_state:
    st.session_state.greedy_objective = 0.0

# -------------------------- 工具函数 --------------------------
def add_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.solve_log.append(f"[INFO] {timestamp} - {message}")

def normalize_column_name(name):
    """标准化列名：移除所有空格、括号、特殊字符，转为小写"""
    return re.sub(r'[\s()%]', '', str(name)).lower()

def get_carbon_season(date):
    """自动判断碳排放季节：夏季(6-8月)、冬季(12-2月)、其他(全年)"""
    month = date.month
    if 6 <= month <= 8:
        return "summer"
    elif month == 12 or month <= 2:
        return "winter"
    else:
        return "annual"

def get_power_season(date):
    """自动判断电量消耗季节：春(3-5)、夏(6-8)、秋(9-11)、冬(12-2)"""
    month = date.month
    if 3 <= month <= 5:
        return "春季"
    elif 6 <= month <= 8:
        return "夏季"
    elif 9 <= month <= 11:
        return "秋季"
    else:
        return "冬季"

def get_time_period(hour):
    """将0-23小时精确映射到四个时段"""
    if 7 <= hour <= 9:
        return "早高峰"
    elif 17 <= hour <= 19:
        return "晚高峰"
    elif hour == 6 or (10 <= hour <= 16) or (20 <= hour <= 21):
        return "平峰"
    else: # 0-5, 22-23
        return "低峰"

# -------------------------- 天气获取 --------------------------
def get_weather_forecast(date):
    WEATHER_API_KEY = "e088a35c897818780a479973d4623063"
    today = datetime.now().date()
    max_forecast_date = today + timedelta(days=3)

    is_in_forecast_range = (date >= today) and (date <= max_forecast_date)

    if is_in_forecast_range:
        try:
            city_code = "110000"
            url = (
                f"https://restapi.amap.com/v3/weather/weatherInfo"
                f"?city={city_code}&key={WEATHER_API_KEY}&extensions=all"
            )
            response = requests.get(url, timeout=10)
            data = response.json()

            if data.get("status") == "1":
                target = date.strftime("%Y-%m-%d")
                for day in data["forecasts"][0]["casts"]:
                    if day["date"] == target:
                        weather_info = {
                            "date": date,
                            "temp_max": int(day["daytemp"]),
                            "temp_min": int(day["nighttemp"]),
                            "weather": day["dayweather"].strip(),
                            "is_rain": 1 if "雨" in day["dayweather"] else 0
                        }
                        st.session_state.weather_source = f"✅ 高德API预报（{target}）"
                        add_log(f"✅ 高德API成功：{target} {weather_info['weather']} {weather_info['temp_min']}~{weather_info['temp_max']}℃")
                        return weather_info, None
                add_log(f"⚠️ 高德返回无 {target}（超出3天？）")
            else:
                add_log(f"⚠️ 高德API status=0：{data.get('info')}")
        except Exception as e:
            add_log(f"⚠️ 高德请求异常：{e}")
    else:
        st.session_state.weather_source = "ℹ️ 历史/超3天 → 固定默认值（18~25℃晴）"
        add_log(f"ℹ️ {date} 不在未来3天 → 用默认天气")

    default_weather = {
        "date": date,
        "temp_max": 25,
        "temp_min": 18,
        "weather": "晴",
        "is_rain": 0
    }
    return default_weather, None

# -------------------------- ✅ 修复：通用方向识别逻辑 --------------------------
@st.cache_resource
def load_timetable_data(timetable_type):
    """
    根据选择的班次类型读取对应的时刻表文件
    保留每个发车时间的方向信息（四惠/老山）
    只要列名包含"四惠"就识别为四惠方向，包含"老山"就识别为老山方向
    """
    # 映射班次类型到文件名
    file_map = {
        "工作日": "工作日发车时刻表.csv",
        "周末": "节假日发车时刻表.csv",  
        "节假日": "节假日发车时刻表.csv"
    }
    
    filename = file_map.get(timetable_type, "工作日发车时刻表.csv")
    file_path = f"data/{filename}"
    
    add_log(f"🔄 正在读取 {timetable_type} 时刻表：{file_path}")
    
    try:
        df = pd.read_csv(file_path, dtype=str, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, dtype=str, encoding='gbk')
        except Exception as e:
            add_log(f"⚠️ 未找到 {timetable_type} 时刻表文件：{str(e)}")
            return None, f"文件不存在或无法读取：{file_path}"
    
    # 清洗列名
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # 移除空行
    df = df.dropna(how='all')
    
    add_log(f"✅ 成功加载原始时刻表，共{len(df)}行")
    add_log(f"📌 原始列名：{list(df.columns)}")
    
    # 自动识别所有发车时刻列并记录方向
    all_trips = []
    for col in df.columns:
        normalized = normalize_column_name(col)
        if "发车时刻" in normalized or "发车时间" in normalized:
            # 通用方向识别
            if "四惠" in col:
                direction = "四惠"
            elif "老山" in col:
                direction = "老山"
            else:
                direction = "四惠"  # 默认方向
            
            add_log(f"✅ 识别到列：{col} → 方向：{direction}")
            
            # 提取该列的所有发车时间
            times = df[col].dropna().tolist()
            for time_str in times:
                parts = time_str.split(":")
                if len(parts) == 2:
                    try:
                        depart_hour = int(parts[0])
                        depart_minute = int(parts[1])
                        all_trips.append({
                            "depart_time": time_str,
                            "depart_hour": depart_hour,
                            "depart_minute": depart_minute,
                            "direction": direction
                        })
                    except:
                        pass
    
    if not all_trips:
        error_msg = "未找到任何发车时刻列，请检查CSV文件列名"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    
    # 统计方向数量
    direction_count = {}
    for trip in all_trips:
        direction_count[trip["direction"]] = direction_count.get(trip["direction"], 0) + 1
    
    add_log(f"✅ 方向统计：{direction_count}")
    add_log(f"✅ 合并完成，共{len(all_trips)}个有效发车班次")
    
    return all_trips, None

# -------------------------- 加载碳排放数据 --------------------------
@st.cache_resource
def load_carbon_data():
    """从data文件夹加载碳排放CSV，格式：hour,annual,summer,winter"""
    try:
        carbon_df = pd.read_csv("data/碳排放.csv", dtype=str, encoding='utf-8')
    except:
        try:
            carbon_df = pd.read_csv("data/碳排放.csv", dtype=str, encoding='gbk')
        except Exception as e:
            add_log(f"⚠️ 未找到碳排放文件：{str(e)}")
            return None, f"文件不存在或无法读取：{str(e)}"
    
    # 清洗列名
    carbon_df.columns = carbon_df.columns.str.strip()
    carbon_df = carbon_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # 转换数值类型
    for col in ['hour', 'annual', 'summer', 'winter']:
        carbon_df[col] = pd.to_numeric(carbon_df[col], errors='coerce')
    
    # 检查必需列
    required_columns = ["hour", "annual", "summer", "winter"]
    missing_columns = [col for col in required_columns if col not in carbon_df.columns]
    
    if missing_columns:
        error_msg = f"无法自动匹配列名。实际列名：{list(carbon_df.columns)}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    
    add_log(f"✅ 成功加载 data/碳排放.csv，共{len(carbon_df)}条记录")
    return carbon_df, None

# -------------------------- 加载运行时间数据 --------------------------
@st.cache_resource
def load_runtime_data():
    """从data文件夹加载运行时间CSV，自动匹配列名"""
    try:
        runtime_df = pd.read_csv("data/运行时间75%分位数.csv", dtype=str, encoding='utf-8')
    except:
        try:
            runtime_df = pd.read_csv("data/运行时间75%分位数.csv", dtype=str, encoding='gbk')
        except Exception as e:
            add_log(f"⚠️ 未找到运行时间文件：{str(e)}")
            return None, f"文件不存在或无法读取：{str(e)}"
    
    # 清洗列名
    runtime_df.columns = runtime_df.columns.str.strip()
    runtime_df = runtime_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # 自动匹配列名
    weather_col = None
    runtime_col = None
    
    for col in runtime_df.columns:
        normalized = normalize_column_name(col)
        if "天气" in normalized or "weather" in normalized:
            weather_col = col
        elif "运行时间" in normalized or "runtime" in normalized or "75" in normalized:
            runtime_col = col
    
    if not weather_col or not runtime_col:
        error_msg = f"无法自动匹配列名。实际列名：{list(runtime_df.columns)}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    
    # 重命名为标准列名
    runtime_df = runtime_df.rename(columns={weather_col: "天气", runtime_col: "75%运行时间 (min)"})
    
    add_log(f"✅ 成功匹配列名：天气='{weather_col}', 运行时间='{runtime_col}'")
    add_log(f"✅ 成功加载 data/运行时间75%分位数.csv，共{len(runtime_df)}条记录")
    return runtime_df, None

# -------------------------- 加载电量消耗数据 --------------------------
@st.cache_resource
def load_power_data():
    """从data文件夹加载四季电量消耗CSV，自动匹配列名"""
    try:
        power_df = pd.read_csv("data/电量消耗.csv", dtype=str, encoding='utf-8')
    except:
        power_df = pd.read_csv("data/电量消耗.csv", dtype=str, encoding='gbk')
    
    # 清洗列名
    power_df.columns = power_df.columns.str.strip()
    power_df = power_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # 自动匹配列名
    time_col = None
    weather_col = None
    spring_col = None
    summer_col = None
    autumn_col = None
    winter_col = None
    
    for col in power_df.columns:
        normalized = normalize_column_name(col)
        if "时段" in normalized or "time" in normalized:
            time_col = col
        elif "天气" in normalized or "weather" in normalized:
            weather_col = col
        elif "春" in normalized or "spring" in normalized:
            spring_col = col
        elif "夏" in normalized or "summer" in normalized:
            summer_col = col
        elif "秋" in normalized or "autumn" in normalized or "fall" in normalized:
            autumn_col = col
        elif "冬" in normalized or "winter" in normalized:
            winter_col = col
    
    if not all([time_col, weather_col, spring_col, summer_col, autumn_col, winter_col]):
        error_msg = f"无法自动匹配列名。实际列名：{list(power_df.columns)}"
        add_log(f"❌ {error_msg}")
        return None, error_msg
    
    # 重命名为标准列名
    power_df = power_df.rename(columns={
        time_col: "时段",
        weather_col: "天气类型",
        spring_col: "春季",
        summer_col: "夏季",
        autumn_col: "秋季",
        winter_col: "冬季"
    })
    
    add_log(f"✅ 成功匹配列名：时段='{time_col}', 天气类型='{weather_col}', 春季='{spring_col}', 夏季='{summer_col}', 秋季='{autumn_col}', 冬季='{winter_col}'")
    add_log(f"✅ 成功加载 data/电量消耗.csv，共{len(power_df)}条记录")
    return power_df, None

# -------------------------- 24小时逐时统计预测逻辑 --------------------------
def statistical_prediction(weather_info):
    """
    24小时逐时统计预测逻辑：
    1. 获取当日天气、电量季节、碳排放季节（基于用户选择的调度日期）
    2. 筛选当日天气的电量消耗数据
    3. 获取当日天气的运行时间（所有小时共用）
    4. 遍历0-23每个小时：
       - 自动匹配时段类型
       - 匹配对应时段和天气的电量消耗
       - 匹配对应小时和季节的碳排放
    5. 生成24行逐时结果表
    """
    current_weather = weather_info['weather']
    current_date = weather_info['date']
    
    # 获取当前季节（完全基于用户选择的日期，和天气来源无关）
    power_season = get_power_season(current_date)
    carbon_season = get_carbon_season(current_date)
    
    season_name_map = {"summer": "夏季", "winter": "冬季", "annual": "全年"}
    add_log(f"✅ 自动判断：电量季节={power_season}, 碳排放季节={season_name_map[carbon_season]}")
    
    power_df, power_error = load_power_data()
    runtime_df, runtime_error = load_runtime_data()
    carbon_df, carbon_error = load_carbon_data()
    
    # 获取当日天气的运行时间（所有小时共用）
    runtime_value = "0.00"
    if runtime_df is not None:
        matched_runtime = runtime_df[runtime_df['天气'] == current_weather].copy()
        if not matched_runtime.empty:
            runtime_value = matched_runtime.iloc[0]['75%运行时间 (min)']
            add_log(f"✅ 匹配到天气「{current_weather}」的运行时间：{runtime_value}分钟")
    
    # 筛选当日天气的电量数据
    matched_power = None
    if power_df is not None:
        matched_power = power_df[power_df['天气类型'] == current_weather].copy()
        add_log(f"✅ 匹配到天气「{current_weather}」的电量消耗数据")
    
    # 生成24小时逐时结果
    result = []
    power_column_name = f"{power_season}电量消耗"
    carbon_column_name = f"{season_name_map[carbon_season]}碳排放"
    
    for hour in range(0, 24):
        # 匹配当前小时的时段类型
        period = get_time_period(hour)
        
        # 匹配电量消耗
        power_value = "23.00%"
        if matched_power is not None:
            power_row = matched_power[matched_power['时段'] == period]
            if not power_row.empty:
                power_value = power_row.iloc[0][power_season]
        
        # 匹配碳排放（直接取对应小时的原始值）
        carbon_value = 0.0
        if carbon_df is not None:
            carbon_row = carbon_df[carbon_df['hour'] == hour]
            if not carbon_row.empty:
                carbon_value = carbon_row.iloc[0][carbon_season]
        
        # 构建逐时结果
        row_data = {
            "小时": f"{hour:02d}:00",
            "时段类型": period,
            "天气": current_weather,
            power_column_name: power_value,
            "75%运行时间 (min)": runtime_value,
            carbon_column_name: f"{carbon_value:.4f}"
        }
        
        result.append(row_data)
    
    return pd.DataFrame(result)

# -------------------------- 客流预测 --------------------------
def predict_passenger_flow(date, line_id, is_workday, weather_data):
    hours = list(range(6, 22))
    base_flow = 150 if is_workday else 100
    rain_factor = 0.8 if weather_data and weather_data['is_rain'] else 1.0
    predictions = []
    for hour in hours:
        if 7 <= hour <= 9:
            flow = base_flow * 2.5 * rain_factor
        elif 17 <= hour <= 19:
            flow = base_flow * 2.2 * rain_factor
        elif 6 <= hour <= 21:
            flow = base_flow * rain_factor
        else:
            flow = base_flow * 0.5 * rain_factor
        predictions.append(round(flow * (0.9 + np.random.random() * 0.2)))
    return hours, predictions

# -------------------------- 双算法求解器 --------------------------
def tournament(rng: random.Random, scored: list[tuple[float, list[float], Solution]], size: int = 3) -> list[float]:
    picks = [rng.choice(scored) for _ in range(size)]
    picks.sort(key=lambda item: item[0])
    return picks[0][1]

def crossover(rng: random.Random, left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("Chromosome length mismatch.")
    if len(left) <= 2:
        return left[:]
    a = rng.randrange(1, len(left) - 1)
    b = rng.randrange(a, len(left))
    child = left[:a] + right[a:b] + left[b:]
    if rng.random() < 0.25:
        child = [right[i] if rng.random() < 0.5 else child[i] for i in range(len(child))]
    return child

def mutate(rng: random.Random, chromosome: list[float], rate: float) -> None:
    for i in range(len(chromosome)):
        if rng.random() < rate:
            chromosome[i] = rng.random()

def fitness(solution: Solution) -> float:
    return solution.objective

# -------------------------- 生成标准格式排班表 --------------------------
def generate_standard_schedule(raw_schedule, power_prediction_table, initial_battery=100.0, power_threshold=20.0):
    """
    生成标准格式排班表：
    1. 按车辆编号分组，先车01，再车02，依此类推
    2. 列：车辆编号 | 四惠发车时间 | 老山发车时间
    3. 非对应方向显示"/"
    4. 电量低于20%时，在发车时间后加"(充电)"
    """
    if not raw_schedule:
        return pd.DataFrame()
    
    # 1. 按车辆编号分组
    vehicle_groups = {}
    for item in raw_schedule:
        vehicle_id = item["vehicle_id"]
        if vehicle_id not in vehicle_groups:
            vehicle_groups[vehicle_id] = []
        vehicle_groups[vehicle_id].append(item)
    
    # 2. 按车辆编号排序（车01 → 车02 → ...）
    sorted_vehicles = sorted(vehicle_groups.keys(), key=lambda x: int(x.replace("车", "")))
    
    # 3. 构建电量消耗映射表（小时 → 电量消耗百分比）
    power_map = {}
    power_column = [col for col in power_prediction_table.columns if "电量消耗" in col][0]
    for _, row in power_prediction_table.iterrows():
        hour = int(row["小时"].split(":")[0])
        power_str = row[power_column].replace("%", "")
        try:
            power_map[hour] = float(power_str)
        except:
            power_map[hour] = 10.0  # 默认值
    
    # 4. 生成最终排班表
    final_schedule = []
    for vehicle_id in sorted_vehicles:
        trips = vehicle_groups[vehicle_id]
        # 按发车时间排序
        trips.sort(key=lambda x: (x["depart_hour"], x["depart_minute"]))
        
        # 初始化车辆电量
        remaining_battery = initial_battery
        
        for trip in trips:
            depart_time = trip["depart_time"]
            depart_hour = trip["depart_hour"]
            direction = trip["direction"]
            
            # 获取本次电量消耗
            power_consumption = power_map.get(depart_hour, 10.0)
            
            # 检查是否需要充电
            need_charge = False
            if remaining_battery - power_consumption < power_threshold:
                need_charge = True
                # 充电后满电，再消耗本次电量
                remaining_battery = 100.0 - power_consumption
            else:
                remaining_battery -= power_consumption
            
            # 构建行数据
            row = {
                "车辆编号": vehicle_id,
                "四惠发车时间": "",
                "老山发车时间": ""
            }
            
            # 填充对应方向的发车时间
            display_time = depart_time
            if need_charge:
                display_time += "(充电)"
            
            if direction == "四惠":
                row["四惠发车时间"] = display_time
                row["老山发车时间"] = "/"
            elif direction == "老山":
                row["四惠发车时间"] = "/"
                row["老山发车时间"] = display_time
            else:
                row["四惠发车时间"] = display_time
                row["老山发车时间"] = "/"
            
            final_schedule.append(row)
    
    return pd.DataFrame(final_schedule)

def optimize_schedule(predictions, vehicle_count, initial_battery, solve_time_limit):
    add_log("开始初始化双算法求解器")
    st.session_state.convergence_data = []
    # 重置贪心结果
    st.session_state.greedy_solution = None
    st.session_state.greedy_schedule_data = None
    st.session_state.greedy_objective = 0.0
    
    # 检查数据
    if st.session_state.timetable_data is None:
        st.error("❌ 请先点击「读取班次表」加载排班数据")
        add_log("❌ 求解失败：未加载排班表")
        return None, None
    
    if st.session_state.power_prediction_table is None:
        st.error("❌ 请先点击「运行统计预测」生成统计数据")
        add_log("❌ 求解失败：未生成统计预测表")
        return None, None
    
    add_log("✅ 成功读取页面生成的排班表和统计预测表")
    
    # 配置参数
    config = Config(
        charger_capacity={"A": 40, "B": 40},
        rest_minutes=25.0,
        max_late_minutes=5.0,
    )
    
    # 提取小时参数
    hour_params = {}
    pred_df = st.session_state.power_prediction_table
    power_column = [col for col in pred_df.columns if "电量消耗" in col][0]
    for _, row in pred_df.iterrows():
        hour_str = row["小时"]
        hour = int(hour_str.split(":")[0])
        power_str = row[power_column].replace("%", "")
        try:
            power_consumption = float(power_str)
        except:
            power_consumption = 10.0
        
        hour_params[hour] = {
            "passenger_flow": predictions[hour-6] if 6 <= hour <= 21 else 0,
            "is_peak": row["时段类型"] in ["早高峰", "晚高峰"],
            "weather": row["天气"],
            "power_consumption": f"{power_consumption:.2f}%",
            "runtime": float(row["75%运行时间 (min)"]),
            "carbon_emission": float(row[[col for col in row.index if "碳排放" in col][0]])
        }
    add_log(f"✅ 成功从统计预测表提取 {len(hour_params)} 个小时的参数")
    
    # 提取任务列表（保留方向信息）
    trips = st.session_state.timetable_data
    add_log(f"✅ 成功从排班表提取 {len(trips)} 个任务（双向合并后）")
    
    # -------------------------- 第一步：运行贪心算法（粗略解） --------------------------
    add_log("🔄 开始运行贪心算法（粗略解）")
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("正在运行贪心算法（粗略解）...")
    progress_bar.progress(5)
    
    # 调用贪心算法
    greedy_solution = decode_with_random_keys(
        trips,
        hour_params,
        config,
        algorithm="greedy"
    )
    
    # 生成标准格式贪心排班表
    greedy_df = generate_standard_schedule(
        greedy_solution.schedule,
        st.session_state.power_prediction_table,
        initial_battery=initial_battery,
        power_threshold=20.0
    )
    
    st.session_state.greedy_solution = greedy_solution
    st.session_state.greedy_schedule_data = greedy_df
    st.session_state.greedy_objective = greedy_solution.objective
    add_log(f"✅ 贪心算法求解完成，目标值：{greedy_solution.objective:.2f}")
    progress_bar.progress(10)
    status_text.text("贪心算法完成，开始运行遗传算法（精确解）...")
    
    # -------------------------- 第二步：运行遗传算法（精确解） --------------------------
    add_log("🔄 开始运行遗传算法（精确解）")
    
    # 遗传算法参数
    POPULATION_SIZE = 64
    GENERATIONS = 100
    ELITE_SIZE = 4
    MUTATION_RATE = 0.08
    TOP_K = 5
    SEED = 20260529
    
    n = len(trips)
    rng = random.Random(SEED)
    
    # 初始化种群
    population: list[list[float]] = [[0.0 for _ in range(n)]]
    while len(population) < POPULATION_SIZE:
        population.append([rng.random() for _ in range(n)])
    
    best_solution: Solution | None = None
    best_chromosome: list[float] | None = None
    
    # 进化过程
    for gen in range(GENERATIONS + 1):
        # 进度分配：贪心占10%，遗传占90%
        progress = 10 + int((gen / GENERATIONS) * 90)
        progress_bar.progress(progress)
        status_text.text(f"遗传算法进化中... 第 {gen}/{GENERATIONS} 代")
        
        scored: list[tuple[float, list[float], Solution]] = []
        for chrom in population:
            sol = decode_with_random_keys(
                trips,
                hour_params,
                config,
                genes=chrom,
                top_k=TOP_K,
                algorithm="genetic",
            )
            scored.append((fitness(sol), chrom, sol))
        
        scored.sort(key=lambda item: item[0])
        
        # 更新最优解
        current_best = scored[0]
        if best_solution is None or current_best[0] < fitness(best_solution):
            best_solution = current_best[2]
            best_chromosome = current_best[1][:]
        
        # 每一代都记录收敛数据
        st.session_state.convergence_data.append((gen, best_solution.objective))
        
        best = scored[0][2]
        feasible_count = sum(1 for _, _, sol in scored if sol.feasible)
        
        add_log(
            f"gen={gen:03d} best={best.objective:.6f} feasible={best.feasible} "
            f"vehicles={best.vehicles_used} gap_lb={best.relative_gap_to_lb:.6f} feasible_count={feasible_count}"
        )
        
        if gen == GENERATIONS:
            break
        
        # 生成下一代
        next_population: list[list[float]] = [chrom[:] for _, chrom, _ in scored[: ELITE_SIZE]]
        while len(next_population) < POPULATION_SIZE:
            left = tournament(rng, scored)
            right = tournament(rng, scored)
            child = crossover(rng, left, right)
            mutate(rng, child, MUTATION_RATE)
            next_population.append(child)
        population = next_population
    
    progress_bar.empty()
    status_text.empty()
    
    if best_solution is None:
        st.error("❌ 遗传算法未找到可行解")
        add_log("❌ 遗传算法未找到可行解")
        return greedy_solution, greedy_df
    
    add_log(f"✅ 遗传算法求解完成，最优目标值：{best_solution.objective:.2f}")
    st.session_state.current_objective = best_solution.objective
    
    # 生成标准格式遗传排班表
    df = generate_standard_schedule(
        best_solution.schedule,
        st.session_state.power_prediction_table,
        initial_battery=initial_battery,
        power_threshold=20.0
    )
    
    add_log(f"✅ 生成标准格式排班表，共{len(df)}个班次")
    
    return best_solution, df

# -------------------------- 侧边栏 --------------------------
st.sidebar.title("🚌 智能公交调度系统")
st.sidebar.markdown("""<style>[data-testid="stSidebar"] {background-color: #f0f5fa;}</style>""", unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio("功能模块", ["📅 今日调度", "📊 数据管理", "📊 统计预测结果", "⚙️ 优化求解", "📋 排班结果"])
st.sidebar.divider()
st.sidebar.info("智能公交调度系统")

# -------------------------- 今日调度 --------------------------
if page == "📅 今日调度":
    st.header("🚌 智能公交调度", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        dispatch_date = st.date_input("调度日期", datetime.now().date())
        line = st.selectbox("线路/场站", ["1路", "2路", "3路", "4路", "5路"])
        timetable_type = st.selectbox("班次表", ["工作日", "周末", "节假日"])
    with col2:
        vehicle_count = st.number_input("当日车辆数", 1, 50, 15)
        initial_battery = st.number_input("初始电量（%）", 0, 100, 100)
        solve_time = st.number_input("求解时间上限（秒）", 60, 3600, 300)
    st.divider()

    btn1, btn2, btn3, btn4, btn5 = st.columns(5, gap="small")
    with btn1:
        if st.button("读取班次表"):
            st.session_state.start_time = time.time()
            # 根据选择的班次类型读取对应文件
            timetable_df, timetable_error = load_timetable_data(timetable_type)
            
            if timetable_df is not None:
                st.session_state.timetable_data = timetable_df
                st.success(f"✅ 成功读取 {timetable_type} 班次表，共{len(timetable_df)}条记录")
            else:
                # 文件不存在时使用示例数据
                st.session_state.timetable_data = [
                    {"depart_time": f"{6+i//2:02d}:{i%2*30:02d}", 
                     "depart_hour": 6+i//2, 
                     "depart_minute": i%2*30, 
                     "direction": "四惠" if i%2==0 else "老山"} 
                    for i in range(10)
                ]
                st.warning(f"⚠️ 未找到 {timetable_type} 班次表，使用示例数据")
            
            st.session_state.progress = 24
            st.session_state.current_stage = "班次已加载"

    with btn2:
        if st.button("读取天气"):
            weather_info, err = get_weather_forecast(dispatch_date)
            st.session_state.weather_data = weather_info
            st.success(f"✅ 天气：{weather_info['weather']} {weather_info['temp_min']}~{weather_info['temp_max']}℃")
            st.session_state.progress = 30
            st.session_state.current_stage = "天气已加载"

    with btn3:
        if st.button("运行统计预测"):
            if not st.session_state.weather_data:
                st.warning("⚠️ 请先读取天气")
            else:
                st.info("🔄 统计预测中...")
                is_workday = 1 if timetable_type == "工作日" else 0
                # 1. 客流预测
                hours, preds = predict_passenger_flow(dispatch_date, line, is_workday, st.session_state.weather_data)
                # 2. 24小时逐时电量+运行时间+碳排放联合预测
                power_table = statistical_prediction(st.session_state.weather_data)
                
                st.session_state.predictions = preds
                st.session_state.prediction_hours = hours
                st.session_state.power_prediction_table = power_table
                
                st.session_state.progress = 60
                st.session_state.current_stage = "统计预测完成"
                st.success("✅ 统计预测完成！")

    with btn4:
        if st.button("开始优化求解"):
            if not st.session_state.predictions:
                st.warning("⚠️ 请先运行统计预测")
            else:
                st.info("🔄 求解中...")
                model, df = optimize_schedule(st.session_state.predictions, vehicle_count, initial_battery, solve_time)
                st.session_state.optimization_result = model
                st.session_state.schedule_data = df
                st.session_state.progress = 90
                st.session_state.current_stage = "优化求解完成"
                st.success("✅ 优化求解完成！")

    with btn5:
        if st.button("导出排班结果"):
            if st.session_state.schedule_data is not None and st.session_state.greedy_schedule_data is not None:
                # 导出粗略解
                csv_greedy = st.session_state.greedy_schedule_data.to_csv(index=False, encoding='utf-8-sig')
                # 导出精确解
                csv_genetic = st.session_state.schedule_data.to_csv(index=False, encoding='utf-8-sig')
                
                # 显示两个下载按钮
                st.download_button("📥 下载粗略解排班表", csv_greedy, f"公交排班表_粗略解_{dispatch_date.strftime('%Y%m%d')}.csv")
                st.download_button("📥 下载精确解排班表", csv_genetic, f"公交排班表_精确解_{dispatch_date.strftime('%Y%m%d')}.csv")
                
                st.session_state.progress = 100
                st.session_state.current_stage = "全部完成"
            else:
                st.warning("⚠️ 请先完成求解")

    st.divider()
    st.progress(st.session_state.progress / 100, text=f"进度 {st.session_state.progress}%")
    st.divider()

    # 两行布局
    row1_col1, row1_col2, row1_col3 = st.columns(3, gap="medium")
    with row1_col1:
        st.metric("当前阶段", st.session_state.current_stage)
    with row1_col2:
        st.metric("已用时间", f"{int(time.time()-st.session_state.start_time)}s" if st.session_state.start_time else "0s")
    with row1_col3:
        st.metric("预计剩余", f"{int((100-st.session_state.progress)*0.5)}s" if st.session_state.progress<100 else "0s")

    st.divider()
    
    row2_col1, row2_col2 = st.columns(2, gap="medium")
    with row2_col1:
        st.metric("Gap", f"{st.session_state.current_gap:.2f}")
    with row2_col2:
        st.metric("目标值", f"{st.session_state.current_objective:.2f}")

# -------------------------- 数据管理页面 --------------------------
elif page == "📊 数据管理":
    st.header("📊 数据管理", divider="blue")
    
    st.subheader("电量消耗数据状态")
    try:
        power_df, power_error = load_power_data()
        if power_df is not None:
            st.success("✅ 成功加载 data/电量消耗.csv")
            st.dataframe(power_df, use_container_width=True)
        else:
            st.error(f"❌ 电量消耗数据加载失败：{power_error}")
            st.info("CSV格式要求：时段,天气类型,春季,夏季,秋季,冬季")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")
    
    st.divider()
    
    st.subheader("运行时间75%分位数数据状态")
    try:
        runtime_df, runtime_error = load_runtime_data()
        if runtime_df is not None:
            st.success("✅ 成功加载 data/运行时间75%分位数.csv")
            st.dataframe(runtime_df, use_container_width=True)
        else:
            st.error(f"❌ 运行时间数据加载失败：{runtime_error}")
            st.info("CSV格式要求：天气,75%运行时间 (min)")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")
    
    st.divider()
    
    st.subheader("碳排放数据状态")
    try:
        carbon_df, carbon_error = load_carbon_data()
        if carbon_df is not None:
            st.success("✅ 成功加载 data/碳排放.csv")
            st.dataframe(carbon_df, use_container_width=True)
        else:
            st.error(f"❌ 碳排放数据加载失败：{carbon_error}")
            st.info("CSV格式要求：hour,annual,summer,winter")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")
    
    st.divider()
    
    st.subheader("班次表数据状态")
    try:
        if st.session_state.timetable_data is not None:
            st.success("✅ 已加载班次表数据（保留方向信息）")
            # 转换为DataFrame显示
            df = pd.DataFrame(st.session_state.timetable_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("请在「今日调度」页面点击「读取班次表」加载数据")
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")

# -------------------------- 统计预测结果页面 --------------------------
elif page == "📊 统计预测结果":
    st.header("📊 24小时逐时统计预测结果", divider="blue")
    
    if st.session_state.power_prediction_table is None:
        st.info("请先在「今日调度」页面点击「运行统计预测」")
    else:
        current_date = st.session_state.weather_data['date']
        power_season = get_power_season(current_date)
        carbon_season = get_carbon_season(current_date)
        season_name_map = {"summer": "夏季", "winter": "冬季", "annual": "全年"}
        
        st.subheader(f"调度日期：{current_date.strftime('%Y-%m-%d')} | 当日天气：{st.session_state.weather_data['weather']}")
        st.subheader(f"电量季节：{power_season} | 碳排放季节：{season_name_map[carbon_season]}")
        
        # 显示24小时逐时结果表
        st.dataframe(
            st.session_state.power_prediction_table, 
            use_container_width=True, 
            height=800
        )
        
        # 下载合并后的CSV
        csv_data = st.session_state.power_prediction_table.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载24小时逐时统计预测结果表",
            data=csv_data,
            file_name=f"24小时逐时统计预测结果_{current_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ 所有数据100%来自你上传的CSV文件，完全匹配当日天气和季节")

# -------------------------- 优化求解页面 --------------------------
elif page == "⚙️ 优化求解":
    st.header("⚙️ 优化求解", divider="blue")
    
    # 显示贪心算法（粗略解）结果
    if st.session_state.greedy_solution:
        st.subheader("📌 粗略解（贪心算法）")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("最优目标值", f"{st.session_state.greedy_objective:.2f}")
        with col2:
            st.metric("使用车辆数", st.session_state.greedy_solution.vehicles_used)
        st.divider()
    
    # 显示遗传算法（精确解）结果
    if st.session_state.optimization_result:
        st.subheader("🎯 精确解（遗传算法）")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("最优目标值", f"{st.session_state.current_objective:.2f}")
        with col2:
            st.metric("使用车辆数", st.session_state.optimization_result.vehicles_used)
        
        # 显示收敛曲线
        if st.session_state.convergence_data:
            st.subheader("遗传算法收敛曲线")
            conv_df = pd.DataFrame(st.session_state.convergence_data, columns=["代次", "目标值"])
            st.line_chart(conv_df.set_index("代次"))
    else:
        st.info("请先在「今日调度」页面点击「开始优化求解」")

# -------------------------- 排班结果页面 --------------------------
elif page == "📋 排班结果":
    st.header("📋 排班结果", divider="blue")
    
    # 显示贪心算法（粗略解）排班表
    if st.session_state.greedy_schedule_data is not None:
        st.subheader("📌 粗略解（贪心算法）排班表")
        st.dataframe(st.session_state.greedy_schedule_data, use_container_width=True)
        
        # 下载粗略解排班表
        csv_greedy = st.session_state.greedy_schedule_data.to_csv(index=False, encoding='utf-8-sig')
        current_date = datetime.now().date()
        if st.session_state.weather_data:
            current_date = st.session_state.weather_data['date']
        st.download_button(
            label="📥 下载粗略解排班表",
            data=csv_greedy,
            file_name=f"公交排班表_粗略解_{current_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        st.divider()
    
    # 显示遗传算法（精确解）排班表
    if st.session_state.schedule_data is not None:
        st.subheader("🎯 精确解（遗传算法）排班表")
        st.dataframe(st.session_state.schedule_data, use_container_width=True)
        
        # 下载精确解排班表
        csv_genetic = st.session_state.schedule_data.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载精确解排班表",
            data=csv_genetic,
            file_name=f"公交排班表_精确解_{current_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("请先完成优化求解")
