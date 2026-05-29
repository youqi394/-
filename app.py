# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import random
from datetime import datetime, timedelta
import re

from heuristic_common import (
    Config,
    Solution,
    decode_with_random_keys,
)

# ==================== 页面全局样式（原样保留） ====================
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
.stButton>button {height: 50px; font-size: 16px; width: 100%; border-radius: 12px; border: none; background-color: #1f77b4; color: white;}
.stButton>button:hover {background-color: #155a8a; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);}
.stMetric {background-color: #f8f9fa; padding: 12px; border-radius: 10px; border-left: 4px solid #1f77b4;}
h1,h2,h3 {color: #2c3e50; font-weight: 600;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==================== 会话状态初始化（原样保留） ====================
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
if 'greedy_solution' not in st.session_state:
    st.session_state.greedy_solution = None
if 'greedy_schedule_data' not in st.session_state:
    st.session_state.greedy_schedule_data = None
if 'greedy_objective' not in st.session_state:
    st.session_state.greedy_objective = 0.0

# ==================== 工具函数（全部原样保留） ====================
def add_log(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.solve_log.append(f"[INFO] {ts} - {message}")

def normalize_column_name(name):
    return re.sub(r'[\s()%]', '', str(name)).lower()

def get_carbon_season(date):
    m = date.month
    if 6 <= m <= 8:
        return "summer"
    elif m in (12,1,2):
        return "winter"
    return "annual"

def get_power_season(date):
    m = date.month
    if 3 <= m <=5:
        return "春季"
    elif 6 <= m <=8:
        return "夏季"
    elif 9 <= m <=11:
        return "秋季"
    return "冬季"

def get_time_period(hour):
    if 7 <= hour <=9:
        return "早高峰"
    elif 17 <= hour <=19:
        return "晚高峰"
    elif hour ==6 or 10<=hour<=16 or 20<=hour<=21:
        return "平峰"
    return "低峰"

# ==================== 天气获取（原样保留） ====================
def get_weather_forecast(date):
    WEATHER_API_KEY = "e088a35c897818780a479973d4623063"
    today = datetime.now().date()
    max_forecast = today + timedelta(days=3)
    in_range = (today <= date <= max_forecast)
    if in_range:
        try:
            url = f"https://restapi.amap.com/v3/weather/weatherInfo?city=110000&key={WEATHER_API_KEY}&extensions=all"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data.get("status") == "1":
                target = date.strftime("%Y-%m-%d")
                for day in data["forecasts"][0]["casts"]:
                    if day["date"] == target:
                        w = {
                            "date": date,
                            "temp_max": int(day["daytemp"]),
                            "temp_min": int(day["nighttemp"]),
                            "weather": day["dayweather"].strip(),
                            "is_rain": 1 if "雨" in day["dayweather"] else 0
                        }
                        st.session_state.weather_source = f"✅ 高德API {target}"
                        add_log(f"天气获取成功：{w['weather']} {w['temp_min']}~{w['temp_max']}℃")
                        return w, None
        except Exception as e:
            add_log(f"天气请求异常：{e}")
    st.session_state.weather_source = "ℹ️ 使用默认天气"
    add_log("使用默认天气数据")
    return {"date":date,"temp_max":25,"temp_min":18,"weather":"晴","is_rain":0}, None

# ==================== 加载时刻表（双向识别 原样保留） ====================
@st.cache_resource
def load_timetable_data(timetable_type):
    file_map = {"工作日":"工作日发车时刻表.csv","周末":"节假日发车时刻表.csv","节假日":"节假日发车时刻表.csv"}
    fn = file_map.get(timetable_type, "工作日发车时刻表.csv")
    path = f"data/{fn}"
    add_log(f"读取时刻表：{path}")
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(path, dtype=str, encoding="gbk")
        except:
            add_log("未找到时刻表文件，启用示例数据")
            sample = []
            for i in range(12):
                h = 6 + i//2
                m = (i%2)*30
                d = "四惠" if i%2==0 else "老山"
                sample.append({"depart_time":f"{h:02d}:{m:02d}","depart_hour":h,"depart_minute":m,"direction":d})
            return sample, "文件不存在，使用示例"
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    trips = []
    for col in df.columns:
        ncol = normalize_column_name(col)
        if "发车时刻" in ncol or "发车时间" in ncol:
            if "四惠" in col and "老山" in col:
                if col.index("四惠") < col.index("老山"):
                    dire = "四惠"
                else:
                    dire = "老山"
            elif "四惠" in col:
                dire = "四惠"
            elif "老山" in col:
                dire = "老山"
            else:
                dire = "四惠"
            times = df[col].dropna().tolist()
            for tstr in times:
                parts = tstr.split(":")
                if len(parts)!=2:
                    continue
                try:
                    h = int(parts[0])
                    m = int(parts[1])
                    trips.append({"depart_time":tstr,"depart_hour":h,"depart_minute":m,"direction":dire})
                except:
                    continue
    add_log(f"解析完成，共{len(trips)}条班次")
    return trips, None

# ==================== 加载基础CSV数据（原样保留） ====================
@st.cache_resource
def load_carbon_data():
    try:
        df = pd.read_csv("data/碳排放.csv", encoding="utf-8")
    except:
        df = pd.read_csv("data/碳排放.csv", encoding="gbk")
    df.columns = df.columns.str.strip()
    for c in ["hour","annual","summer","winter"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, None

@st.cache_resource
def load_runtime_data():
    try:
        df = pd.read_csv("data/运行时间75%分位数.csv", encoding="utf-8")
    except:
        df = pd.read_csv("data/运行时间75%分位数.csv", encoding="gbk")
    df.columns = df.columns.str.strip()
    return df, None

@st.cache_resource
def load_power_data():
    try:
        df = pd.read_csv("data/电量消耗.csv", encoding="utf-8")
    except:
        df = pd.read_csv("data/电量消耗.csv", encoding="gbk")
    df.columns = df.columns.str.strip()
    return df, None

# ==============================================
# 【重点】统计预测函数 → 完全原样保留，不做任何修改！
# ==============================================
def statistical_prediction(weather_info):
    w_now = weather_info["weather"]
    date = weather_info["date"]
    p_season = get_power_season(date)
    c_season = get_carbon_season(date)
    c_map = {"summer":"夏季","winter":"冬季","annual":"全年"}

    power_df, _ = load_power_data()
    runtime_df, _ = load_runtime_data()
    carbon_df, _ = load_carbon_data()

    # 匹配当前天气对应的运行时间（全局统一）
    run_val = 45.0
    if runtime_df is not None:
        rt = runtime_df[runtime_df["天气"] == w_now]
        if not rt.empty:
            run_val = float(rt.iloc[0]["75%运行时间 (min)"])

    # 匹配当前天气的电量表
    power_match = None
    if power_df is not None:
        power_match = power_df[power_df["天气类型"] == w_now]

    res = []
    p_col = f"{p_season}电量消耗"
    c_col = f"{c_map[c_season]}碳排放"

    for hour in range(24):
        period = get_time_period(hour)
        # 取当前时段+天气的电量
        pct = "10.00%"
        if power_match is not None and not power_match.empty:
            row = power_match[power_match["时段"] == period]
            if not row.empty:
                pct = f"{float(row.iloc[0][p_season]):.2f}%"
        # 取当前小时碳排放
        carbon = 0.0
        if carbon_df is not None:
            cr = carbon_df[carbon_df["hour"] == hour]
            if not cr.empty:
                carbon = float(cr.iloc[0][c_season])

        res.append({
            "小时": f"{hour:02d}:00",
            "时段类型": period,
            "天气": w_now,
            p_col: pct,
            "75%运行时间 (min)": f"{run_val:.2f}",
            c_col: f"{carbon:.4f}"
        })
    return pd.DataFrame(res)

# 客流预测 → 原样保留
def predict_passenger_flow(date, line_id, is_workday, weather_data):
    hours = list(range(6,22))
    base = 150 if is_workday else 100
    rain = 0.8 if (weather_data and weather_data["is_rain"]) else 1.0
    out = []
    for h in hours:
        if 7<=h<=9:
            f = base * 2.5 * rain
        elif 17<=h<=19:
            f = base * 2.2 * rain
        else:
            f = base * rain
        out.append(round(f * (0.9 + np.random.rand()*0.2)))
    return hours, out

# ==============================================
# 【新增函数】仅用于优化求解：从「已生成的预测表」解析参数
# ==============================================
def build_hour_params_from_pred_table(pred_df):
    """
    从页面已生成的24小时统计预测表提取：运行时间、电量、高峰时段
    仅在【优化求解】环节调用，不触碰统计预测原有逻辑
    """
    hour_params = {}
    run_col = "75%运行时间 (min)"
    pwr_col = None

    # 自动匹配电量列
    for c in pred_df.columns:
        if "电量消耗" in c:
            pwr_col = c
            break
    if pwr_col is None:
        pwr_col = "春季电量消耗"

    for _, row in pred_df.iterrows():
        h_str = row["小时"]
        h = int(h_str.split(":")[0])
        period = row["时段类型"]

        # 读取预测表内运行时间
        try:
            run_t = float(row[run_col])
        except:
            run_t = 45.0
        # 读取预测表内电量
        try:
            pct_str = row[pwr_col]
        except:
            pct_str = "10.00%"
        # 判断是否高峰
        peak = period in ("早高峰", "晚高峰")
        pax_flow = 150 if peak else 100

        hour_params[h] = {
            "runtime": run_t,
            "power_consumption": pct_str,
            "is_peak": peak,
            "passenger_flow": pax_flow
        }
    add_log("✅ 优化求解：读取页面统计预测表的运行时间、电量参数")
    return hour_params

# ==============================================
# 【修改】排班表生成：基于预测表计算电量/到达时间/充电标记
# ==============================================
def generate_standard_schedule(raw_schedule, pred_df, initial_battery=100.0, power_threshold=20.0):
    if not raw_schedule:
        return pd.DataFrame()
    # 车辆分组
    groups = {}
    for item in raw_schedule:
        vid = item["vehicle_id"]
        groups.setdefault(vid, []).append(item)
    # 车辆排序
    vid_list = sorted(groups.keys(), key=lambda x: int(x.replace("车","")))

    # 构建小时→电量数值映射
    pwr_map = {}
    pwr_col = None
    for c in pred_df.columns:
        if "电量消耗" in c:
            pwr_col = c
            break
    for _, row in pred_df.iterrows():
        h = int(row["小时"].split(":")[0])
        val = float(row[pwr_col].replace("%",""))
        pwr_map[h] = val

    out = []
    for vid in vid_list:
        trips = groups[vid]
        trips.sort(key=lambda x: (x["depart_hour"], x["depart_minute"]))
        bat = initial_battery
        station = "四惠"

        for t in trips:
            dtime = t["depart_time"]
            dh = t["depart_hour"]
            dire = t["direction"]
            arrive_t = t["arrive_time"]
            run_t = t["runtime"]
            pct_use = pwr_map.get(dh, 10.0)

            need_charge = False
            if bat - pct_use < power_threshold:
                need_charge = True
                bat = 100.0 - pct_use
            else:
                bat -= pct_use

            disp = dtime
            if need_charge:
                disp += "(充电)"

            row = {"车辆编号":vid, "四惠发车时间":"", "老山发车时间":"", "到达时间":arrive_t, "单趟运行时长(min)":f"{run_t:.1f}"}
            if dire == "四惠":
                row["四惠发车时间"] = disp
                row["老山发车时间"] = "/"
                station = "老山"
            else:
                row["老山发车时间"] = disp
                row["四惠发车时间"] = "/"
                station = "四惠"
            out.append(row)
    return pd.DataFrame(out)

# ==================== 遗传算法算子（原样保留） ====================
def tournament(rng, scored, size=3):
    picks = [rng.choice(scored) for _ in range(size)]
    picks.sort(key=lambda x:x[0])
    return picks[0][1]

def crossover(rng, left, right):
    if len(left)!=len(right) or len(left)<=2:
        return left[:]
    a = rng.randrange(1, len(left)-1)
    b = rng.randrange(a, len(left))
    child = left[:a] + right[a:b] + left[b:]
    if rng.random() < 0.25:
        child = [right[i] if rng.random()<0.5 else child[i] for i in range(len(child))]
    return child

def mutate(rng, chrom, rate):
    for i in range(len(chrom)):
        if rng.random() < rate:
            chrom[i] = rng.random()

def fitness(sol):
    return sol.objective

# ==============================================
# 【修改】优化主函数：读取页面预测表，不再读取原始CSV
# ==============================================
def optimize_schedule(pred_flow, veh_cnt, init_bat, solve_limit):
    add_log("开始优化调度")
    st.session_state.convergence_data = []
    st.session_state.greedy_solution = None
    st.session_state.greedy_schedule_data = None

    # 前置校验
    if st.session_state.timetable_data is None:
        st.error("请先读取班次表")
        return None, None
    if st.session_state.power_prediction_table is None:
        st.error("请先运行【统计预测】生成运行时间、电量表")
        return None, None

    # ========== 核心：读取【页面已生成的统计预测表】解析参数 ==========
    hour_params = build_hour_params_from_pred_table(st.session_state.power_prediction_table)
    trips = st.session_state.timetable_data

    # 调度配置（原样保留）
    cfg = Config(
        charger_capacity={"A":40,"B":40},
        rest_minutes=25.0,
        max_late_minutes=5.0
    )

    # -------- 第一步：贪心算法 --------
    add_log("执行贪心算法")
    prog = st.progress(0)
    stat = st.empty()
    stat.text("贪心算法运行中...")
    prog.progress(5)

    greedy_sol = decode_with_random_keys(trips, hour_params, cfg, algorithm="greedy")
    greedy_df = generate_standard_schedule(
        greedy_sol.schedule,
        st.session_state.power_prediction_table,
        initial_battery=init_bat,
        power_threshold=20.0
    )
    st.session_state.greedy_solution = greedy_sol
    st.session_state.greedy_schedule_data = greedy_df
    st.session_state.greedy_objective = greedy_sol.objective
    add_log(f"贪心完成，目标值：{greedy_sol.objective:.2f}")
    prog.progress(10)
    stat.text("贪心完成，启动遗传算法...")

    # -------- 第二步：遗传算法 --------
    POP = 64
    GEN = 100
    ELITE = 4
    MUT_RATE = 0.08
    SEED = 20260529
    n_trip = len(trips)
    rng = random.Random(SEED)

    pop = [[0.0]*n_trip]
    while len(pop) < POP:
        pop.append([rng.random() for _ in range(n_trip)])

    best_sol = None
    best_chrom = None

    for g in range(GEN+1):
        pct = 10 + int((g/GEN)*90)
        prog.progress(pct)
        stat.text(f"遗传进化 第{g}/{GEN}代")

        scored = []
        for chrom in pop:
            s = decode_with_random_keys(trips, hour_params, cfg, genes=chrom, algorithm="genetic")
            scored.append((fitness(s), chrom, s))
        scored.sort(key=lambda x:x[0])

        curr_best = scored[0]
        if best_sol is None or curr_best[0] < fitness(best_sol):
            best_sol = curr_best[2]
            best_chrom = curr_best[1][:]

        st.session_state.convergence_data.append((g, best_sol.objective))
        add_log(f"第{g}代 最优目标：{best_sol.objective:.4f} 用车数：{best_sol.vehicles_used}")

        if g == GEN:
            break

        next_pop = [x[1][:] for x in scored[:ELITE]]
        while len(next_pop) < POP:
            l = tournament(rng, scored)
            r = tournament(rng, scored)
            c = crossover(rng, l, r)
            mutate(rng, c, MUT_RATE)
            next_pop.append(c)
        pop = next_pop

    prog.empty()
    stat.empty()

    if best_sol is None:
        st.warning("遗传未找到可行解，返回贪心结果")
        return greedy_sol, greedy_df

    st.session_state.current_objective = best_sol.objective
    final_df = generate_standard_schedule(
        best_sol.schedule,
        st.session_state.power_prediction_table,
        initial_battery=init_bat,
        power_threshold=20.0
    )
    add_log("遗传算法求解完毕")
    return best_sol, final_df

# ==================== 页面布局、按钮逻辑（全部原样保留） ====================
st.sidebar.title("🚌 智能公交调度系统")
st.sidebar.divider()
page = st.sidebar.radio("功能模块", ["📅 今日调度", "📊 数据管理", "📊 统计预测结果", "⚙️ 优化求解", "📋 排班结果"])

if page == "📅 今日调度":
    st.header("🚌 智能公交调度", divider="blue")
    c1,c2 = st.columns(2)
    with c1:
        disp_date = st.date_input("调度日期", datetime.now().date())
        line = st.selectbox("线路", ["1路","2路","3路"])
        tt_type = st.selectbox("班次表", ["工作日","周末","节假日"])
    with c2:
        veh_num = st.number_input("当日可用车辆数", 1, 50, 15)
        init_bat = st.number_input("车辆初始电量(%)", 0, 100, 100)
        solve_t = st.number_input("求解时长上限(秒)", 60, 3600, 300)
    st.divider()

    b1,b2,b3,b4,b5 = st.columns(5)
    with b1:
        if st.button("读取班次表"):
            st.session_state.start_time = time.time()
            tt_data, err = load_timetable_data(tt_type)
            st.session_state.timetable_data = tt_data
            st.session_state.progress = 24
            st.session_state.current_stage = "班次表加载完成"
            st.success(f"加载班次共 {len(tt_data)} 条")
    with b2:
        if st.button("读取天气"):
            wd, _ = get_weather_forecast(disp_date)
            st.session_state.weather_data = wd
            st.success(f"天气：{wd['weather']} {wd['temp_min']}~{wd['temp_max']}℃")
            st.session_state.progress = 30
            st.session_state.current_stage = "天气加载完成"
    with b3:
        if st.button("运行统计预测"):
            if not st.session_state.weather_data:
                st.warning("请先读取天气")
            else:
                is_work = 1 if tt_type=="工作日" else 0
                hrs, flow = predict_passenger_flow(disp_date, line, is_work, st.session_state.weather_data)
                pred_table = statistical_prediction(st.session_state.weather_data)
                st.session_state.predictions = flow
                st.session_state.prediction_hours = hrs
                st.session_state.power_prediction_table = pred_table
                st.session_state.progress = 60
                st.session_state.current_stage = "统计预测完成（运行时间+电量已生成）"
                st.success("✅ 24小时运行时间、电量消耗预测表生成完毕")
    with b4:
        if st.button("开始优化求解"):
            if not st.session_state.power_prediction_table:
                st.warning("请先运行【统计预测】")
            else:
                res, df = optimize_schedule(st.session_state.predictions, veh_num, init_bat, solve_t)
                st.session_state.optimization_result = res
                st.session_state.schedule_data = df
                st.session_state.progress = 90
                st.session_state.current_stage = "优化求解完成"
                st.success("✅ 调度求解完成，时间、电量全部匹配预测表")
    with b5:
        if st.button("导出排班结果"):
            if st.session_state.schedule_data is not None:
                d = disp_date.strftime("%Y%m%d")
                csv1 = st.session_state.greedy_schedule_data.to_csv(index=False, encoding="utf-8-sig")
                csv2 = st.session_state.schedule_data.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("下载贪心排班表", csv1, f"排班_贪心_{d}.csv")
                st.download_button("下载遗传排班表", csv2, f"排班_遗传_{d}.csv")
                st.session_state.progress = 100
                st.session_state.current_stage = "全部完成"
            else:
                st.warning("请先完成求解")

    st.divider()
    st.progress(st.session_state.progress/100, text=f"整体进度 {st.session_state.progress}%")
    st.divider()
    r1c1,r1c2,r1c3 = st.columns(3)
    r1c1.metric("当前阶段", st.session_state.current_stage)
    r1c2.metric("已用时", f"{int(time.time()-(st.session_state.start_time or time.time()))}s")
    r1c3.metric("目标函数值", f"{st.session_state.current_objective:.2f}")

elif page == "📊 数据管理":
    st.header("📊 数据管理", divider="blue")
    st.subheader("电量消耗表")
    df,_ = load_power_data()
    st.dataframe(df, use_container_width=True)
    st.subheader("运行时间75%分位数表")
    df,_ = load_runtime_data()
    st.dataframe(df, use_container_width=True)
    st.subheader("碳排放表")
    df,_ = load_carbon_data()
    st.dataframe(df, use_container_width=True)

elif page == "📊 统计预测结果":
    st.header("📊 24小时逐时预测结果（运行时间+电量）", divider="blue")
    if st.session_state.power_prediction_table is None:
        st.info("请先在【今日调度】执行「运行统计预测」")
    else:
        st.dataframe(st.session_state.power_prediction_table, use_container_width=True)
        csv = st.session_state.power_prediction_table.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下载预测表", csv, "24小时运行时间_电量预测表.csv")

elif page == "⚙️ 优化求解":
    st.header("⚙️ 求解结果", divider="blue")
    if st.session_state.greedy_solution:
        st.subheader("贪心算法（粗略解）")
        st.metric("目标值", f"{st.session_state.greedy_objective:.2f}")
        st.metric("使用车辆数", st.session_state.greedy_solution.vehicles_used)
    if st.session_state.optimization_result:
        st.subheader("遗传算法（最优解）")
        st.metric("目标值", f"{st.session_state.current_objective:.2f}")
        st.metric("使用车辆数", st.session_state.optimization_result.vehicles_used)
        if st.session_state.convergence_data:
            cd = pd.DataFrame(st.session_state.convergence_data, columns=["迭代代次","目标值"])
            st.line_chart(cd.set_index("迭代代次"))

elif page == "📋 排班结果":
    st.header("📋 最终排班表（含真实运行时长、到达时间、充电标记）", divider="blue")
    if st.session_state.greedy_schedule_data is not None:
        st.subheader("贪心算法排班")
        st.dataframe(st.session_state.greedy_schedule_data, use_container_width=True)
    if st.session_state.schedule_data is not None:
        st.subheader("遗传算法排班")
        st.dataframe(st.session_state.schedule_data, use_container_width=True)
