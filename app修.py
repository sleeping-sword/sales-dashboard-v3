import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import os

# ==========================================
# 0. 基礎設定 & 字體處理
# ==========================================
st.set_page_config(page_title="消費趨勢智慧分析平台", layout="wide")

font_path = "NotoSansTC-Regular.otf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['Noto Sans TC']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
    
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 模型定義
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out

# ==========================================
# 2. 運算核心
# ==========================================

# --- A. 執行線性回歸 (Linear Regression) ---
def run_linear_regression(df, target_col):
    """回傳: 預測值, 趨勢線數據"""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[target_col].values
    model = LinearRegression()
    model.fit(X, y)
    
    # 預測下一期
    next_index = np.array([[len(df)]])
    lr_pred = model.predict(next_index)[0]
    trend_line = model.predict(X)
    
    return lr_pred, trend_line

# --- B. 執行 LSTM (深度學習) ---
def run_lstm_prediction(df, feature_cols, target_col, seq_length=5, epochs=100, lr=0.01):
    """回傳: 預測值, Loss, 下限, 上限"""
    try:
        data_X = df[feature_cols].select_dtypes(include=[np.number]).values.astype('float32')
        data_y = df[[target_col]].select_dtypes(include=[np.number]).values.astype('float32')
    except Exception as e:
        return None, f"資料轉換錯誤: {e}", None, None

    if len(df) <= seq_length:
        return None, f"資料筆數不足", None, None

    # 標準化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(data_X)
    y_scaled = scaler_y.fit_transform(data_y)

    # 建立資料集
    X_train, y_train = [], []
    for i in range(len(X_scaled) - seq_length):
        X_train.append(X_scaled[i : i + seq_length])
        y_train.append(y_scaled[i + seq_length])

    X_train = torch.FloatTensor(np.array(X_train))
    y_train = torch.FloatTensor(np.array(y_train))

    # 建立模型
    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 訓練
    model.train()
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            progress_bar.progress((epoch + 1) / epochs)
    progress_bar.empty()

    # 預測
    model.eval()
    last_sequence = X_scaled[-seq_length:] 
    last_sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)

    with torch.no_grad():
        predicted_scaled = model(last_sequence_tensor).numpy()

    lstm_pred = scaler_y.inverse_transform(predicted_scaled)[0][0]
    pred_low = lstm_pred * 0.98
    pred_high = lstm_pred * 1.02
    
    return lstm_pred, loss.item(), pred_low, pred_high

# ==========================================
# 3. Streamlit 介面
# ==========================================
st.title("📊 消費趨勢智慧分析平台")

page = st.sidebar.selectbox(
    "功能選擇",
    ["綜合預測分析 (迴歸 + LSTM)", "分析市場趨勢", "試算獲利潛力組合"]
)

if page == "綜合預測分析 (迴歸 + LSTM)":
    st.subheader("📈 綜合預測分析")
    st.markdown("依序執行 **線性回歸 (長期趨勢)** 與 **LSTM (深度學習)** 分析。")

    uploaded_file = st.file_uploader("📤 上傳銷售資料 CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # --- 日期與資料前處理 ---
        date_cols = [col for col in df.columns if col.lower() in ['date', '月份', '日期', 'time']]
        if date_cols:
            date_col = date_cols[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                x_axis = df[date_col]
            except:
                x_axis = range(len(df))
        else:
            x_axis = range(len(df))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ 無數值欄位")
        else:
            # --- 欄位與參數設定 ---
            c1, c2 = st.columns(2)
            with c1:
                target_col = st.selectbox("🎯 目標欄位", numeric_cols, index=0)
            with c2:
                feature_cols = st.multiselect("⚙️ LSTM 輔助特徵", numeric_cols, default=[target_col])

            with st.expander("🔧 進階參數設定"):
                seq_len = st.slider("LSTM 參考期數", 2, 24, 5)
                epochs = st.slider("LSTM 訓練次數", 50, 500, 150)
                lr = st.number_input("學習率", value=0.01)

            # --- 執行按鈕 ---
            if st.button('🚀 開始雙模型分析'):
                if not feature_cols:
                    st.error("請選擇特徵欄位！")
                else:
                    # 計算 X 軸預測點的位置
                    if isinstance(x_axis, pd.Series) and pd.api.types.is_datetime64_any_dtype(x_axis):
                        last_date = x_axis.iloc[-1]
                        next_date = last_date + (last_date - x_axis.iloc[-2])
                        ax_x = next_date
                    else:
                        ax_x = len(df)

                    # ==========================================
                    # 第一階段：線性回歸分析
                    # ==========================================
                    st.markdown("---")
                    st.subheader("1️⃣ 線性回歸分析 (Linear Regression)")
                    st.caption("用途：觀察整體的成長或衰退趨勢，忽略短期波動。")
                    
                    lr_pred, trend_line = run_linear_regression(df, target_col)
                    
                    # 計算區間
                    lr_low = lr_pred * 0.98
                    lr_high = lr_pred * 1.02

                    # 【修改重點】直接顯示一個區間字串
                    st.metric("長期趨勢預測區間 (±2%)", f"{int(lr_low):,} ~ {int(lr_high):,}")
                    
                    # 畫圖 1
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(x_axis, df[target_col], label='歷史數據', color='#1f77b4', linewidth=1)
                    ax1.plot(x_axis, trend_line, label='趨勢線', color='orange', linestyle='--', linewidth=1.5)
                    
                    ax1.scatter([ax_x], [lr_pred], color='orange', s=80, marker='s', label='趨勢預測點')
                    
                    # 畫出區間線
                    ax1.vlines(x=ax_x, ymin=lr_low, ymax=lr_high, color='orange', linestyle=':', linewidth=2, label='趨勢區間')
                    ax1.hlines(y=[lr_low, lr_high], xmin=ax_x, xmax=ax_x, color='orange', linewidth=4)
                    
                    ax1.set_title(f"{target_col} - 長期趨勢分析")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1)

                    # ==========================================
                    # 第二階段：LSTM 深度學習
                    # ==========================================
                    st.markdown("---")
                    st.subheader("2️⃣ LSTM 深度學習預測")
                    st.caption(f"用途：AI 學習過去 {seq_len} 期的波動模式，預測下一期精確數值。")

                    with st.spinner('AI 正在進行深度運算...'):
                        lstm_pred, loss, low, high = run_lstm_prediction(
                            df, feature_cols, target_col, 
                            seq_length=seq_len, epochs=epochs, lr=lr
                        )

                    if lstm_pred is not None:
                        # 這裡也可以改成顯示區間，或者保留 3 欄
                        col_lstm_1, col_lstm_2, col_lstm_3 = st.columns(3)
                        col_lstm_1.metric("AI 精確預測值", f"{int(lstm_pred):,}")
                        col_lstm_2.metric("預測下限 (-2%)", f"{int(low):,}")
                        col_lstm_3.metric("預測上限 (+2%)", f"{int(high):,}")

                        # 畫圖 2
                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                        ax2.plot(x_axis, df[target_col], label='歷史數據', color='#1f77b4', linewidth=1, marker='o', markersize=3)
                        
                        ax2.scatter([ax_x], [lstm_pred], color='red', s=100, label='AI 預測點', zorder=5)
                        
                        ax2.vlines(x=ax_x, ymin=low, ymax=high, color='red', linestyle=':', linewidth=2, label='信心區間')
                        ax2.hlines(y=[low, high], xmin=ax_x, xmax=ax_x, color='red', linewidth=4)

                        ax2.set_title(f"{target_col} - AI 短期波動預測")
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        st.pyplot(fig2)

                        # --- 綜合結論區 ---
                        st.markdown("### 📊 綜合分析結論")
                        diff = lstm_pred - lr_pred
                        trend_text = "高於" if diff > 0 else "低於"
                        
                        st.info(f"""
                        * **長期來看**：根據回歸分析，市場趨勢預測範圍約在 **{int(lr_low):,} ~ {int(lr_high):,}**。
                        * **短期來看**：考慮近期波動後，AI 認為下一期數值會落在 **{int(low):,} ~ {int(high):,}**。
                        * **結論**：AI 的預測結果 **{trend_text}** 長期趨勢線，建議決策者多加留意近期的市場變化因子。
                        """)

                    else:
                        st.error(loss)

elif page == "分析市場趨勢":
    st.subheader("📊 分析市場趨勢")
    regions = ['北部', '中部', '南部', '東部']
    spending = [50, 40, 70, 30]
    fig, ax = plt.subplots()
    ax.bar(regions, spending, color=['#007bff','#17a2b8','#28a745','#ffc107'])
    st.pyplot(fig)

else:
    st.subheader("💡 試算獲利潛力組合")
    price = st.slider("產品價格", 50, 500, 200)
    discount = st.slider("折扣", 0, 50, 10)
    demand = max(0, 1000 - (price - 200) * 2 + discount * 5)
    profit = demand * (price * (1 - discount / 100) * 0.3)
    st.metric("預估獲利", f"{profit:,.0f}")