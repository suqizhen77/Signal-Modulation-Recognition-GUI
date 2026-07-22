# ==========================================
# 导入系统与基础数学处理库
# ==========================================
import sys
import time
import numpy as np  # 核心数学库，用来做矩阵运算和生成随机数
import matplotlib.pyplot as plt  # 核心绘图库，用来画波形和频谱

# ==========================================
# 导入 PyQt5 图形界面库
# ==========================================
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton, QFrame,
                             QTabWidget, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ==========================================
# 导入信号处理与 AI 机器学习库
# ==========================================
from scipy.signal import welch, find_peaks
from scipy.stats import kurtosis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 第一部分：底层苦力类 —— 专门负责生成物理信号、加噪声、提取特征
# ==============================================================================
class SignalProcessor:
    def __init__(self):
        # 物理环境基本参数设定
        self.fs = 100000  # 采样率：100 kHz（每秒采集10万个点）
        self.N = 4096  # 截获点数：每次截取 4096 个点送给 AI
        self.t = np.arange(self.N) / self.fs  # 生成对应这 41 毫秒的时间轴坐标
        self.fc = 10000  # 载波频率：10 kHz
        self.sym_len = 32  # 码元长度：每个信息比特占用 32 个采样点
        self.last_symbols = None  # 【新增】用来记录刚生成的真实基带数据，供画图对齐用

    def generate_signal(self, mod_type, snr_db):
        """ 根据信号类型和环境信噪比，生成真实的带噪物理波形 """
        # 计算这 4096 个点里总共包含了多少个码元（4096 / 32 = 128 个码元）
        num_symbols = self.N // self.sym_len

        # 1. 纯净信号生成区（数学模拟）
        if mod_type == 'ASK':
            # 【关键修改】：把 1到4 改成了 0和1，生成标准的二进制 OOK 信号
            symbols = np.random.randint(0, 2, num_symbols)
            self.last_symbols = symbols  # 把真实的0101记录下来
            baseband = np.repeat(symbols, self.sym_len)[:self.N]
            sig = baseband * np.cos(2 * np.pi * self.fc * self.t)

        elif mod_type == 'PSK':
            symbols = np.random.randint(0, 4, num_symbols)
            self.last_symbols = symbols % 2  # 画图仅作示意
            phases = symbols * (np.pi / 2)
            baseband_phase = np.repeat(phases, self.sym_len)[:self.N]
            sig = np.cos(2 * np.pi * self.fc * self.t + baseband_phase)

        elif mod_type == 'FSK':
            symbols = np.random.randint(0, 4, num_symbols)
            self.last_symbols = symbols % 2  # 画图仅作示意
            freqs = self.fc + symbols * 2000
            baseband_freq = np.repeat(freqs, self.sym_len)[:self.N]
            phase = np.cumsum(2 * np.pi * baseband_freq / self.fs)
            sig = np.cos(phase)

        elif mod_type == '16QAM':
            levels = np.array([-3, -1, 1, 3])
            I_sym = np.random.choice(levels, num_symbols)
            Q_sym = np.random.choice(levels, num_symbols)
            self.last_symbols = np.where(I_sym > 0, 1, 0)  # 画图仅作示意
            I = np.repeat(I_sym, self.sym_len)[:self.N]
            Q = np.repeat(Q_sym, self.sym_len)[:self.N]
            sig = I * np.cos(2 * np.pi * self.fc * self.t) - Q * np.sin(2 * np.pi * self.fc * self.t)

        # 把生成的纯净信号功率归一化
        current_power = np.mean(sig ** 2)
        if current_power > 0:
            sig = sig / np.sqrt(current_power)

        # 2. 环境噪声叠加区（物理模拟）
        signal_power = 1.0
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), self.N)

        rx_sig = sig + noise
        rx_sig = rx_sig / np.std(rx_sig)

        return rx_sig

    def extract_features(self, sig):
        """ 对输入的那一段 41毫秒带噪信号，提取 7 维身份特征 """
        env = np.abs(sig)
        env_var = np.var(env)
        papr = np.max(sig ** 2) / np.mean(sig ** 2)
        kurt = kurtosis(env)

        hist, _ = np.histogram(env, bins=30, density=True)
        hist = hist[hist > 0]
        amp_entropy = -np.sum(hist * np.log2(hist))

        f, Pxx = welch(sig, self.fs, nperseg=512)
        max_psd = np.max(Pxx)
        peak_freq = f[np.argmax(Pxx)]
        peaks, _ = find_peaks(Pxx, height=max_psd * 0.3)
        peak_count = len(peaks)

        return np.array([env_var, papr, kurt, amp_entropy, max_psd, peak_freq, peak_count])


# ==============================================================================
# 第二部分：界面与调度类 —— 负责展示图表、与人交互、指挥 AI 训练和预测
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工程级通信信号智能侦察系统 (带时间轴完美对齐版)")
        self.resize(1300, 750)

        self.processor = SignalProcessor()
        self.svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=50.0, gamma='scale'))
        self.is_trained = False

        self.mod_types = ["ASK", "PSK", "FSK", "16QAM"]
        self.current_features = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左半边：特征观测区
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("<h2>📡 模块1&2：物理环境模拟与特征观测区</h2>"))

        ctrl_layout = QHBoxLayout()
        self.cb_mod = QComboBox()
        self.cb_mod.addItems(self.mod_types)
        ctrl_layout.addWidget(QLabel("未知信号截获:"))
        ctrl_layout.addWidget(self.cb_mod)

        self.cb_snr = QComboBox()
        self.cb_snr.addItems([f"{i} dB" for i in range(-10, 21, 2)])
        self.cb_snr.setCurrentText("10 dB")
        ctrl_layout.addWidget(QLabel("环境 SNR:"))
        ctrl_layout.addWidget(self.cb_snr)

        self.btn_generate = QPushButton("生成信号波形")
        self.btn_generate.clicked.connect(self.plot_micro_view)
        ctrl_layout.addWidget(self.btn_generate)
        left_layout.addLayout(ctrl_layout)

        self.fig_micro, (self.ax_b, self.ax_t, self.ax_f) = plt.subplots(3, 1, figsize=(5, 4.5))
        self.canvas_micro = FigureCanvas(self.fig_micro)
        left_layout.addWidget(self.canvas_micro)

        self.lbl_features = QLabel("7维指纹特征 X = [等待提取...]")
        self.lbl_features.setStyleSheet("color: blue; font-weight: bold; font-size: 12px;")
        left_layout.addWidget(self.lbl_features)

        left_layout.addWidget(QLabel("<h2>🎯 模块5：落地应用模块 (实时推断)</h2>"))
        self.btn_predict = QPushButton("对当前信号进行 AI 实时识别")
        self.btn_predict.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;")
        self.btn_predict.clicked.connect(self.realtime_predict)
        left_layout.addWidget(self.btn_predict)

        self.lbl_predict_res = QLabel("侦察结果：等待执行推断...")
        self.lbl_predict_res.setStyleSheet("font-size: 16px; color: #D32F2F; font-weight: bold;")
        left_layout.addWidget(self.lbl_predict_res)

        # 右半边：AI 训练与评估区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("<h2>🧠 模块3：AI 训练区 (Train)</h2>"))
        train_ctrl_layout = QHBoxLayout()
        self.btn_train = QPushButton("1. 生成多SNR混合数据集并训练模型")
        self.btn_train.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 10px;")
        self.btn_train.clicked.connect(self.run_offline_training)
        train_ctrl_layout.addWidget(self.btn_train)

        self.lbl_train_info = QLabel("模型状态: 未训练")
        train_ctrl_layout.addWidget(self.lbl_train_info)
        right_layout.addLayout(train_ctrl_layout)

        right_layout.addWidget(QLabel("<h2>📊 模块4：模型性能评估分析区 (Eval)</h2>"))
        eval_ctrl_layout = QHBoxLayout()
        self.btn_eval = QPushButton("2. 加载验证集进行多维度性能评估")
        self.btn_eval.setStyleSheet("background-color: #F57C00; color: white; font-weight: bold; padding: 10px;")
        self.btn_eval.setEnabled(False)
        self.btn_eval.clicked.connect(self.run_evaluation)
        eval_ctrl_layout.addWidget(self.btn_eval)
        right_layout.addLayout(eval_ctrl_layout)

        self.tabs = QTabWidget()

        self.tab_acc = QWidget()
        tab_acc_layout = QVBoxLayout(self.tab_acc)
        self.fig_macro, self.ax_macro = plt.subplots(figsize=(5, 4))
        self.canvas_macro = FigureCanvas(self.fig_macro)
        tab_acc_layout.addWidget(self.canvas_macro)
        self.tabs.addTab(self.tab_acc, "📉 全信噪比达标评估曲线")

        self.tab_cm = QWidget()
        tab_cm_layout = QVBoxLayout(self.tab_cm)
        self.fig_cm, self.ax_cm = plt.subplots(figsize=(5, 4))
        self.canvas_cm = FigureCanvas(self.fig_cm)
        tab_cm_layout.addWidget(self.canvas_cm)
        self.tabs.addTab(self.tab_cm, "🔍 混淆矩阵与误判分析")

        right_layout.addWidget(self.tabs)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(left_panel, 4)
        layout.addWidget(line)
        layout.addWidget(right_panel, 6)

        self.plot_micro_view()

    def plot_micro_view(self):
        mod_type = self.cb_mod.currentText()
        snr = int(self.cb_snr.currentText().split()[0])

        sig = self.processor.generate_signal(mod_type, snr)
        self.current_features = self.processor.extract_features(sig)

        self.lbl_features.setText(
            f"特征: [方差:{self.current_features[0]:.2f}, 峰均:{self.current_features[1]:.2f}, "
            f"峰度:{self.current_features[2]:.2f}, 熵:{self.current_features[3]:.2f}, "
            f"谱幅:{self.current_features[4]:.4f}, 主频:{self.current_features[5]:.0f}Hz, "
            f"峰数:{int(self.current_features[6])}]"
        )

        # =========================================================
        # 核心画图对齐逻辑
        # =========================================================

        # 图1：信源数字基带
        self.ax_b.clear()

        symbol_time_ms = (self.processor.sym_len / self.processor.fs) * 1000
        syms_to_show = self.processor.last_symbols[:19]
        time_axis = np.arange(20) * symbol_time_ms
        syms_padded = np.append(syms_to_show, syms_to_show[-1])

        self.ax_b.step(time_axis, syms_padded, where='post', color='green', linewidth=2)

        self.ax_b.set_title("信源数字基带 (底层真实的 0101 数据)", fontsize=10, fontweight='bold')
        self.ax_b.set_xlim(0, 6)
        self.ax_b.set_ylim(-0.2, 1.2)
        self.ax_b.set_yticks([0, 1])
        self.ax_b.set_xlabel("时间 (ms)", fontsize=9)
        self.ax_b.set_ylabel("逻辑电平", fontsize=9)
        self.ax_b.grid(True)

        # 图2：真实物理波形图
        self.ax_t.clear()
        t_ms = self.processor.t * 1000

        self.ax_t.plot(t_ms[:600], sig[:600], color='blue')
        self.ax_t.set_title(f"{mod_type} 真实物理波形 (上下时间轴严格对齐)", fontsize=10, fontweight='bold')
        self.ax_t.set_xlim(0, 6)
        self.ax_t.set_xlabel("时间 (ms)", fontsize=9)
        self.ax_t.set_ylabel("幅度 (V)", fontsize=9)
        self.ax_t.grid(True)

        # 图3：功率谱密度图
        self.ax_f.clear()
        f, Pxx = welch(sig, self.processor.fs, nperseg=1024)
        self.ax_f.plot(f / 1000, Pxx, color='orange')
        self.ax_f.set_title("功率谱密度 (频域特征)", fontsize=10)
        self.ax_f.set_xlabel("频率 (kHz)", fontsize=9)
        self.ax_f.set_ylabel("PSD (W/Hz)", fontsize=9)
        self.ax_f.grid(True)

        self.fig_micro.tight_layout(pad=1.0)
        self.canvas_micro.draw()

        self.lbl_predict_res.setText("侦察结果：等待执行推断...")

    def run_offline_training(self):
        self.btn_train.setEnabled(False)
        self.btn_train.setText("正在采集中...并训练...")
        QApplication.processEvents()

        snr_list = range(-10, 22, 2)
        # 【已修改】：训练数据量从 50 增加到 100
        samples_per_mod = 100
        X_train, y_train = [], []

        start_time = time.time()

        for snr in snr_list:
            for i, mod in enumerate(self.mod_types):
                for _ in range(samples_per_mod):
                    sig = self.processor.generate_signal(mod, snr)
                    feat = self.processor.extract_features(sig)
                    X_train.append(feat)
                    y_train.append(i)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.svm_model.fit(X_train, y_train)

        end_time = time.time()
        train_cost = end_time - start_time

        self.is_trained = True
        self.btn_train.setText("1. 重新生成并训练模型")
        self.btn_train.setEnabled(True)
        self.btn_eval.setEnabled(True)

        self.lbl_train_info.setText(
            f"✅ 训练完成! 样本量: {len(X_train)} | "
            f"<font color='red'><b>总耗时: {train_cost:.2f} 秒</b></font>"
        )
        QMessageBox.information(self, "训练成功", "SVM模型已成功训练，现保存在内存中，可前往模块4进行评估！")

    def run_evaluation(self):
        self.btn_eval.setEnabled(False)
        self.btn_eval.setText("正在执行严苛评估...")
        QApplication.processEvents()

        snr_list = range(-10, 22, 2)
        accuracy_list = []
        # 【已修改】：测试评估数据量从 100 减少到 40
        samples_per_mod = 40

        y_true_all = []
        y_pred_all = []

        for snr in snr_list:
            X_test, y_test = [], []
            for i, mod in enumerate(self.mod_types):
                for _ in range(samples_per_mod):
                    sig = self.processor.generate_signal(mod, snr)
                    feat = self.processor.extract_features(sig)
                    X_test.append(feat)
                    y_test.append(i)

            y_pred = self.svm_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            accuracy_list.append(acc)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        self.ax_macro.clear()
        self.ax_macro.plot(snr_list, accuracy_list, 'ro-', linewidth=2, markersize=7)
        self.ax_macro.axhline(90, color='blue', linestyle='--', label="90% 工程达标线")
        self.ax_macro.set_title("独立测试集性能评估 (未见数据)")
        self.ax_macro.set_xlabel("环境 SNR (dB)")
        self.ax_macro.set_ylabel("正确率 Accuracy (%)")
        self.ax_macro.set_ylim(0, 105)
        self.ax_macro.grid(True)
        self.ax_macro.legend()
        self.canvas_macro.draw()

        self.ax_cm.clear()
        cm = confusion_matrix(y_true_all, y_pred_all)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.mod_types)
        disp.plot(ax=self.ax_cm, cmap=plt.cm.Blues, colorbar=True)

        total_samples = len(y_true_all)
        error_samples = total_samples - np.trace(cm)
        error_rate = (error_samples / total_samples) * 100

        self.ax_cm.set_title(f"多级信噪比综合混淆矩阵 (全局误判率: {error_rate:.2f}%)")
        self.fig_cm.tight_layout()
        self.canvas_cm.draw()

        self.btn_eval.setText("2. 加载验证集进行多维度性能评估")
        self.btn_eval.setEnabled(True)
        self.tabs.setCurrentIndex(1)

    def realtime_predict(self):
        if not self.is_trained:
            QMessageBox.warning(self, "警告", "尚未部署AI模型！请先在右侧进行模块3的训练。")
            return

        if self.current_features is None:
            return

        pred_idx = self.svm_model.predict([self.current_features])[0]
        pred_mod = self.mod_types[pred_idx]

        true_mod = self.cb_mod.currentText()
        if pred_mod == true_mod:
            self.lbl_predict_res.setText(f"✅ 侦察结果：【{pred_mod}】 (识别正确)")
            self.lbl_predict_res.setStyleSheet("font-size: 16px; color: green; font-weight: bold;")
        else:
            self.lbl_predict_res.setText(f"❌ 侦察结果：【{pred_mod}】 (发生误判，真实为{true_mod})")
            self.lbl_predict_res.setStyleSheet("font-size: 16px; color: red; font-weight: bold;")


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())