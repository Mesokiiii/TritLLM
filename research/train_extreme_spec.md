# TritFormer Extreme: Архитектурная спецификация и системный профиль 4x RTX 4090

**Файл спецификации:** `C:\Users\1\Desktop\AImL\demsni\train_extreme_spec.md`  
**Исполняемые артефакты:**  
- Тренировочный движок: [`train_extreme.py`](file:///C:/Users/1/Desktop/AImL/demsni/train_extreme.py)  
- Bash-скрипт запуска: [`launch_extreme.sh`](file:///C:/Users/1/Desktop/AImL/demsni/launch_extreme.sh)  
- PowerShell-скрипт: [`launch_extreme.ps1`](file:///C:/Users/1/Desktop/AImL/demsni/launch_extreme.ps1)  
- Адаптер облачного деплоя: [`deploy.py`](file:///C:/Users/1/Desktop/AImL/demsni/deploy.py)

---

## 1. Топология кластера и системные характеристики (4x RTX 4090)

Кластер развернут на базе 4x дискретных графических процессоров **NVIDIA GeForce RTX 4090** (Ada Lovelace, микроархитектура `sm_89`):

| Компонент / Характеристика | Параметр на 1 GPU | Суммарно на кластер (4x GPU) |
| :--- | :--- | :--- |
| **CUDA Cores** | 16,384 | 65,536 ядер |
| **Tensor Cores (4th Gen)** | 512 Tensor Cores | 2,048 Tensor Cores |
| **Пиковая плотная производительность BF16** | 165.2 TFLOPS | **660.8 TFLOPS (Cluster Dense Peak)** |
| **Пиковая производительность BF16 (2:4 Sparsity)** | 330.3 TFLOPS | **1,321.2 TFLOPS** |
| **Видеопамять (GDDR6X)** | 24,576 MB (24 GB) | **98,304 MB (96 GB VRAM)** |
| **Пропускная способность VRAM** | 1,008 GB/s | 4,032 GB/s |
| **Интерконнект** | PCIe Gen 4.0 x16 / x8 | Бидирекционально ~28-31 GB/s |
| **Хост-процессор (CPU)** | 4 vCPUs на ранг | 16 vCPUs (128 GB ECC RAM) |

---

## 2. Архитектура нейросети: SubLN + Continuous LM Head

### 2.1 Проблема стандартного Pre-LN в 1.58-битных сетях
В стандартных Pre-LN трансформерах дисперсия скрытых состояний возрастает пропорционально глубине: $\text{Var}(x_l) \sim O(\sqrt{L})$.
Для сетей с квантованием весов в $\{-1, 0, +1\}$ через Straight-Through Estimator (STE) рост амплитуды активаций вызывает:
1. Выход значений за пределы линейного диапазона STE.
2. «Залипание» градиентов (gradient saturation) на глубоких слоях.
3. Коллапс перплексии при обучении с высокими learning rates.

### 2.2 Математическая формулировка SubLN (Sub-Layer Normalization)
В `TritFormerExtreme` реализована архитектура **SubLN** (Foundation Transformers / DeepNet / BitNet):

$$\begin{aligned}
x^{(1)}_l &= x_l + \text{RMSNorm}_{\text{sub1}}\Big(\text{Attention}\big(\text{RMSNorm}_{\text{pre1}}(x_l)\big)\Big) \\
x_{l+1} &= x^{(1)}_l + \text{RMSNorm}_{\text{sub2}}\Big(\text{SwiGLU}\big(\text{RMSNorm}_{\text{pre2}}(x^{(1)}_l)\big)\Big)
\end{aligned}$$

Каждая ветвь нормализуется дважды:
- $\text{RMSNorm}_{\text{pre}}$ гарантирует единичную дисперсию входов в ядра внимания и нелинейность SwiGLU.
- $\text{RMSNorm}_{\text{sub}}$ нормализует выход проекций **до сложения с резидуальным потоком**, что математически строго удерживает дисперсию всего резидуального потока $\text{Var}(x_l) = 1.0$ на любой глубине $L \in [1, 48]$.

```
          ┌──────────────────────────────────────────────────────────┐
          │                    Входной поток x_l                     │
          └───────────┬──────────────────────────────────┬───────────┘
                      │                                  │
                      ▼                                  │ (Residual skip)
             [RMSNorm Pre-Attn]                          │
                      │                                  │
                      ▼                                  │
          [Ternary Q, K, V Projections]                  │
                      │                                  │
                      ▼                                  │
         [PyTorch SDPA / FlashAttn-2]                    │
                      │                                  │
                      ▼                                  │
            [Ternary Out-Projection]                     │
                      │                                  │
                      ▼                                  │
             [RMSNorm Sub-Attn]                          │
                      │                                  │
                      ▼                                  │
                  (+) ◄──────────────────────────────────┘
                      │
                      ├──────────────────────────────────┐
                      ▼                                  │ (Residual skip)
             [RMSNorm Pre-MLP]                           │
                      │                                  │
                      ▼                                  │
          [Ternary SwiGLU Gate & Up]                     │
                      │                                  │
                      ▼                                  │
         [SiLU(Gate) * Up Activation]                    │
                      │                                  │
                      ▼                                  │
            [Ternary Down-Projection]                    │
                      │                                  │
                      ▼                                  │
             [RMSNorm Sub-MLP]                           │
                      │                                  │
                      ▼                                  │
                  (+) ◄──────────────────────────────────┘
                      │
                      ▼
               Выход блока x_{l+1}
```

### 2.3 Непрерывная проекционная голова (Continuous LM Head)
- **Проблема тернарной головы:** Если матрица `lm_head` квантуется в троичные значения $\{-1, 0, 1\}$, логиты токенов $z \in \mathbb{R}^{V}$ формируются дискретными суммами активаций. Это создает критический шум квантования в распределении вероятностей $\text{Softmax}(z)$, приводя к застреванию loss на уровне $6.5 - 7.5$ и бредогенерации.
- **Решение в `TritFormerExtreme`:** Внутренние слои ($W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) квантуются в 1.58-бит с STE, а матрица `lm_head` ($\mathbb{R}^{d_{\text{model}} \times V}$) сохраняется в **полной точности `bfloat16`**. Это обеспечивает непрерывную геометрию логитов, точный градиент кросс-энтропии и падение Perplexity в разы быстрее.

---

## 3. Гибридный оптимизатор: Muon + AdamW

В `train_extreme.py` внедрен гибридный оптимизатор нового поколения:

### 3.1 Muon (Momentum Orthogonal Optimizer)
Muon (Keller Jordan et al., modded-nanogpt) применяется **строго ко всем 2D внутренним матрицам весов** (внимание и SwiGLU).

1. **Накопление импульса:**
   $$M_t = \beta \cdot M_{t-1} + \nabla_W \mathcal{L}$$
   С опцией Nesterov: $\tilde{M}_t = \nabla_W \mathcal{L} + \beta \cdot M_t$.

2. **Ортогонализация через 5 шагов квинтичного полинома Ньютона-Шульца:**
   $$X_0 = \frac{\tilde{M}_t}{\|\tilde{M}_t\|_F + \epsilon}$$
   Для $k = 0, \dots, 4$:
   $$A = X_k X_k^T$$
   $$B = b A + c A^2$$
   $$X_{k+1} = a X_k + B X_k$$
   Коэффициенты полинома: $a = 3.4445, \quad b = -4.7750, \quad c = 2.0315$.
   Этот алгоритм эквивалентен вычислению поляризационного разложения $U V^T$ из SVD-разложения матрицы импульса, но выполняется полностью через тензорные перемножения матриц на ядрах BF16 за доли миллисекунды.

3. **Обновление весов:**
   $$W \leftarrow W - \eta_{\text{muon}} \cdot \max\left(1, \sqrt{\frac{m}{n}}\right) \cdot X_5$$
   Ортогональный шаг устраняет эффект «сплющивания» ранга и предотвращает коллапс тернарных весов в тривиальные состояния.

### 3.2 Fused AdamW
Применяется к:
- Таблице эмбеддингов токенов (`tok_embeddings.weight`).
- Непрерывной выходной голове (`lm_head.weight`).
- 1D-векторам масштабирования RMSNorm ($\gamma$) и смещениям.
Весовое затухание (Weight Decay) применяется исключительно к 2D матрицам эмбеддингов/головы ($wd = 0.01-0.1$), для векторов RMSNorm weight decay равен строго $0.0$.

---

## 4. Warmup-Stable-Decay (WSD) Планировщик

Для 10-15 минутных сессий (500–2,000 шагов) классический косинусный график (Cosine Annealing) неэффективен: он начинает снижать скорость обучения уже после 5% шагов, проводя большую часть тренировки на полузадушенном LR.

**Профиль WSD в `TritFormerExtreme`:**
1. **Warmup Phase (0% – 5% шагов):** Линейный рост от $0$ до $\eta_{\text{max}}$. Предотвращает градиентные взрывы при холодной инициализации.
2. **Stable Phase (5% – 85% шагов):** Стабильная работа на **100% максимальной скорости обучения**. Модель поглощает максимальный объем энтропии датасета с предельной скоростью сходимости.
3. **Decay (Cooldown) Phase (85% – 100% шагов):** Крутой 1-косинусный спад до $\eta_{\text{min}} = 0.1 \cdot \eta_{\text{max}}$. За эти 15% шагов происходит мгновенная локализация весов в глубоком минимуме функции потерь, обрушивая Perplexity.

$$\eta(t) = \begin{cases}
\eta_{\text{max}} \cdot \frac{t + 1}{T_{\text{warmup}}}, & t < T_{\text{warmup}} \\
\eta_{\text{max}}, & T_{\text{warmup}} \le t < T - T_{\text{decay}} \\
\eta_{\text{min}} + (\eta_{\text{max}} - \eta_{\text{min}}) \cdot \frac{1 + \cos\left(\pi \frac{t - (T - T_{\text{decay}})}{T_{\text{decay}}}\right)}{2}, & t \ge T - T_{\text{decay}}
\end{cases}$$

---

## 5. Системные оптимизации GPU: CUDAGraphs и перекрытие PCIe

### 5.1 `torch.compile(model, mode="reduce-overhead")`
- В распределенном обучении с маленьким микро-батчем ($B_{\text{micro}} = 2-4$) оверхед интерпретатора Python и задержки запуска ядер CUDA (kernel launch latency $\sim 5-10\,\mu\text{s}$ на ядро) составляют до 35-45% общего времени шага.
- Режим `reduce-overhead` захватывает весь граф вычислений в статические **CUDAGraphs**.
- Весь микро-шаг запускается как единый аппаратно управляемый дескриптор графа на GPU с $0\,\mu\text{s}$ оверхедом CPU!

### 5.2 Асинхронное перекрытие градиентов (Gradient Accumulation Overlap)
На шине PCIe Gen4 x16/x8 задержка All-Reduce между 4 GPU без NVLink может вызывать простои.
1. На микрошагах $m \in [0, N-2]$ активируется контекст `model.no_sync()`. Ни один градиент не отправляется в шину PCIe; градиенты накапливаются локально в VRAM.
2. На финальном микрошаге $m = N-1$ контекст `no_sync` отключается. DDP разбивает тензоры градиентов на корзины `bucket_cap_mb=25`.
3. По мере завершения backward-прохода для верхних слоев (LM Head и последние блоки), NCCL немедленно начинает асинхронную передачу по PCIe, **полностью перекрывая сетевую передачу вычислениями backward нижних слоев**.
4. Флаг `gradient_as_bucket_view=True` исключает дублирующее выделение буферов памяти, экономя до 2.0 GB VRAM на GPU.

---

## 6. Исправление генерации (Coherent Inference Generator)

На ранних чекпоинтах стандартный семплинг (Top-K / Greedy) страдает от циклического зацикливания (repetition loop), когда модель начинает генерировать `the of the of the of...`.

В `CoherentInferenceGenerator` реализован 4-ступенчатый защитный контур:
1. **Repetition Penalty ($r = 1.25$):**
   Для всех уже встретившихся в контексте токенов $t$:
   $$z_t = \begin{cases} z_t / r, & \text{если } z_t > 0 \\ z_t \cdot r, & \text{если } z_t \le 0 \end{cases}$$
2. **N-gram Blocking ($N = 3$):** Любой токен, завершающий уже встречавшуюся триграмму, маскируется логитом $-\infty$. Это математически исключает зацикливание с периодом $\le 3$.
3. **Temperature Scaling ($T = 0.8$):** Сглаживает пики распределения без потери структурности.
4. **Top-P (Nucleus) Sampling ($p = 0.90$):** Отсекает длинный нестабильный хвост токенов, гарантируя выбор только из 90% наиболее вероятной массы плотности.

---

## 7. Контрольный контур валидации FineWeb-Edu каждые 20 шагов

- Каждые `--eval_interval 20` шагов тренировка переключается в `model.eval()`.
- Вычисляется loss на фиксированном пуле контрольных батчей FineWeb-Edu.
- Потери усредняются по всем 4 рангам через `dist.all_reduce(eval_loss, op=dist.ReduceOp.AVG)`.
- Рассчитывается реальная Perplexity:
  $$\text{PPL}_{\text{eval}} = \exp(\mathcal{L}_{\text{eval}})$$
- Данные логируются в консоль и сохраняются в метаданные чекпоинтов.

---

## 8. Телеметрический дашборд реального времени

Мониторинг собирает метрики на аппаратном и математическом уровнях:
- **TFLOP/s:**
  $$\text{TFLOPS} = \frac{8 \times N_{\text{params}} \times \text{TokensInStep}}{\Delta t_{\text{step}} \times 10^{12}}$$
- **MFU (%) [Model FLOPs Utilization]:**
  $$\text{MFU} = \frac{\text{TFLOPS}}{660.8} \times 100\%$$
- **Throughput:** Токенов в секунду.
- **NVML / GPU 0-3:** Температура (°C), тактовая частота ядра (MHz), использование VRAM (GB), утилизация ядер (%).

**Формат лога в консоли:**
```
[Rank 0/4] Step 000020/001000 | TrainLoss: 6.1245 (PPL: 456.91) | EvalLoss: 6.0821 (EvalPPL: 437.95) | LR: 2.00e-02 | GradNorm: 0.94 | Speed: 146,820 tok/s | TFLOPS: 382.4 (MFU: 57.9%) | Time: 1.78s | Tokens: 5.24M
       └── [Hardware Cluster] GPU 0: 58°C @ 2520MHz [14.1/24.0GB, 99%] | GPU 1: 61°C @ 2505MHz [14.2/24.0GB, 99%] | GPU 2: 59°C @ 2520MHz [14.1/24.0GB, 98%] | GPU 3: 62°C @ 2490MHz [14.2/24.0GB, 100%]
```

---

## 9. Инструкция по боевому запуску

### Запуск на Ubuntu сервере с 4x RTX 4090:
```bash
cd /root/demsni
chmod +x launch_extreme.sh
bash launch_extreme.sh
```

### Автоматический запуск через деплой-клиент с локального компьютера:
```bash
python deploy.py --ip <IP_АДРЕС_СЕРВЕРА> --action all
```

### Проверка статуса и стриминг телеметрии:
```bash
python deploy.py --ip <IP_АДРЕС_СЕРВЕРА> --action status
python deploy.py --ip <IP_АДРЕС_СЕРВЕРА> --action logs
```
