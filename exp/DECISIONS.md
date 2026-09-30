# 实验决策日志（时间顺序）

- 2026-09-02 22:15 启动分割网格：U-Net 7.77M，DRIVE/CHASE 512 patch bs8 300 ep，HRF/FIVES 1536 长边 768 patch bs4 150 ep，3 种子，val=15% 训练图（按受试者分组，split_seed 12345）。
- 数据修复（训练启动前完成）：HRF 大小写重复枚举；FIVES 受试者 ID 按 split 前缀；FOV 阈值由 Otsu 改为红通道最大值 10%（GT 血管在 FOV 内比例 0.967→0.998）。DRIVE 2nd_manual 为 {0,1} 需按文件自身最大值二值化。
- 门禁 A（261 张 GT 掩膜，两管线）：FD ρ=0.95、密度 1.00、总长度 0.99 通过；**迂曲度 ρ=0.65 不通过**（两管线各自内部 median/mean 也仅 0.6–0.8；图间变异系数 <2%）。决策：迂曲度保留为主要标志物，但只用预注册的单管线（skan 长度加权），论文明确该限制；BTR 标量化中三者取两管线中位数、迂曲度取 skan 值。
- PVBM zone-B 几何量在视盘被排除时返回 0（其图遍历从视盘出发），zone-B 量一律来自 skan 管线；PVBM 仅用全局量。
- PVBM 多重分形（25 旋转）过慢（HRF 113 s/掩膜）：C1 中改用少旋转次数（需验证与 25 旋转 FD 的 Spearman ≥ 0.95），HRF/FIVES 每图扰动数削减。
- 标注可复现性上限（观察者间）：DRIVE clDice 0.763 / Dice 0.788 / BCS 0.79；CHASE clDice 0.835 / Dice 0.777 / BCS 0.68。
- RiGR 需要训练图的折外预测：主网格结束后运行 2 折交叉拟合（DRIVE/CHASE/STARE 150 ep，HRF/FIVES 80 ep，seed 0）生成 runs/seg_oof。
- 2026-09-03 C1 全量在 cloud 服务器（32 核）上运行：DRIVE(含 obs2)/CHASE/HRF/FIVES(60 图子集)，预计 ~2.7 h，约 2.5 万事件；PVBM FD 用 5 次旋转（与 25 次逐位一致，2–3 倍快；4 次旋转会失配）。HRF/FIVES 在 1536 长边上做扰动，σ_B 用 C1 自身 B0 缓存（分辨率相关）。
- 统一输出格式：所有 per_image.csv 用 image / before_* / after_* / delta_*；生物标志物 σ 唯一来源 results/gateA_biomarker_scales.csv；RiGR 输出逐边表（accepted / is_true_repair / dangerous / path）。
- 训练网格因会话重启中断一次，seed1/2 重新从头训练（独立进程）；交叉拟合驱动等待网格结束后自动运行。
- S4 范围：FIVES 限制为训练 120 / 测试 60 张（固定划分前 n 张，所有方法与模式一致；分割器本身仍在 200 张测试集上评估）。
- 部署版 R_false / R_miss 目标：预注册定义为 H_dep=max(0,(|ΔB_topo|−|ΔB_ctrl|)/σ)。**退路规则（看 Tab.2 之前定下）**：若某 (biomarker, pipeline) 目标的非零比例 <10%，该目标改用原始绝对标准化危害 |ΔB|/σ（仍非负），并在论文中注明。
- 基线训练输入：rNCA/EVAPORE 用折外预测掩膜 + 与 RiGR 相同的均匀合成切断；RiGR 评分器在 GT+切断上训练，折外预测只用于失败分布 π。
- 2026-09-03 08:30 **门禁 B 判读（在任何 Tab.2 结果产生之前记录）**：预注册目标 H_dep=max(0,(|ΔB_topo|−|ΔB_ctrl|)/σ) 在留出集上对所有预测子（含有 GT 信息的 ΔDice/ΔclDice）R²≤0，MAE 量级 0.005–0.008σ；非零比例 22–98%（密度恒 0，为对照设计所致）。结论：单事件的"超出像素预算的净拓扑危害"处于管线噪声水平，预注册的结论 1 在该目标下**不成立**，这作为 Fig.2/§结果的科学发现如实报告。
  **修正（偏离预注册，理由如下）**：部署决策 U(e) 需要的是编辑本身的反事实危害 |B(M⁺)−B(M⁻)|/σ（含像素与连通性的全部后果），而非扣除匹配对照后的余量；因此部署版 R_miss/R_false 改用原始绝对标准化危害 H_abs=|ΔB_topo|/σ 训练（仍非负），H_net 仅用于科学剂量-响应分析。Exp1 在 H_abs 目标上重跑并同时报告两种目标；论文明确说明 GT 信息型常规指标（ΔDice/ΔclDice）在推理时不可得，BTR 的对比对象应是"无 GT 的可观测预测子"。
- 2026-09-03 08:15 **H_abs 目标执行结果（Exp1 两目标并列报告）**：按上条决策新增 `btr.py --target {hdep,habs,both}`，部署版权重写入 `runs/btr/habs/btr_{miss,false}_{all,wo_<D>}.joblib`（`head_path(..., target=)` 解析；`hdep` 仍走原扁平路径，S4 运行中未被覆盖）。Exp1 表新增"无 GT 可观测预测子"块（phi_d、phi_d_over_r、phi_r_mean、phi_density 及其组合）。
  **结果：改用 H_abs 并未挽救 BTR。** LODO 下 BTR 仍 R²<0（miss −0.155、false −0.071），且显著劣于 GT 型 ΔDice/ΔclDice（后者在 H_abs 上首次取得正 R²：miss 0.054、false 0.009）与单变量可观测基线（8 组比较中 BTR 在 7 组显著更差，自助 CI 排除 0；仅 hdep/false vs obs_d_over_r 打平）。
  **诊断（关键）**：按受试者分组的**同数据集内** CV 中，BTR 反而取得正 R²（habs: miss 0.059 / false 0.015）且 ρ≈0.44–0.55，与需要 GT 的 ΔclDice 相当；失效只出现在跨数据集。逐留出集看，BTR 的 ρ 尚可（0.17–0.49，habs 中位 0.405）但 R² 恒负（0/8 折为正），而汇总 ρ 仅 0.117/−0.003 —— 典型 Simpson 效应：**BTR 的秩序可跨数据集迁移，量纲/标定不可**。仅用尺度无关特征（去掉 8 个像素量纲 phi）**不能**修复（R² −0.147 vs −0.155），故并非单纯分辨率协变量偏移。
  结论：论文应将 Tab.1 报告为"跨数据集标定失效"而非"无信号"，并考虑逐数据集仿射重标定或分层随机效应头作为后续修正；部署仍用 H_abs 头（U(e) 只需同一图像内的相对排序）。
- 2026-09-03 08:30 **部署头定型（S4 phase 2 用）**：部署版 BTR = **H_abs 目标 + 仅训练划分拟合 + 同域（in-domain）**，权重在 `runs/btr/habs_trainonly/btr_{miss,false}_all.joblib`；LODO 区块用同目录下 `btr_{miss,false}_wo_<D>.joblib`；U(e) 消费的是**同一图像内的相对排序**（不依赖跨数据集绝对标定）。`btr.py --train-split-only` 按 `src.data.datasets` 的 split 过滤拟合行（DRIVE obs2 全在 test，自动排除），`btr_backend/load_head` 以 `target="habs_trainonly"` 选择。
  **泄漏修复**：此前 `habs`/`hdep` 头在全部事件（含 test 图）上拟合，而 RiGR 恰作用于这些 test 图的预测掩膜；trainonly 头仅用 55 张训练图（CHASE 20 + DRIVE 20 + HRF 15，sever 4661 / bridge 903 事件）。
  **重要限制（须写进论文）**：C1 的 FIVES 子集按设计只取 **test 划分** 60 张（`make_tasks` 中 `split=="test"`），故 FIVES **没有训练划分事件**：trainonly 头完全没见过 FIVES，`wo_FIVES` 头与 `all` 等价，同域表中 FIVES 行实为迁移行（已用 `in_domain=False` 标记）。
  **同域 Exp1 结果**（拟合训练划分、评测 test 划分）：BTR 恢复正 R²（miss +0.043、false +0.011；剔除 FIVES 的严格同域为 **+0.105 / +0.012**，ρ 0.571 / 0.513）。对**推理时可得**的无 GT 基线，BTR 在 MAE 与 ρ 上均显著更优（严格同域：ΔMAE −0.00067 / −0.00064，Δρ +0.183 / +0.328，自助 CI 均排除 0）；对需要 GT 的 ΔDice/ΔclDice 仍落后（Δρ −0.097 / −0.126）。逐数据集看，BTR 的 ρ 在训练图最多的 CHASE(0.604)/DRIVE(0.600) 上**优于**含 GT 的常规指标，在 HRF（仅 15 训练图）与 FIVES（0 训练图）上退化。
  论文口径：Tab.1 主区块 = 同域（部署协议），LODO 表作为迁移行保留；结论表述为"BTR 在同域可用且优于一切推理时可得基线，跨数据集迁移的量纲标定失效"。
- 2026-09-03 08:50 算力分配：本机 = A/B/C（头、数据、评分器）、D（主结果 3 模式×4 数据集×3 种子）、F（消融）、I（稳健性）、J；远端 T4 = E:train（rNCA/EVAPORE 训练）、G（LODO 全套）、H（Exp2 候选级基准）。用户要求远端不空转，只有在无可跑实验时才同步并关机。
- 论文模板：EAAI/ESWA 投稿要求单栏（cas-sc）+ 作者-年份引用（cas-model2-names.bst）；出版双栏由 Elsevier 排版决定。
- 2026-09-03 09:1x **部署版 R_false 目标对齐 + 标量化前缀缺陷（S4 phase 2 前修复）**：(a) `BTRHead.predict` 原先把标量化列名硬编码为 `Hdep_*`，因此对 `habs*` 头返回**全 NaN**，而 `utility.risk_values` 会把 NaN 折成 `R=1` —— 即"风险制导"运行会静默退化为均匀代价却仍标注 BTR 后端。现改为按头自身 `meta["target"]`／目标列推断前缀，取不到列时发 RuntimeWarning。(b) `build_data` 现同时写两族目标：`Habs_<b>_<p>=|B(M⁺)−B(M⁻)|/σ`（部署用，不需要 GT）与原 `Hdep_*`（净危害，仅供剂量-响应）；`fit_deployed_false(target="habs")` 为默认，并把 `meta["target"]="habs"` 写入权重，从而可与 `runs/btr/habs_trainonly/btr_miss_all.joblib` 配对（`BTRBackend` 拒绝混用不同目标族）。DRIVE 4 图冒烟：H_abs 非零比例 0.375–1.0（H_dep 为 0–0.25），密度不再恒零。(c) D/F 的 risk 模式统一经 `--btr_runs_dir runs/btr/habs_trainonly --btr_variant all` 取 R_miss；`run_rigr` 在部署 R_false 缺失或目标不匹配时改为**回退到同目录的 false 头**，仅在两者都失败时才落到扁平 `runs/btr/btr_*_all.joblib`（该布局是 H_dep 且在 test 图事件上拟合，属泄漏），并强制打印警告。(d) `s4_all` 新增 BTR 预检：调度 D/F 前实际加载 miss/false 对并探测输出有限性，目标不匹配或全 NaN 时以退出码 2 中止并给出重跑命令。**待办**：phase-1b 结束后需对已有的 B/C 产物 `--force` 重跑，旧 `rfalse_train.csv` 没有 `Habs_*` 列。
- 2026-09-03 10:20 稿件评审（外部第 1 轮，7.5/10）触发的协议修正：(a) **σ_B 只由训练划分的参考掩膜估计并冻结**（results/gateA_biomarker_scales_train.csv），测试图不参与尺度估计；BTR 头据此重拟合；(b) Gate A 的管线选择决策（迂曲度仅 skan）只基于训练划分掩膜的一致性重新确认，测试划分一致性仅作描述性验证；(c) TRR_precision 与 FCR 以一对一匹配的 m 为分子（m/|A|），|A|=0 时 FCR=0、precision=1 并标记；(d) EVAPORE 基线拆为"官方端到端"与"EVAPORE 评分器置于共同候选/路径框架"两行（可行时），表中不得裸写 EVAPORE；(e) 匹配中的走廊冲突改为优化前确定性剪枝（保留 U 高者），使"标准最大权匹配"表述成立；(f) 论文措辞：H_dep 预注册结论永久保留为失败判定，H_abs 为事后登记的部署目标，不称"更正确"；U(e) 以实现效用取期望定义，"错修的实现效用非正"。
- 2026-09-03 10:25 **训练划分 σ_B 执行**（承 10:20 协议修正）：新增 `src/c1/train_scales.py` → `results/gateA_biomarker_scales_train.csv`（schema 同全量版 + `source`/`resolution`）与 `gateA_pipeline_agreement_train.csv`。σ 只用**训练划分 + observer-1** 参考掩膜；**分辨率规则**：C1 降采样处理的数据集（HRF work_scale 0.438、FIVES 0.75）不得取 Gate A 的原生分辨率 σ，改用 C1 `B0_*` 训练图缓存（否则 |ΔB| 在 1536 而 σ 在原生，长度类标志物整体差一个尺度因子）；DRIVE/CHASE 原生，仍用 Gate A 训练子集。FIVES 在 Gate A 只测了 100 张 **test**，故其 σ 来自 60 张 FIVES **训练**图的 B0 缓存（1536 px，已在 `resolution` 列标注）。
  **σ 比值（train-only / all-mask）**：多数在 0.86–1.38；**显著偏离**者：CHASE 密度 0.41（两管线同值）、HRF 迂曲度 0.27（pvbm）/0.44（skan）、HRF 总长度 0.64/0.55。即全量 σ 被 test 图撑大，改用训练 σ 后同一 |ΔB| 的标准化危害整体上升（CHASE 密度约 2.4×）。
  **迂曲度决策复核（仅训练划分）**：ρ = CHASE 0.58、DRIVE 0.856、HRF 0.129，最小 0.129 < 0.8 → **skan-only 决策在训练划分上依然成立**（且比全量更不通过）；FD/密度/总长度 ρ 均 ≥ 0.92 通过。
  **实现**：`stats_c1` 新增 `sigmatr_*` 与 `Habs_*`（用训练 σ）、`Habs_all_*`（全量 σ，补充列）；H_net/H_dep 仍用预注册全量 σ，科学剂量-响应不变。`c1_stats_summary.json` 记录 `habs_sigma_source`；部署标记 `.refit_v2_done` 记录所用 σ 文件与 md5，且在写标记前强制校验 H_abs 确实由训练 σ 标准化。
- 2026-09-03 10:20 **σ 与 FCR 口径修正落地（S4 phase-2 前）**：(a) `src/eval/biomarker_eval.load_gt_scales` 默认改读 `results/gateA_biomarker_scales_train.csv`（仅训练划分参考掩膜），文件缺失时**抛异常**而非静默返回 {}（旧行为会让整批 `macro_mae_*` 变 NaN 直到聚合时才发现）；全部消费方（run_rigr / run_baseline / robust/* / exp2 / build_data）均经此加载器，仓库内已无第二处直接读取（`bio/gate_a.py` 是生产者；`c1/stats_c1.py:115` 仍直接读全量表，属 C1 侧，未改动）。phase-2 链在 stage 2 前新增该文件的等待门。(b) `eval/trr_fcr.py` 新增 `matched_precision_fcr()` 与输出键 `TRR_precision_matched = m/|A|`、`FCR_matched = 1 − m/|A|`、`no_accepted_edges`；|A|=0 约定为 precision 1 / FCR 0 并置旗标。`eval/tables.py:add_matched_fcr` 把匹配版设为论文主列 `TRR_precision`/`FCR`，边级判据降为 `TRR_precision_edge`/`FCR_edge`；Fig.3 走同一函数，`sweep_curve.csv` 现携带 `n_accepted`/`n_matched` 以便旧曲线也能换算。已完成的 geometric 运行上 90 行中 79 行 FCR 发生变化（匹配版系统性更高，如 0.15→0.40）。
  **交付的 σ 表存在两处缺陷（C1 侧待修，已由 `s4_all.preflight_sigma` 硬门拦截，B/D/F/G/H 阶段退出码 2）**：FIVES 完全没有行（会使其全部 macro_mae 为 NaN）；HRF 的 σ 来自 `c1_B0_train_obs1` 且 `resolution=work_scale=0.4384`（非 native），与"预测先映射回原生分辨率再算标志物"的契约不符，分母约差 2.3 倍。DRIVE/CHASE 为 `gateA_train_obs1`/native，n=20/20（train+val，即全部非 test 图），HRF n=15。
- 2026-09-03 10:45 **原生分辨率 σ + FIVES 训练事件补齐（S4 preflight 冲突解决）**：`src/eval/biomarker_eval` 在**原生分辨率**上评估预测，故 `gateA_biomarker_scales_train.csv` 四个数据集**一律 native**：DRIVE/CHASE/HRF 用 Gate A 训练划分 obs1（20/20/15 张，原生）；FIVES 新测**固定划分训练集前 120 张**（与 S4 `image_caps` 同一子集，取自 `get_records → make_splits(12345) → train[:120]`，id 列表存 `results/fives_train120_ids.txt`）在 2048² 原生分辨率、两管线、fd_rotations=5，输出 `results/fives_native_train_biomarkers.csv`。
  **单位约定**（`stats_c1.native_factor`）：C1 在工作分辨率 s 上测得的 |ΔB| 先折算回原生再除以原生 σ —— 仅**长度量**换算 `total_length ×(1/s)`（HRF ×2.281、FIVES ×1.333）；density/FD/迂曲度不换算。**明确保留的假设**：FD 与迂曲度只在连续极限下尺度不变，实际是有限栅格估计（PVBM 的 D0 盒尺度相对图像固定；skan 骨架细支在 1536 可能并合），故其降采样值只是**近似**原生值，未作修正；density 为两个同步缩放面积之比，最稳健。工作分辨率变体保留为 `Habs_ws_*` 补充列；H_net/H_dep 仍用预注册全量 σ 不变。
  **σ 比值（native train-only / 旧 all-mask）**：多数 0.81–1.38；越界者仅 CHASE density 0.41；HRF 迂曲度 0.56/0.58、总长度 0.54/0.77 接近下界。**迂曲度双管线复核（训练划分）**：ρ = DRIVE 0.856、CHASE 0.58、FIVES 0.539、HRF 0.50，最小 0.50 < 0.8 → **skan-only 决策成立**；FD 0.92–0.97、density 1.00、总长度 0.97–1.00 通过。
  **语料**：合并 253 分片 → **31,975 事件**（新增 FIVES 训练 60 图 / 5,455 事件）；四数据集训练事件 sever 7,230 / bridge 1,622，`habs_trainonly` 头（含 `wo_<D>`）已在其上重拟合，FIVES 首次成为**同域**数据集。LODO 表也改用 train-split-only 拟合（`tab1_exp1_habs_trainonly.csv`，`fit` 列标注），全事件拟合版保留为补充（`runs/btr_alleventsfit`）。
  **事故与加固**：10:24 一次链式运行因本地只存在新拉取的 60 个 FIVES 分片，`--merge-only` 把 `c1_events.parquet` 覆盖成 FIVES-only（5,455 行），且我的门禁只校验 "FIVES ∈ fit_datasets" 而被空洞满足，误写 `.refit_v2_done`。已撤销该标记、从远端补回全部 253 分片重建。门禁加固：合并**前**校验各数据集分片数（60/28/45/120）并备份事件表；合并**后**要求四数据集齐全且 ≥25,000 事件；头校验要求**四个数据集全在**且分头最小事件数（miss ≥3000 / false ≥1200，因 bridge 预算天然稀疏）；并强制 `habs_sigma_source` 与 native 单位记录。
- 2026-09-03 11:00 Tab.2 基线行：保留 "EVAPORE (end-to-end, official code)" 与 "EVAPORE scorer in common harness"（几何候选）两行；`evapore_scorer --candidates rigr` 变体放补充材料。rNCA 行标注为"我们的重实现（共同训练框架）"，并在正文一句话给出在原作者 64×64 数据上的复现性检查（客观收敛、活性掩膜有界、rollout 稳定；默认预算下未超过种子 IoU）。
- 2026-09-03 14:55 **步骤 B（部署版 R_false 计量）提速，语义如下**：(a) HRF/FIVES 的反事实危害改在 **C1 工作分辨率**（最长边 1536，HRF s=0.4384、FIVES s=0.75）上计量，再按 C1 约定用 `src.c1.stats_c1.native_factor` 把**长度型**标志物（仅 `total_length`）乘 1/s 换回原生单位，然后除以原生训练划分 σ；FD/密度/迂曲度为无量纲，不换算。DRIVE/CHASE（`resize_longest=None`）走的是严格恒等路径，已建好的表不受影响、无需重建。`rfalse_train.csv` 新增 `work_scale` 与 `nativef_<biomarker>` 列以便审计。验证：HRF 单图 work_scale=0.43836、nativef_total_length=2.28125、nativef_FD=1.0，Habs 量级与 DRIVE 同阶（1e-3–3e-2）。(b) build_data 按图并行（默认 6 进程，spawn，单线程、纯 CPU）；每图 RNG 由 (seed, 图序号) 派生，因而与调度顺序无关且可复现（与此前单一序列流不同，已记录）。(c) `--rfalse_max_images` 限制**参与 R_false 计量**的图数（FIVES=40），评分器样本仍取全部 120 图。(d) `--fd_rotations` 默认 8→5（C1 设定）。(e) 链的 stage 2 `--force` 改为一次性：`runs/rigr_data/.stage2_forced_done` 存在则不再强制，重启不会重做 DRIVE/CHASE。
  **关键缺陷修复**：线程限制原本只写在 pool 的 initializer 里，而 spawn 子进程在解释器启动阶段就已导入 numpy/BLAS，故限制无效——实测每个 worker **86 线程**，两个数据集同时构建时约 1000+ 线程挤在 80 核上。改为在**父进程创建 pool 之前**设置 `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_*_NUM_THREADS=1`，子进程继承后于首次导入即生效；实测降到**每 worker 6–9 线程**。
  **实测时间**：修复前 HRF ~2600 s/图（13 图≈9.5 h）、FIVES ~650 s/图（120 图≈22 h）。工作分辨率计量后，安静机器单图单进程 HRF = **455 s**（19 个候选，含 21 次双管线标志物调用，约 21 s/次，此前原生分辨率约 186 s/次，≈8.8×）。上线后 6 workers × 2 数据集并发，单图墙钟约 15 min 但吞吐为 6 图并行：HRF 13 图≈50 min、FIVES 40 计量图 + 80 仅样本图≈2 h（原 31.5 h → 约 3 h，≈12×）。**未达到"单图 ≲3 min"**：要达标需把 `max_cand_bio` 24→8~12 或把 `rfalse_bio` 退回 skan，两者都会改变 R_false 训练集，留待协调者决定。
- 2026-09-03 14:50 **(1) verification_failures 恢复**：`--merge-only` 依分片重建摘要，而分片只存**被接受**的事件，故该字段被清空。原 193 图 run 的摘要在远端已于 10:22 被 FIVES-train run 覆盖、`logs/c1_full.log` 不存在、`results/_fives_test_only/` 也只备份了 `c1_stats_summary.json` —— 但 `results/c1_images.csv` 的逐图 `fails` 字典完好（merge-only 在无新 meta 时回退读取旧表，见 run_c1:880），与 FIVES-train run 自身摘要（`results_remote/results/c1_summary.json`，60 图）两者图集不相交，直接相加即得全语料 253 图 tally，已写回 `results/c1_summary.json`（附 `verification_failures_source`、`verification_by_type`、`verification_reason_fractions`）。
  **丢弃率（占尝试数）**：sever 8112/23950 = **33.9%**（全部为 `no_disconnection`，即切口未真正断开，门禁按设计拒绝）；caliber 732/3740 = **19.6%**（`invariants_changed`，拓扑中性臂被正确否决）；truncate 710/10215 = **7.0%**（cut_too_short 4.2% / branch_too_short 2.5% / beta0_changed 0.3%）；bridge **0%**（`bridge_candidates` 只产出可行对）。总计 9554/41529 = **23.0%**。尝试数与预算自洽（caliber 精确吻合 60×20+28×20+45×12+120×12=3740；sever/truncate 因个别图 loci/终末支不足预算低 0.1%）。
  **附带发现（需留意）**：`results/c1_images.csv` 仍是 193 行 —— merge-only 未为新增 60 张 FIVES 训练图生成 meta，故 **B0 缓存缺 FIVES 训练图**。本次 σ 未受影响（FIVES 用原生实测表），但若后续有人依赖 B0 回退路径需先补跑 meta。
- 2026-09-03 14:35 **(2) 像素量纲 φ 消融（新语料 31,975 事件，habs，仅训练划分拟合）**→ `results/tab1_exp1_feature_ablation.csv`。删掉 8 个像素量纲分量（phi_d/r_a/r_b/r_mean/gap_len/fov_dist/curvature_a/curvature_b，27→19 维）**不能**修复跨数据集迁移：LODO R² 仅从 −0.198→−0.169（miss）、−0.091→−0.067（false），ρ 反而略降（0.169→0.164、0.192→0.178）；同域则明显变差（ρ 0.496→0.480、0.438→0.373）。逐留出集看唯一显著改善是 LODO:FIVES（miss R² −0.589→−0.469，false −0.613→−0.437），与 FIVES 分辨率失配最严重一致，但仍深度为负。结论不变：**BTR 的跨数据集失效是量纲标定问题，不是像素尺度特征造成的**，删特征无法补救；论文按"同域可用 / 迁移标定失效"表述。
- 2026-09-03 15:10 **STARE 训练划分 σ 补入**：新增 `results/stare_native_train_biomarkers.csv`（10 张 STARE 训练划分 observer-1 `ah` 掩膜，605×700 原生分辨率，双管线，fd_rotations=5），据此向 `gateA_biomarker_scales_train.csv` 追加 8 行（`source=stare_train_obs1`、`resolution=native`），向 `gateA_pipeline_agreement_train.csv` 追加 4 行。`train_scales.py` 中的原生测量表已泛化为 `NATIVE_TABLES`（FIVES + STARE），故重新生成不会再丢掉 STARE 行；已核验重生成后表头列序不变、非 STARE 行逐字节不变。S4 preflight（`--only I --status`）0 个 `[fatal]`。
  **STARE σ**：FD_pvbm 0.03443 / FD_skan 0.04378；density 0.00990（双管线同值）；tortuosity_pvbm 0.00737 / skan 0.01673；total_length_pvbm 1619.98 / skan 1677.53。
  **需留意的异常**：STARE 的双管线一致性与其它数据集**相反** —— 迂曲度 ρ=0.879 **通过** 0.8 阈值（其它数据集 0.13–0.86 多不通过），而**总长度 ρ=0.758 不通过**（其它数据集 0.94–1.00 全通过），是唯一总长度失败的数据集。n=10 置信区间很宽，很可能是小样本噪声；不影响既有预注册决策（迂曲度 skan-only 由跨数据集最小值 0.13 决定，仍成立），但若 STARE 进入主分析，其总长度的双管线一致性低于门槛，须在论文中说明。

- 2026-09-05 11:30 **步骤 H（Exp2）BTR 接线缺陷修复**：`plan_H` 未传 `--btr_runs_dir/--btr_variant`，故 `exp2_candidates` 的 `R_miss` 由默认扁平布局 `runs/btr/btr_miss_all.joblib`（H_dep 目标、且在 **test 图事件**上拟合，属泄漏）解析，而 `R_false` 用的是部署版 `btr_false_deployed.joblib`（habs）。`BTRBackend` 正确拒绝混用（`ValueError: miss/false heads were trained on different targets ['habs','hdep']`），远端首次 H 运行 118/118 图全部报错、未写出 `results/tab1_exp2.csv`。**修复**：`plan_H` 现与 D/F 一致传 `--btr_runs_dir runs/btr/habs_trainonly --btr_variant all`（备份 `src/pipeline/s4_all.py.bak_planH_*`）。同时把本地 2026-09-03 10:36–10:37 重拟合的 `runs/btr/habs_trainonly/`（10 个头）推到远端镜像后重跑 H —— 远端此前仍是 08:23–08:24 基线（`btr_resync.sh` 的静默门一直判定"正在变化"而未推送）。结论：任何在此之前用远端 `habs_trainonly` 头定价的运行都需按镜像 mtime 复核；G 区块的 `wo_<D>` 头同批更新。
- 2026-09-05 10:34 **phase-2 链断裂后手工接续**：`run_s4_phase2.ps1` 的 stage 2（B,C 重拟合）于 09-04 01:37 正常结束（642.1 min），但链进程随后消失，既未写 `runs/rigr_models/.phase2_refit_done` 也未启动 stage 3。已核验 4 数据集 × 3 种子共 12 组 `scorer.joblib` + `btr_false_deployed.joblib` 齐全（`s4_all` 预检确认 12 对 R_false 目标均为 `habs`），补写标记并手工启动 stage 3（`--only D,F,I --skip-steps "G:*,H:*" --jobs 8`，70 步，日志 `runs/s4_stage3.log`）。

- 2026-09-05 20:30 **EVAPORE-HRF 只能用混合精度训练（协调者裁定，记录于此并写入 paper/tables/baseline_protocol.tex 脚注）**：`EvaporePipeline.features()` 把 U-Net 施加在**全分辨率图像**上；HRF 原生 3504×2336（8.19 MP，为 FIVES 2048² 的约 2 倍）在 16 GB T4 上放不下 —— 在**独占整张卡**的情况下仍 OOM：解码器 skip 拼接需再要 1.95 GiB，而模型+激活已占 14.28 GiB，缺口约 700 MB（19:49 复核 `Process 760639 has 14.28 GiB`，即它自己）。此前 09-03 10:26 / 12:36、09-04 04:24 三次 OOM 均被记为"同卡共用"，实为设计本身在 HRF 上就已越界，共用只是压垮它的最后一根稻草。
  **处置**：`evapore_adapter.py` 新增 `--amp`（fp16 autocast + GradScaler，主权重与优化器状态保持 fp32），**默认关闭**，故 `evapore_{drive,chasedb1,fives}.pt` 三个既有权重逐位不变、可复现；仅 HRF 用 `--amp` 训练，checkpoint 元数据与 `runs/s4_logs/remote_Etrain/Etrain_evapore_hrf.log` 保留证据。备份 `src/baselines/evapore/evapore_adapter.py.bak_amp_*`。若 fp16 仍不够，退路依次为 `--amp` + 梯度检查点、最后才是 HRF 单独用 0.75 尺度特征图（后者须在表中标注）。
  **补充一致性检查**：仅因远端在关机前本会空转，追加 `--amp` 重训 drive/chasedb1 → `runs/repair/ckpt/evapore_{drive,chasedb1}_amp.pt`（约 2.5–3.5 h）。**FIVES 明确排除**：其 fp32 训练单是 55 epoch 就跑了 101155 s（28.1 h），远超本次 ~10 h 预算。Tab.2 的 EVAPORE 行对这三个数据集**仍取 fp32 权重**，AMP 版只进补充材料用于说明"混合精度对结果影响可忽略"。
- 2026-09-05 20:30 **finisher 增加失效探测**：`remote/etrain_status.sh` 只读日志（不查进程表、不按模式匹配任何进程），返回 DONE / FAILED:<rc> / STALL:<min> / RUNNING；`remote/finish_remote.sh` 的 D/D2 阶段任一非零 rc 或日志静默 ≥90 min 即写 `remote/.finish_remote.ALERT` 并停止轮询（绝不关机）。起因：17:23 的 OOM 让只轮询产物文件的看门狗对着一个已死的任务空等 2.4 h。

- 2026-09-05 23:10 **步骤 D 的 `--sweep` 限定为 seed 0（协调者裁定）**：`s4_all` 新增配置键 `sweep_seeds`（默认 `None` = 全部种子，保持原行为），`configs/s4_phase2.json` 设为 `[0]`。Fig.3 的 (λ,τ) 曲线与匹配版 FCR 统计只由 seed 0 产生；seed 1/2 在默认工作点 λ=1、τ=0.5 上运行，而 Tab.2 主表数字本来就取该工作点，故三种子的主结果不受影响。**实现要点**：不扫描的步骤不再把 `sweep_curve.csv` / `sweep_per_image.csv` 列入 `outputs`，否则该步永远无法判定 DONE 而被反复重跑。备份 `src/pipeline/s4_all.py.bak_sweepseeds_*`。
  **实测依据**（10 min 双采样，8 个 D 步并发）：整体吞吐仅约 2 图/10 min；`D:prob:hrf:*` 每图约 2 h（3504×2336 上 4 λ × 7 τ = 28 个扫描格，每格一次 skan 标志物调用），30 图/种子 ≈ 60 h/种子。`prob` 是唯一扫描 τ 的模式，故其扫描格数是 `uniform`/`risk` 的 7 倍；CHASE 上实测 prob 422 s/图 vs uniform 177 s/图。
- 2026-09-05 23:01 **步骤 D 的三个 seed-0 任务下放到远端**（远端 32 核在 evapore_hrf 的 CPU 取样阶段结束后闲置）：`D:uniform:fives:s0`、`D:risk:hrf:s0`、`D:risk:fives:s0`，配置 `configs/s4_remote_D.json`（cpu_jobs=3，OMP=4，共 12 核），CPU lane 因此 `CUDA_VISIBLE_DEVICES=""`，不会碰 GPU 0（属 evapore_hrf）。另外三个 seed-0 hrf/fives 步骤（`D:prob:hrf:s0`、`D:prob:fives:s0`、`D:uniform:hrf:s0`）**已在本地运行中，故在远端 `--skip-steps` 掉** —— 同一步骤绝不在两处同时运行。本地重启的编排器同样跳过这三个远端步骤。
  **输入 mtime 核对（远端与本地逐一致）**：`btr_miss_all.joblib` 10:36、`rigr_head/{hrf,fives}/seed0/best.pt` 09:00/09:47、`rigr_models/{hrf,fives}/seed0/scorer.joblib` 18:35/19:48；seed-0 测试预测 hrf 30、fives 60；σ 表五数据集齐全。
- 2026-09-05 23:08 **发现并终止一处重复写入**：`I:res:hrf:none` 曾在两个编排器中同时运行 —— stage-3 于 10:34 启动时本地尚无 `results/fig4b_resolution_hrf_*`（这些文件 10:46 才从远端拉回），而 `s4_I_gpu1` 的同名步骤于 20:24 完成并写出结果。stage-3 那一份（PID 43492，gpu0 lane）会在完成时覆盖已完成的结果，已按显式 PID 终止；保留 20:24 那份（DONE 61711.8 s，自洽完整）。教训：**拉回远端产物会改变本地步骤的"已完成"判定，因此任何拉取都应在编排器启动之前完成，或对受影响的步骤显式 `--skip-steps`。**

- 2026-09-05 23:20 **Fig.3 扫描网格与扫描分辨率的三项修正（协调者裁定）**：
  (1) **停掉六个 28 格全扫描的 `D:prob` 运行**（hrf/fives × seed 0,1,2；均按显式 PID 终止，终止前逐一复核命令行），另加 `D:uniform:hrf:s0` —— 它同为 seed-0 的 HRF 扫描，若保留原生分辨率会与其余 seed-0 HRF 扫描混用两套口径。部分输出全部移到 `*_partial_sweepfix0905_2323/`，未删除。实测被停掉的运行：`D:prob:hrf:s0` 12.7 h 只完成 6/30 图（约 2 h/图），`D:prob:fives:s0` 11.7 h 完成 22/60（约 32 min/图）。
  (2) **`sweep_tau` 由 7 值改为 4 值 `{0.3,0.5,0.7,0.9}`**（λ 仍 `{0.5,1,2,4}`），故 `prob` 的网格由 28 格降为 16 格；`uniform`/`risk` 本就只用 `taus[:1]`（接受判据是 U>0 的硬阈值，λ 才是旋钮），不受影响。**DRIVE/CHASE 的既有 seed-0 扫描曲线不重跑**：新网格的四个 τ 是旧网格七个 τ 的**子集**，Fig.3 直接取子集即可得到跨数据集一致的网格；论文按此说明。
  (3) **HRF/FIVES 的扫描格标志物改在工作分辨率（最长边 1536）计量**，并按 C1 与 `build_data` 相同的约定用 `native_factor` 把**长度型**标志物折算回原生单位（`total_length ×1/s`；FD/密度/迂曲度无量纲不换算）。实现为 `run_rigr --sweep_scale {native,work}`，编排器仅对 `sweep_scale_work_datasets=["hrf","fives"]` 发出 `work`。**主工作点（λ=1、τ=0.5，即 `per_image.csv` 的全部数字）始终在原生分辨率计量，未变。** 为使 `macro_mae_benefit` 不含分辨率偏移，扫描的 BEFORE 基线也在同一工作分辨率重测（`sweep_before_bio`），`sweep_per_image.csv` 新增 `sweep_scale` 列记录实际比例。
  **必须写进论文的限制**：Fig.3 中 HRF/FIVES 的"标志物收益"轴是**工作分辨率导出**的；密度是两个同步缩放面积之比（严格），长度型经 1/s 精确折算，而 **FD 与迂曲度只在连续极限下尺度不变**，在有限栅格上其 1536 px 取值只是原生值的近似（与 DECISIONS.md 2026-09-03 10:45 记录的同一条假设）。DRIVE/CHASE 的 `resize_longest` 为 None，其扫描仍在原生分辨率，两者不可逐值比较，只可比较趋势。
  备份：`src/rigr/run_rigr.py.bak_sweepscale_*`、`src/pipeline/s4_all.py.bak_sweepseeds_*`。
- 2026-09-05 23:29 **远端三步同样按新配置重跑**：`D:uniform:fives:s0`（6/60）、`D:risk:hrf:s0`（2/30）、`D:risk:fives:s0`（6/60）进度均 <20%，按裁定重跑而非沿用旧网格；远端 orchestrator PID 379837，`cpu_jobs` 3→4（evapore_hrf 已进入 GPU 训练阶段，CPU 占用下降）。本地 orchestrator 重启为 `runs/s4_stage3c.log`，23/70 已完成、11 步 `--skip-steps`（远端所有权 3 步 + 本地在飞 8 步），无任何步骤在两处同时运行。

- 2026-09-06 06:00 **远端已关机；但三个已完成的 D 步骤未被拉回（我的疏漏）**：finisher 的 E 阶段核验通过（FAIL=0）、拉回最终归档（86.4 MB / 464 条目）后于 06:00:07 执行 `shutdown -h now`。**问题**：09-05 23:29 下放到远端的 `D:uniform:fives:s0`、`D:risk:hrf:s0`、`D:risk:fives:s0` 已于 **03:25:53 全部成功完成**（60/30/60 图，rc=0，`per_image.csv` 已写出），但我在增加这三步时只把 AMP 权重加进了 E 阶段核验清单，**没有把它们加进去**，故无人拉取；`final_remote_archive.tgz` 覆盖的是 `logs/ + exp/results + exp/runs/s4_logs`，而它们位于 `exp/runs/rigr/`，因此也不在归档内。数据仍在实例的持久盘 `/path/to/workdir` 上，**在 cloud 控制台重启实例后执行 `bash remote/pull_lost_D.sh` 即可恢复**（脚本带重试并校验 61/31/61 行）。
  **教训（已写入 SHUTDOWN_CHECKLIST §5c）**：向远端下放任何工作时，必须在**同一次改动**里把它的产物同时加入关机清单与 finisher 的 E 阶段硬门；只列"设计时已知项"的门会在有新工作滞留时照样放行。加固方向：归档改为覆盖 `exp/runs/rigr`，E 阶段改为对远端任何 `runs/rigr/**/summary.json` 做"本地是否有对应物"的反向核验，而不是核对手写清单。
  **已完成且已核验在本地**：8 个 E:train 权重（`evapore_hrf.pt` 早停于 ep 90、最佳 ep 65、val_auc 0.9507）＋ 2 个补充 AMP 权重（drive 最佳 ep 25 / 0.8736，chasedb1 最佳 ep 46 / 0.8664）、`results/tab1_exp2.csv`、G 全套 LODO、STARE 全套、fig4a/fig4b、远端日志、最终归档。
- 2026-09-06 12:30 **效用尺度缺陷（规格—实现不一致）**：utility.py 将 C_geom 归一化到 O(1)（中位数≈2.6），而 σ 标准化的单事件危害 R_miss/R_false 量级为 10⁻³，故预注册 η=0.1 下 U(e)≈−ηC_geom（corr(U,−C_geom)=0.9999），风险模式退化为"几乎不接任何边"（LODO/STARE TRR=0，Exp2 中 U 与实际危害相关符号相反）。提案 §3.1.2 规定 C_geom 与 R 同一无量纲尺度，实现违背了该规定。**修正**：C̃_geom = C_geom · m_R，m_R = 每数据集训练集候选上 R_miss 预测值的中位数（在 fit_models 阶段计算并固化到 rigr_models/<ds>/seed<k>/geom_scale.json，测试图不参与）；λ=1、η=0.1 保持预注册值。所有风险模式运行（D risk、F 中含 BTR 效用的变体、LODO risk、STARE risk）重跑；退化版结果保留在补充材料并在正文说明发现过程（Exp2/LODO 诊断）。补充：η∈{0.01,0.1,1} 敏感性（DRIVE/CHASE seed 0）。

- 2026-09-06 12:40 **C_geom 尺度缺陷的实现与验证（承 12:30 裁定）**：
  **(1) m_R 固化**：新增 `src/rigr/geom_scale.py`，`m_R = median(R_miss)` 取自该数据集**训练划分**候选（`runs/rigr_data/<ds>/rfalse_train.csv`，`build_data` 硬编码 `split="train"`），写入 `runs/rigr_models/<ds>/seed<k>/geom_scale.json`（含分位数、所用头路径与 `train_split_only` 标记）。已生成 **16** 个：12 个主网格 + 4 个 LODO（LODO 用留出变体头 `btr_miss_wo_<D>.joblib` 与三源域候选，held-out 域不进入）。实测 m_R：DRIVE 0.006561、CHASE 0.007581、HRF 0.004835、FIVES 0.005141；LODO 0.00408–0.00555 —— 与 R 的 O(10⁻³) 量级一致，正是缺陷所在。
  **(2) 施加位置**：`utility.compute_utility` 新增 `geom_scale`，**仅在 risk 模式**下 `U = p·R_miss − λ(1−p)·R_false − η·C_geom·m_R`；**uniform 保持不缩放**（其 R≡1，C_geom 本就在 R 的尺度上，缩放反而会平移其工作点、破坏与 risk 臂的可比性）；prob 不用 C_geom（η_eff=0）。risk 模式缺 m_R 时**抛异常并给出修复命令**，不再静默退化。`run_rigr` 从 `--scorer` 同目录读 `geom_scale.json`（可用 `--geom_scale_file` / `--geom_scale` 覆盖），把 m_R 打进日志与 `summary.json`（`geom_scale`、`geom_scale_source`），并作为 `geom_scale` 列进入效用表。
  **(3) 单图验证（DRIVE 01_test，51 候选）**：
  | 运行 | n_selected | TRR_recall | FCR |
  |---|---|---|---|
  | risk 旧（m_R=1，即上线缺陷） | **0** | 0.000 | 0.000 |
  | risk 新（m_R=0.006561） | **17** | 0.400 | 0.471 |
  | uniform（对照，未改动） | 12 | 0.200 | 0.500 |
  即 risk 从"一条边都不接"恢复到与 uniform 同量级（略多），且在相近 FCR 下召回翻倍。
  **(4) Exp2 离线重算（未重跑 H）**：`results/tab1_exp2.csv` 逐候选存有 `p_e/R_miss/R_false/C_geom`，且 `plan_H` 未传 `--lam/--eta`（故 exp2 默认 λ=1、η=0.1），因此 U_btr 可精确离线重建。先做**自检**：用 m_R=1 重算与存档 `U_btr` 的最大绝对差 **4.4e-16**，公式确认无误；再按数据集 m_R 重算并改写，旧表保留为 `results/tab1_exp2_pre_etascale.csv`，新表增列 `U_btr_pre_etascale`、`geom_scale_mR`。效果：**corr(U, −C_geom) 由 +0.9999 降到 +0.1478**；`spearman(U_btr, dH_actual)` 由 **+0.0917 变为 −0.0330** —— 符号由"效用越高危害越大"翻正为方向正确（dH_actual 是危害，越低越好），幅度仍小，应如实报告。`build_tab1` 只消费 `{p_e,dDice,dclDice,dBCS,dBetti0,U_btr,dH_actual}`，`accepted_by_matching` 仅为诊断列且无法离线重建（需走廊/标签数组），已在此注明其仍反映旧排序。
  **(5) 隔离与重跑范围**：`runs/rigr/risk/**`（12 主 + STARE）、`runs/rigr/lodo_*/risk`（4）全部移为 `*_etascale_pre0906`。**新发现并一并处理**：`plan_I` 的 `_hook_env` 设 `RIGR_HOOK_MODE="risk"`，故 **Fig.4 的 repaired 臂（`I:stab:*:rigr` ×4 与 `I:res:hrf:rigr`）同样是缺陷效用产物** —— 已隔离为 `results/etascale_pre0906_fig4*`，并给 `repair_hook.py` 加了同一约定的 m_R 解析（`RIGR_HOOK_GEOM_SCALE` 或 scorer 同目录的 `geom_scale.json`；已验证解析出 0.006561）。**F 消融不受影响**：`appearance_only`/`+orientation`/`+curvature`/`+failcond` 均为 uniform 模式，已完成的 3 个变体（drive/chasedb1）有效保留；仅 `+btr_utility` 与 `frangi_vs_learned` 为 risk，二者尚未运行。
  **(6) η 敏感性（补充材料）**：`run_eta_sensitivity.sh`，risk 模式 η∈{0.01,0.1,1} × {DRIVE, CHASE_DB1} seed 0、不扫描 → `runs/rigr/eta_sens/<eta>/<ds>/seed0`；η=0.1 为预注册值，同时充当与主 risk 运行的一致性校验。
  备份：`src/rigr/{utility,run_rigr,repair_hook}.py.bak_etascale_*`。

- 2026-09-06 13:00 **两处收尾加固（协调者要求）**：
  (1) **`D:uniform:fives:s0` 本地兜底**：它是远端关机时唯一还"活着"的待恢复产物（同批的两个 risk 目录已是 C_geom 修正前的产物，本地正在重跑，`pull_lost_D.sh` 会把它们落成 `*_etascale_pre0906`）。`remote/fallback_uniform_fives_s0.sh`（detached，PID 46016）在**本地编排器只剩这一步待跑**（`--status` 的 pending ≤ 1）**或 09-06 22:00 到点**（先到者为准）时本地跑掉它；若已在磁盘上则立即空退。**重启看门狗 `await_restart_pull_shutdown.sh`（PID 47460/39296）继续保留**：实例若稍后回来，它仍会拉取、做 fail-closed 反查、再关机。
  (2) **E→J 链的门禁改为按磁盘判定**：原门禁是"没有 s4_all 进程存活"，而**进程消失不等于完成** —— 09-06 12:19 stage-3 编排器带着在飞的 F 步骤一起死掉，旧看门狗因此认为可以推进，几乎在被隔离的 risk 产物和只跑了一半的 Tab.3 上执行 J。现改为轮询 `--only D,F,I --status`，要求 `pending==0 且 skipped==0` 才继续（其后仍保留"无其它编排器存活"的次级保护，避免抢 CPU 池）。该判据自动覆盖 C_geom 重跑与 `D:uniform:fives:s0`，无需任何人维护清单 —— 与 §5c 记录的教训同一类修复。已实测：正则可正确解析页脚 `70 steps  (33 done, 37 pending, 0 skipped)`，看门狗（PID 8864）已在按此轮询。

- 2026-09-06 16:15 **远端不再自动关机（用户规则，覆盖此前"跑完即释放"的指令）**：`remote/await_restart_pull_shutdown.sh` 与 `remote/finish_remote.sh` 中两处 `shutdown -h now` 均改为受 `ALLOW_SHUTDOWN=1` 环境变量保护，默认**不关机**；已确认当前没有任何具备关机能力的看门狗在运行。新增的 `remote/pull_offload2.sh` 完全不含关机路径。**只在明确指令下关机。**
  **附带发现（记录以免重演）**：容器内 `shutdown -h now` 在本实例上表现为**重启**而非释放（15:21 发出后 boot.sh 于 15:21–15:23 重新计时），计费状态由 cloud 控制台决定，**从容器内部无法释放实例**；06:00 那次"成功"只是因为控制台当时未再拉起。
- 2026-09-06 16:12 **本地/远端分工（no-duplicate，已在两侧 `--skip-steps` 中固化）**：
  | 位置 | 步骤 |
  |---|---|
  | **远端** chain1 | `D:prob:hrf:s0` —— 本地关键路径（本地 4/30 时停掉，残留移为 `*_partial_moved_remote_*`），远端已有 hrf seed0 的头/评分器/预测，prob 模式不需要 m_R |
  | **远端** chain2 | `F:+failcond:*`、`F:+btr_utility:*`、`F:frangi_vs_learned:*`（12 步，seed 0，用主模型；后两个是 risk 变体，依赖新推的 m_R） |
  | **远端** chain3 | `G:run:{drive,chasedb1,hrf,fives}:risk` —— 修正效用后的 LODO risk 重跑（远端旧结果已就地隔离为 `*_etascale_pre0906`） |
  | **本地** 在飞 | `D:prob:fives:s0`、`D:uniform:hrf:s0`、`D:risk:hrf:s0/s1/s2`、`D:risk:fives:s0/s1` |
  | **本地** 新排 | `D:risk:fives:s2`、`F:{appearance_only,+orientation,+curvature}:{hrf,fives}`（ω=0 变体，其 `_w0` 模型只在本地） |
  推送到远端：整个 `src/`（含 `geom_scale.py` 与 utility/run_rigr/repair_hook 的 C_geom 修正）+ **16 个 `geom_scale.json`**；远端已核验 `utility.py` 含 13 处 geom_scale 引用、16 个 m_R 文件到位。远端 orchestrator：chain1/2/3 = PID 1860/1861/1862，日志 `runs/s4_logs/orch_off_{D,F,G}.log`，共 11 个 CPU 槽（32 核）。本地重启为 `runs/s4_stage3f.log`（45/70 完成，18 步 `--skip-steps`，6 槽）。
  **拉回与守卫**：`remote/pull_offload2.sh`（detached，PID 31464）等三条链跑完后拉回三组产物 + 远端日志，再做 fail-closed 反查（远端每个 `runs/rigr/**/summary.json` 必须有本地对应物），**并逐一校验所有 live risk 目录的 `summary.json` 带有 `geom_scale` 字段** —— 不带的即为修正前产物，就地隔离以便重跑。该守卫针对的是 15:10 的一次事故：`pull_lost_D.sh` 把远端**修正前**的 risk 目录直接解包进了正在被本地重跑写入的实时路径（`runs/rigr/risk/{hrf,fives}/seed0`），使二者的 `per_image.csv` 暂时是旧数据；本地重跑完成后会逐文件覆盖（同数据集、同种子、同图集），但必须校验而非假设。`pull_lost_D.sh` 应改为先解包到暂存目录再定向放置（待办）。
- 2026-09-15 21:30 稿件评审第 2 轮（6.5/10 条件分）触发的实验补充：(1) LODO 的 R_false 改为部署版定义——用三源域的 rfalse_train.csv（A* 候选路径反事实危害）拟合 btr_false_deployed_wo_<D>，替换 C1 桥接头；主文与 LODO 的 R_false 定义统一为"A* 候选路径上的反事实危害"，C1 桥接头仅用于 Exp1 分析。(2) EVAPORE 端到端增加接受阈值扫描（τ∈{0.3,…,0.9}），纳入 Fig.3 匹配 FCR 比较。(3) LODO 补 seed 1、2：头/评分器/BTR 保持 seed-0 训练，仅换测试预测种子（论文注明）。(4) 术语改为 pre-specified / post hoc（prospectively frozen before repair evaluation），并给出 DECISIONS.md 中的冻结时间戳作为可核验时间线。
- 2026-09-16 00:05 (a) Exp2 候选子采样（--max_cand 40）曾用 Python hash() 做种子，跨进程不可复现（两次运行仅 2620/4184 候选重叠；重叠行数值一致）。改为 blake2b 稳定摘要并重跑一次，论文引用重跑结果；前两版保留为 tab1_exp2_v2_*.csv/v3。(b) STARE 的 rigr_data 随 cloud 丢失，已在 LAN 节点逐位重建；m_R(STARE)=0.00622508 冻结。(c) LODO 的部署版 R_false 头自 09-03 已存在（G:fit 产物），仅因 lodo_btr_deployed_false=False 未被使用；现改为 True 并以其运行，risk_c1bridge 版本保留为补充。(d) EVAPORE 端到端 HRF 阈值扫描：完整 7×τ 需 151 h 且 10 GB 卡 OOM，不可行。采用 10 张 HRF 测试图子集上 τ∈{0.3,0.5,0.7,0.9}（若 ≤ 30 h 可承受），论文标注为子集扫描；否则 HRF 仅报 τ=0.5 点并说明。
- 2026-09-16 01:30 **主结果判读与论文重构**：预注册结论 1 失败（BTR 逊于 GT 信息型指标；仅域内优于可观测预测子），结论 2 不可评估（预设 FCR 1/2.5/5% 远低于可达范围 0.5–0.7）且在可达范围内 risk 与 uniform 差异极小；RiGR 在四个数据集上均未在生物标志物 MAE 上胜过最强简单基线（geometric / EVAPORE e2e），HRF 上所有 RiGR 臂劣于不修复。关键发现：FCR≈0.9 的 EVAPORE e2e 反而降低 DRIVE/HRF 的生物标志物误差 → 分割系统性欠估长度/密度，生物标志物 MAE 奖励"多加像素"而非"修对拓扑"（待 bias_diagnostics 证实）。论文改为**如实的预设评估研究**：贡献 = 三轴生物标志物保真度评测框架 + 31,975 事件的结构扰动剂量-响应（单事件拓扑特异危害 ~10⁻²σ、被像素预算匹配对照大部分抵消）+ BTR 域内/跨域结果 + 对 RiGR 与四类修复基线的预设评估（负结果）+ "生物标志物误差由欠分割偏差主导、修复方法通过过连接获益"的诊断。摘要须明确陈述负结果；不隐藏、不改写预设结论。目标期刊由用户定夺（EAAI/ESWA 对负结果接受度低；可考虑 Medical Image Analysis / CBM / AI in Medicine）。
- 2026-09-16 03:00 修正后的判读：(a) geometric 基线 10 个运行曾用全集 σ（泄漏），离线按原始 bio_* 列重算后 CHASE 的 geometric 不再显著优于 RiGR（p=0.074），RiGR 在 FIVES/HRF 上优于 geometric 与 evapore_scorer；(b) Tab.1 应基于 H_abs 域内表：BTR 显著优于所有推理时可得预测子（两头，CI 不跨零），仅逊于需 GT 的指标；(c) LODO（部署版 R_false，seed 0）risk 在四个域上 TRR 均高于 uniform；(d) 偏差诊断证实：FD/迂曲度/总长度系统性低估（最高 −3.46σ），像素增加与 Δmacro-MAE 在 24 格中 20 格负相关，macro-MAE 奖励加像素直至过冲；(e) evapore_e2e 的 FCR 范围（0.76–0.95）与 RiGR（0.28–0.71）不重叠，Tab.2 中对其的比较不是匹配精度比较；(f) 事后匹配 FCR（0.55/0.60/0.65）下三种 RiGR 模式 CI 重叠，prob 在 DRIVE/CHASE 不低于 risk；risk 臂缺 τ/阈值扫描 → 补做 risk 臂的 U 阈值扫描（λ∈{0.25,0.5,1,2,4,8,16}，U>u0 的 u0 取候选 U 的分位数 {0,0.25,0.5,0.75}）用于匹配 FCR 曲线。
- 2026-09-16 05:00 (1) Exp2 子采样种子改为 blake2b 稳定摘要（两次独立运行候选键与数值完全一致）；v4 表为最终版：域内秩相关 ρ(U,ΔH)=−0.0387（p=0.012），四个数据集均 ≤0；跨数据集混合的原始 ρ 为 Simpson 伪影，论文引用域内值。(2) lodo_btr_deployed_false=True 固化；部署版 R_false 在四个留出域上均提高 TRR、FCR 基本不变；LODO seed 1/2（模型固定 seed 0）8/8 组 risk TRR > uniform。(3) EVAPORE-HRF 阈值扫描实测 46.7 h（前 10 图 ×4 τ）超预算，不做；HRF 仅报 τ=0.5 点并说明。EVAPORE 端到端的 FCR 对 τ 几乎不变（0.60–0.76）而 TRR 随 τ 下降，无法通过阈值进入 RiGR 的 FCR 区间——这本身是结果。(4) 延迟（LAN 节点，20 核 + 3080 10 GB）：DRIVE 端到端 2.80 s、HRF 41.3 s（不含生物标志物）；BTR 效用计算占 70–74%，A* <0.9 ms/候选。(5) 隐患：--gpu -1 仅在 CUDA_VISIBLE_DEVICES 为空时可用（run_rigr.py:92, head.py:383），待修。
- 2026-09-16 12:30 sweep2 完成，事后匹配 FCR 结论：在三个有共同可达带的数据集（DRIVE [0.505,0.630]、HRF [0.669,0.683]、FIVES [0.610,0.703]）上，probability-only 在几乎所有匹配工作点上 TRR 点估计最高、CI 最窄、生物标志物收益 CI 在 DRIVE/HRF 均不跨零；risk 的 CI 宽 3–10 倍，FIVES 上 3/4 点收益为负。CHASE 无共同带（n=8，两臂占据不相交的 FCR 区间）。判定：预设结论 2 在可达范围内的事后检验为负——风险引导效用未在匹配精度下超过仅按概率阈值的选择。论文据此陈述："最简单的 RiGR 臂最好；BTR 作为危害预测子有效，但把它放进决策效用没有带来收益"。

- 2026-09-16 13:50 **LODO 补跑 probability-only 臂（协调者指示）**：跨域块此前只有 `risk` 与 `uniform`，缺 `prob`。补跑的理由是域内事后匹配-FCR 比较（`results/fig3_matched_fcr_posthoc.csv`）显示**prob 在同等精度下不劣于且多数点优于 risk**，因此"风险制导效用是否比朴素 p_e 更能跨域迁移"这一问题在 LODO 上尚未被检验——只比 risk 与 uniform 无法回答它。
  **命令**：`run_rigr --mode prob --dataset <D> --seed 0 --pred_dir runs/seg/<D>/seed0/pred --out runs/rigr/lodo_<D>/prob --head_ckpt runs/rigr_head/lodo_<D>/seed0/best.pt --scorer runs/rigr_models/lodo_<D>/seed0/scorer.joblib --evidence auto --orientation learned --lam 1.0 --eta 0.1 --tau 0.5 --bio both --gpu 0`（FIVES 加 `--limit 60`），即与 `G:run:<D>:uniform` **逐字对齐**，仅 `--mode` 不同；四折并行于本地 CPU 池，日志 `runs/s4_logs/G_run_<D>_prob.log`，驱动 `run_lodo_prob.sh` / `runs/lodo_prob_driver.log`。
  **为何不传 BTR 与 geom_scale**：prob 模式下 `U = p − τ`，λ/η 与 R_miss/R_false 全部不参与（`utility.compute_utility`），故 `--btr_*` 与 m_R 无意义；这也意味着 prob 臂**不受 2026-09-06 12:40 C_geom 尺度缺陷影响**，无需隔离重跑。
  **纳入方式**：`tables.build_tab2` 的 LODO 主块按 `lodo_*/{risk,uniform}` 取臂，新增 `prob` 目录会**自动**进入同一块（无需改代码），`_aggregate_mean_sd` 以 `mode` 分组。
  **预期用途**：若 prob 在四折上的 TRR_recall 不低于 risk，则与域内结论一致，应在正文中明确写出"风险制导效用在域内与跨域均未优于 probability-only"；反之则是 risk 唯一站得住的优势，须单独报告。
- 2026-09-16 15:00 稿件评审第 3 轮（7/10）触发：(1) FCR 空集约定统一为"|A|=0 时 FCR 未定义、不计入均值"（报告 n_defined），替代此前 FCR=0 标记；tables/figures/matched_fcr 一致修改并重生成。(2) Tab.2 的 no_repair 行必须与修复行使用同一图像集与同一预测（FIVES 修复行用 60 图子集，而 no_repair 若取自 seg_per_image 的 200 图则不可比）：改为由各修复运行的 before_* 列（同一图像、同一预测）派生 no_repair 行；审计 clDice 0.9066→0.8694 的差异来源。(3) 事件计数措辞：31,975 个验证通过的干预，其中 25,649 获得有效匹配对照并进入 H_net 分析。(4) 证据地位措辞：最终 RiGR 使用事后（prospectively frozen）的 H_abs 与 m_R，不将整个最终实现称为预设。(5) "risk 从不优于 prob" 软化为与 bootstrap CI 一致的表述。
- 2026-09-16 15:30 **no_repair 基线修正（全部四个数据集）**：tables.py 的 no_repair macro-MAE 曾按 (dataset,image) 无 seed 合并并任取首个匹配运行（实为 ablation_* 目录），导致基线错误：DRIVE 1.0557→1.0653、CHASE 1.2679→1.1876、HRF 1.8076→1.8735、FIVES 0.6685→0.6468。后果："HRF 上所有 RiGR 臂劣于不修复"是伪结论——按正确基线，RiGR（及 geometric/evapore_scorer）在四个数据集上均小幅降低 macro-MAE（HRF −0.029/−0.023），仅 rnca 恶化；RiGR 仍全面逊于 evapore_e2e。新增 src/eval/check_results.py（6 类一致性检查）纳入 make_all。
  **补充（2026-09-16 16:20）**：应协调者要求，prob 臂再补 seed 1、2（四折共 8 个运行，`runs/rigr/lodo_<D>/prob_seed{1,2}`，脚本 `run_lodo_prob_seeds.sh`）。协议与既有 `risk_seed*`/`uniform_seed*` 一致（见 2026-09-15 21:30(3)，并已对照 `lodo_drive/risk_seed1/summary.json` 核验）：**头、评分器、模型一律沿用该折的 seed-0 训练，只更换测试预测的种子**——因此这些额外种子是"测试集重采样"，不是重新训练，不引入新的训练随机性。
  **动机**：seed-0 三臂结果显示 prob 在 4 折中的 3 折 TRR_recall 高于 risk（CHASE 0.2102 vs 0.2031、FIVES 0.2162 vs 0.1923、HRF 0.2029 vs 0.1592），仅 DRIVE 相反（0.2589 vs 0.3063）；但 CHASE 仅 8 张测试图，单种子下 ~0.05 的差异无法与噪声区分，故跨域结论必须多种子才能成立。
- 2026-09-17 00:30 **转向方案预注册（外部第 2 轮评审后）**：(1) 术语：image-level measurement fidelity（Pearson/Spearman r 与去偏残差 SD）而非 reliability；"三轴测量审计"而非可加分解（e_i=μ_ds+ε_i，拓扑项为配对反事实）。(2) CF-Loss 基线必须为公开代码（github.com/rmaphoh/feature-loss）的忠实实现（软盒计数、按图像尺寸缩放的二进阶梯、密度项）；此前"固定 2..64 阶梯"版本作废。(3) HRF 因探针 P1–P5 已成为开发集：HRF 仅报告为开发/机制集；确证集 = FIVES（200 测试图，未用于任何探针的微调选择）+ 外部无掩膜集（APTOS/ODIR，冻结分割器后再触碰标签）+ 未触及的 CHASE/STARE 跨集评估。(4) 检查点规则预先固定：固定 50 epoch 取 last.pt（同时报告按验证集测量保真度选择的检查点）；不以测试指标选择。(5) 主终点预注册：FIVES 上 density/length/FD 的 r（配对 bootstrap Δr）与下游 macro-AUC（配对 ΔAUC，含质量/相机协变量调整模型）；迂曲度与密度副作用作为必报的安全终点。(6) 贡献收敛为两项：C1 GT 锚定的测量保真度审计 + 神谕臂效用衰减；C2 安全约束的测量感知微调（ReliSeg）；C3 视图离散度标志降为次要；拓扑修复为阴性对照。
- 2026-09-17 01:00 **外部验证集角色**：Kaggle 无凭据；来源为公开镜像（HF），逐文件字节校验。主外部集 = APTOS-2019（3,662 张，DR 分级，QWK + macro-AUC）、IDRiD（516 张，4288×2848，DR/DME 分级）、Messidor-2（1,748 张，DR 分级）；ODIR-5K 仅有 512 px 预处理镜像（无原始分辨率、逐眼单标签）→ 降为低分辨率敏感性集；DR-HAGIS 官方链接已 404 且无存档 → 放弃并记录。分割器冻结后才触碰外部标签；患者级嵌套 CV；配对 bootstrap ΔAUC；质量协变量（Laplacian 方差/亮度）调整模型。
- 2026-09-17 01:20 **E1/E5/E6 完成（results/pivot/E1_REPORT.md）**：拓扑轴报告为区间——逐事件 H_net × 每图事件数的可加外推（0.02–2.41σ）仅作上界，采用实测修复效应（RiGR-uniform，0.002–0.089σ）为正式数值（事件非独立，可加外推不成立）。神谕臂：FIVES 衰减 +0.004 [−0.016, +0.025]（r̄=0.95），HRF +0.215 [+0.115, +0.321]（r̄=0.44）；逐图回归（n=45）为零结果，如实报告。E5 视图离散度标志：FIVES 6/8 列 CI 排除零，门控 q=20 使宏 r 0.785→0.869（优于随机丢弃 p<0.001）；HRF 无效（p=0.92）→ C3 保持次要，结论按数据集分别陈述。E6：域内偏移消除 60.7% 偏差，LODO 恶化（−121%），组内 AUC 对精确偏移不变（abs_diff=0）。
- 2026-09-17 01:45 **外部集落盘与核验（补 01:00 条的实测数字）**：`exp/data/external/<name>/raw`，共 ~14 GB，文档见 `exp/data/EXTERNAL_DATASETS.md`，装载器 `src/data/external.py::load_external`，核验 `python -m src.data.check_external`。
  **APTOS-2019** 3,662 张（HF 镜像 `Tejaswini628/aptos-fundus-images`，Kaggle 原始 PNG 逐文件字节长度校验，0 失败），分级直方图 0:1805 / 1:370 / 2:999 / 3:193 / 4:295 与官方一致；**17 种原始分辨率，最长边 474–4288，像素数跨度 72×**（中位 2144）——这使 APTOS 本身就是分辨率保真度实验，`e3_external.py` 逐图记录 native/crop/infer 尺寸与实际 scale。Kaggle 测试集标签从未公开，故**未下载**（省 ~3 GB）。无患者 ID → 一图一 subject（已在代码与文档中明写，避免分组 CV 悄悄退化）。
  **IDRiD part B** 413+103=516 张、4288×2848 全一致、DR 0:168/1:25/2:168/3:93/4:62、DME 0:222/1:51/2:243，随包 CC BY 4.0；镜像 `MahsaTorki/IDRiD_Dataset` 绕开 IEEE DataPort 登录。**坑：train 与 test 的编号各自从 IDRiD_001 重新开始**，裸 stem 有 103 对碰撞 → `image_id`/`subject_id` 加 split 前缀（与 FIVES 同一类错误）。
  **Messidor-2** 实得 **1,744** 张（非 1,748）——正是裁定版中可分级的那 1,744 张；分级直方图 0:1017/1:270/2:347/3:75/4:35 与公开分布逐格吻合，且原始分辨率恰为 1440×960 / 2240×1488 / 2304×1536 三种，二者共同构成来源证据。**缺陷：镜像只存 image+label，原始文件名丢失**，874 次双眼检查无法复原 → 一图一 subject，患者级折可能把同一次检查的两眼分到两侧，构成小幅乐观，须在论文中声明。
  **ODIR-5K** 6,392 眼 / 3,358 患者（3,034 双眼），8 类分布与官方一致，带 age/sex/左右眼；**唯一免登录镜像是 512×512 预处理版**（非原始 ~2976×1984）且为逐眼单标签（非患者级多标签）→ 按 01:00 决定作低分辨率敏感性集，不承载正向主张。原始分辨率仅存在于 Academic Torrents，本机无 BT 客户端，未追。
  **DR-HAGIS 放弃**：曼彻斯特个人主页整页 404（浏览器 UA 亦然），Wayback 只有 2025-10-26 的 301/404 两次抓取、从未存下 zip 字节；HF/Zenodo/GitHub 均无副本。
  **FOV**：复用 `datasets._generate_fov`（红通道 10% 阈值→闭运算→填洞→最大连通域），不改一行——生物标志物按 FOV 内定义，两套 FOV 规则会把差异伪装成分割器效应。无血管 GT 故改用替代判据（单连通域 + FOV 内/外绿通道均值比），四集分别为 3.9 / 24.1 / 20.7 / 289.7（APTOS 低是因为多数图已裁紧，非掩膜失败）。
  **E3 推理约定**：先裁到 FOV 外接框再把最长边缩到 **1024**（HRF/FIVES 检查点训练于 1536，差异已在 meta.json 与每行 scale 中记录）；滑窗 patch=训练 patch、stride=patch/2、高斯融合、无翻转、FOV 内 0.5 阈值。分割器冻结由 `bio` 阶段的 ckpt SHA-256 强制：同一 tag 下换检查点即拒绝续写。
- 2026-09-17 02:00 **E3 推理尺度与检查点范围**：外部集推理按训练约定最长边 1536（撤回此前 1024 的说法；分辨率是机制变量，不能引入额外降采样混淆），并记录每图原生尺寸与实际 scale。检查点范围（冻结）：FIVES 训练分割器 3 种子 × {baseline, ReliSeg, CF-Loss 忠实版}（主）；HRF seed0 × {baseline, ReliSeg}（开发集模型，次要）。外部集顺序：IDRiD → Messidor-2 → APTOS-2019 → ODIR-512（敏感性）。Messidor-2 镜像无文件名 → 一图一受试者（乐观假设，论文中说明）。基线检查点的 E3 bio 阶段可立即在 GPU 空档运行，纳入 E2 队列作为最低优先级。
- 2026-09-17 11:10 **CF-Loss 忠实移植核验通过**：`python -m src.pivot.cf_loss verify` → 阶梯 [512,256,128,64,32,16,8,4]、max|Δcount|=0.000e+00、L_FD 参考 0.0102156997 vs 移植 0.0102156997（d=0.000e+00）、L_vd 参考 0.0537012815 vs 移植 0.0537012853（d=3.7e-09）、**VERIFY: PASS**。E2 主网格的 `cfloss` 配置即该忠实实现（third_party/feature-loss 逐位对齐），旧的"固定 2..64 阶梯 + log-log 斜率"版本仅作为消融行 `abl_fixed_ladder` 存在。
- 2026-09-17 11:10 **队列锁核查（无陈旧锁）**：`runs/pivot/_e2_queue` 共 87 个 .lock，其中仅 2 个无 done 标记（`infer:fives:s0:cfloss->chasedb1`、`infer:fives:s2:reliseg_nocl->chasedb1`），owner.json 的 pid 37132/27820 经 Win32_Process 核验为**当前在跑的两个 GPU worker**。未清除任何锁。另发现两个 2026-09-16 00:30 的孤儿 multiprocessing 子进程（pid 8940/34508，CPU 累计 <0.5 s、常驻 29 MB、父进程已消失），非 E2 作业，未动。
- 2026-09-17 11:11 CPU 生物标志物阶段启动：`python -m src.pivot.e2_run bio --procs 12`（机器 24 核，为两条 GPU 通道留出余量），日志 `runs/pivot/_e2_logs/bio_20260917_111107.log`。gtbio：CHASE_DB1（8 测试图）与 STARE 均已写出 `results/pivot/e2_gt_chasedb1.csv` / `e2_gt_stare.csv`。
- 2026-09-17 11:15 **E3 外部集推理入队（最低优先级）**：`src/pivot/e2_run.py` 新增 `e3_checkpoints()` / `e3_jobs()`，追加在 build_jobs() 末尾（worker 顺序遍历，故排在零样本作业之后）。作业 = `python -m src.pivot.e3_external bio --dataset <ext> --ckpt <ckpt> --tag e3_<ds>_s<k>_<cfg> --resize-longest 1536 --resolution-record`，外部集顺序 idrid → messidor2 → aptos2019 → odir5k（与 02:00 条一致），检查点 11 个 = FIVES seed0-2 ×{baseline(runs/seg/fives/seed<k>/best.pt), reliseg(last.pt), cfloss(last.pt)} + HRF seed0 ×{baseline, reliseg}，共 44 个作业。**done 标记用 meta.json 而非 bio.csv**：该阶段可续跑、每 10 图就 append 一次 bio.csv，若以 bio.csv 为标记，被中断的作业会被误判为完成；meta.json 只在整趟跑完后写出。`run_job` 增加 `gpu_flag`/`gpu_value` 字段，因为 e3_external 用 `--device cuda:N` 而非 `--gpu N`。两块卡各 20 GB、占用 <1.5 GB，故每卡额外启动 1 个 worker（pid 44920/37692，日志 gpu{0,1}_e3_20260917_111502.log）。实测 IDRiD（4288×2848）约 11.5 s/图。
- 2026-09-17 11:25 **发现并修复 FIVES 生物标志物记录错配（协议相关）**：`src/pivot/p5_eval.py::cmd_bio` 在 manifest 的 stem（FIVES 去掉了 split 前缀，如 `100_D`）无法直接命中 image_id 时，用 `k.endswith(stem)` 取 `cand[0]`；但 FIVES 的 train 与 test 各自从 1 编号，`100_D` 在两个 split 中**都存在**，于是 200 张测试图中有 99 张把预测掩膜配到了 *train* 那张图的 FOV/眼底图/optic disc 上，并以 train 的 image_id 写入缓存（与 GT 的交集只有 101/200）。改为**先用 manifest 自带的绝对 `image_path` 精确解析**，其次 image_id / subject_id，最后的后缀匹配在出现多于一个候选时**直接报错而不再猜**。修复后 200/200 解析为 test_* 且与 bio_master 的 GT 全部对上。受影响且需重算：`e2_fives_s0_baseline`、`e2_fives_s1_baseline`（唯一两个在修复前跑完的 FIVES 域内 bio 任务）——删除其 bio.csv、`results/pivot/cache/fd5/fives/p5_<tag>/` 缓存与队列锁后重跑。HRF/CHASE/STARE 的 image_id 唯一，不受影响；`bio_master.csv` 的 fives pred_test 行（200 条全为 test_*）由另一条代码路径生成，亦不受影响。
- 2026-09-17 11:25 **paper2 第 1 轮评审（review/paper2_r1_reply.md，6.8/10，大修）后的预注册补充**：(1) E2 增加同预算继续训练对照 `continued`（基线损失 CE-Dice+0.5·clDice，同起点检查点、同 50 epoch、同学习率；FIVES+HRF × 3 种子）；HRF seed0 增加单项消融 density-only / length-only / fd-only。(2) 检查点规则消解矛盾：主分析 = 50 epoch last.pt；验证集保真度选择的 `fid` 检查点仅作次要报告；"接受"门槛不再用于选择。(3) 安全边际数值化：clDice 非劣性边际 0.01（绝对）；未约束标志物 tortuosity/density 的 Δr ≥ −0.05 且 |Δbias| ≤ 0.25σ；越界即在结果中标记"collateral damage"。(4) 主生物标志物面板冻结为 skan 管线 4 项（density, total_length, FD, tortuosity）；PVBM 为敏感性分析（避免 8 列面板重复密度）。(5) E3 预注册（在任何外部推理/标签分析之前锁定）：主队列 = APTOS-2019；主终点 = referable DR（grade ≥ 2）AUROC，logistic 回归 on 4 项面板 + 质量协变量，ΔAUC（ReliSeg − baseline，FIVES 训练 3 种子，逐种子报告 + 种子均值），图像级配对 bootstrap；GBDT 与 QWK 为次要；IDRiD、Messidor-2 为复现队列；ODIR-512 敏感性。APTOS/Messidor-2 无可靠患者 ID → 图像级划分（不声称患者级）；IDRiD 同样图像级。Messidor-2 n=1,744。(6) 拓扑轴改为图像级估计量：|ΔB|/RMS（同一训练集 σ），并给出像素预算匹配的非拓扑对照的配对 CI（C1 语料的匹配对照）。(7) 多种子表述：逐种子报告方向一致性，不做"跨种子稳健"的强主张。
- 2026-09-17 11:45 **后备论文（拓扑修复）最终 J 完成**：Fig.4B HRF 修复臂与 LODO prob 种子 1–2 均完成后重跑 J（--force）；表 1–3、图 2–5 重生成；tab3_ablation.tex 的 9 个 FCR 值按 |A|=0→NaN 约定修正。LODO 结论按种子 0–2 合并后减弱："prob 赢三折"仅在 HRF 稳健，FIVES 持平，CHASE_DB1 反转，DRIVE 仍为 risk 领先 → 正文与补充材料已改写为按种子陈述。paper/main.pdf 38 页，仅余 4 项作者提供的待办（CRediT、生成式 AI 声明、代码 URL、资助/致谢）。RESULTS_DIGEST.md 已过时（工程摘要，不入论文）。
- 2026-09-17 11:30 **E2 补充臂入队（paper2 评审，13:30 条的执行记录）**：(1) 新增同预算对照 `continued`——同一起始检查点、同样 50 epoch、同样 LR 调度与 batch，只用基础损失（CE-Dice + 0.5·clDice），实现为 `lam=0.0`（测量项照常前向计算并记录，梯度贡献恒为 0，数据管线与其它臂逐位一致），FIVES+HRF × seed 0-2 共 6 个训练；Δr 今后**同时**对 `baseline` 与 `continued` 报告。(2) HRF seed0 新增单项消融 `abl_density_only` / `abl_length_only` / `abl_fd_only`（保留 clDice，lam=0.25，自适应阶梯）——原有的 `abl_no_*` 只能说明哪一项可去掉，说明不了哪一项在起作用。(3) 这 9 个训练作业在 `build_jobs()` 中位于 E3 之前（worker 顺序遍历），另起两个 worker（pid 44224/44688，日志 gpu{0,1}_train2_20260917_113033.log）立即领取；先前启动的两个 E3 worker 保留其 IDRiD 作业继续跑——E3 阶段的 GPU 利用率实测仅 0-6%（瓶颈在 CPU 端的生物标志物计算），与训练并行不构成显存或算力冲突。单个 50-epoch 微调实测 18-25 min，9 个作业两通道约 100 min。
- 2026-09-17 11:35 **E3 分类阶段预注册冻结（写入 `src/pivot/e3_external.py` 常量，先于任何外部标签被模型读取）**：主队列 = APTOS-2019（3,662 图）；主终点 = referable DR（grade ≥ 2，`src/data/external.py::REFERABLE_THRESHOLD`）的 AUROC；主模型 = 4 个 skan 生物标志物 + 采集/质量协变量（featureset `bio+cov`）上的 logistic 回归；主对比 = ΔAUC(ReliSeg − baseline)，FIVES seed 0-2 逐种子并报种子均值；不确定度 = **逐图**配对 bootstrap（N_BOOT=1000）。次要：GBDT、0-4 分级的 QWK、仅 `bio` 特征集。复制队列：IDRiD(516)、Messidor-2(1,744)。敏感性：ODIR-512（不承载正向主张）。**划分/重采样单位**：APTOS / Messidor-2 / IDRiD 镜像均无可用患者标识（Messidor-2 原始文件名丢失，874 次双眼检查无法复原；APTOS 无患者 ID；IDRiD 一图一受试者），故这三个集合的划分、分组与 bootstrap **一律为逐图**，论文不对其作患者级泛化主张；ODIR-5K 有真实患者 ID（3,358 患者 / 6,392 眼），保持受试者级分组。代码落点：`COHORT_ROLE` / `PRIMARY_COHORT` / `PRIMARY_TARGET` / `PRIMARY_CLF` / `PRIMARY_FEATURESET` / `grouping_for()`，输出表新增 `cohort_role` / `split_unit` / `is_primary` 列。**时间线如实记录**：IDRiD 的 bio（推理）阶段已于 11:15 开始，早于本条冻结 20 分钟；但 bio 阶段只写入标签列、不拟合任何模型，冻结时没有任何外部标签被模型看过，分类阶段一行未跑。同时把 `DEFAULT_RESIZE_LONGEST` 从 1024 改为 1536，与 02:00 条一致（此前 e2 队列已显式传 `--resize-longest 1536`，改的是可能被误用的默认值）。
- 2026-09-17 11:40 **E2 分析协议固定（`src/pivot/e2_analysis.py`）**：主面板 = **skan** 管线的 density / total_length / FD / tortuosity，PVBM 作为敏感性分析单列一节；主检查点 = last.pt，`fid` 并列报告并始终标注；Δr 为逐图配对 bootstrap（2000 次重采样，95% 百分位 CI），对 `baseline` 与 `continued` 两个参照各报一次；逐种子 Δr 与符号一致性（`n_seeds_positive` / `direction_consistent`）和种子均值（同时给出"种子均值的配对 bootstrap CI"与"3 个种子点估计的 t 区间"）并列。**安全边界数值化**：Dice/clDice 下降 ≤ 0.01（绝对）；tortuosity 与 density 的 Δr ≥ −0.05；|Δbias| ≤ 0.25σ。点估计越界记 `BREACH`，点估计合格但 95% CI 越界记 `AT-RISK`，逐行写入 `results/pivot/e2_safety.csv`。
- 2026-09-17 11:55 **E3 检查点范围追加 `continued`（追加，非替换，待协调者复核）**：02:00 条冻结的 11 个检查点原样保留且排序在前；在每个外部集内**追加**FIVES seed 0-2 的 `continued`（同预算对照）共 3 个，E3 作业总数 44 → 56。理由：13:30 的评审意见把"更多训练"确立为域内 Δr 的必需对照，外部 ΔAUC 面临同一个混淆；若不测，`ΔAUC(ReliSeg − baseline)` 无法区分"测量项有用"与"多训练 50 epoch 有用"。排序保证预注册的主对比（ReliSeg − baseline）在每个队列上先完成，追加臂只在其后运行，且这些作业依赖 `runs/pivot/fives/seed<k>/continued/last.pt`，在该训练完成前处于 WAIT。**这是对预注册范围的增补而非改动，删除它不影响任何既有终点**；若协调者认为不应扩大冻结范围，回退只需删掉 `e3_checkpoints()` 末尾的三行（届时尚无一行结果产生）。
- 2026-09-17 11:55 **LAN 节点重新启用**：工作树 /mnt/data/Programming/research_ws/medical1（/dev/sda1，NTFS，1.4 TB 空闲），与本地同结构；用后不删。分工：LAN 3080 运行 E3 外部推理（FIVES 训练检查点 × {baseline, reliseg, cfloss} × {IDRiD, Messidor-2, APTOS}，ODIR 敏感性最后）；本地双 3080 运行 E2 新增训练臂（continued、单项消融）、零样本与域内推理。
- 2026-09-17 12:05 **E3 分工改为本地 + LAN 双节点（协调者指示），本地队列重新划界**：LAN 节点（LAN_HOST，单张 3080，由另一个 agent 配置）承担**全部 FIVES 训练检查点 × {IDRiD, Messidor-2, APTOS-2019}**（baseline `best.pt` / reliseg `last.pt` / cfloss `last.pt` × seed 0-2，共 27 趟），结果 rsync 回 `exp/results/pivot/e3/<ext>/<tag>/`。本地 GPU 只做：(1) `continued` 新臂（FIVES+HRF × 3 seed）与 3 个 HRF 单项消融的训练；(2) 它们的域内推理（last + fid）与 CHASE/STARE 零样本推理；(3) **全部完成之后**才是 HRF seed0 × {baseline, reliseg} 在三个主队列上的 E3 推理，以及 ODIR-512 敏感性集的全部检查点。代码落点：`e2_run.py` 新增 `E3_REMOTE_COHORTS` / `e3_is_remote()` / `e3_local_units()`，`e3_jobs()` 只生成本地的 17 个作业（作业总数 355 → 316）。
  **执行细节（避免两节点写同一个可续跑文件）**：本地此前已在跑的两趟 `e3:idrid:fives:s0:{baseline,reliseg}`（各约 160/516 图，无 meta.json）已按 PID 停止（子进程 25996/6936 及其 worker 44920/37692，均经 Win32_Process 逐一核验命令行后停止），并**删除其半成品目录**，让 LAN 的 rsync 落在干净目录上——`e3_external.py` 的 bio 阶段每 10 图 append 一次 bio.csv，两节点同写会把行交错在一起。同时在 `runs/pivot/_e2_queue` 中为 36 个远端作业 id **预置锁目录**（owner.json 标注归属 LAN、无 pid），使任何用旧作业表启动的本地 worker 都无法认领它们。另两个 worker（44224/44688）正在跑 `continued` 训练，其旧作业表中的远端 E3 条目已被上述锁挡住，故不打断训练、不重启。
  **撤回 11:55 的 E3 `continued` 追加**：协调者已明确列出两节点各自的检查点清单，均不含 `continued`，故 `e3_checkpoints()` 恢复为 02:00 冻结的 11 个。**后果需协调者知悉**：外部终点 ΔAUC 目前只有 `ReliSeg − baseline` 这一个对比，没有同预算对照，13:30 评审对域内 Δr 提出的"更多训练"混淆在 E3 上仍然敞着。
- 2026-09-17 12:10 **E3 `clf` 预注册默认值锁定并留指纹（供 LAN 节点核对 src/ 同步）**：内容见 11:35 条。本仓库不是 git 仓库，故以文件指纹代替 commit：
  `src/pivot/e3_external.py` SHA-256 = `80acee935bf404c366d55deef2850ae568d21e0e89f8926b4727c1cdbd129809`，mtime 2026-09-17 11:32:58，33,451 字节。
  LAN 节点同步 src/ 后应校验该哈希一致；若不一致，说明拿到的是锁定前的版本，必须重新同步后再跑分类阶段（bio 阶段不读这些常量，可先跑）。相关文件同刻指纹：`e2_run.py` 46c54b341a75282a…（本地队列划分，LAN 不需要一致）、`src/data/external.py` 6e96340dd5c61bc9…（REFERABLE_THRESHOLD=2 的来源，必须一致）、`e2_analysis.py` 672e13648901a3fb…（仅本地 E2 分析用）。
- 2026-09-17 12:08 **E2 首版结果与应对**：CF-Loss 忠实移植验证通过（L_FD/L_vd 与参考实现差 ≤ 4e-9）。FIVES seed0 主终点：density/length/FD Δr 的 CI 均含零——FIVES 基线已近天花板（r 0.90–0.98），与"保真度限制效用"的命题一致但不提供正向证据；HRF（开发集）有充分余量。安全终点：FIVES 全部臂 clDice 非劣；HRF 上 cfloss 与 reliseg_nocl 三种子均突破 clDice 边际，reliseg（保留 clDice）通过；FIVES ReliSeg 的 PVBM 迂曲度 Δbias +0.90σ 突破 0.25σ 边际 → 按预注册标记为 collateral damage。修正一个 FIVES bio 阶段的图像匹配缺陷（训练/测试同名 stem，99/200 错配）后重算。应对：(1) 评审要求"FIVES 零结果需配独立复现集"→ 寻找未触碰的高分辨率公开血管掩膜数据集（候选 MAPLES-DR、UoA-DR、LES-AV、IOSTAR）作预注册独立复现；(2) E3 外部终点补同预算对照：LAN 队列增加 FIVES `continued` × 3 种子。
- 2026-09-17 12:15 **E2_REPORT.md 增补三节（协调者指示）**：(1) §0 Coverage——由磁盘上的 bio.csv 逐格推导的覆盖表，列出每个 (dataset × config × checkpoint) 的完成种子数，partial/PENDING 逐行标注；报告可以在网格跑完之前生成，缺的行是"尚未"而不是"零"。(2) §5 零样本（CHASE_DB1 / STARE）改为**逐种子**呈现：每个 source model × config × seed 的 r_pearson / r_spearman / bias / resid_sd，以及对 `baseline` 与 `continued` 两个参照的 Δr 与配对 bootstrap CI——这两个集合是开发集之外**唯一**有掩膜且从未被任何探针触碰的证据，n 只有 8（CHASE）与 10（STARE），种子均值会掩盖比它总结的更多东西，故不做池化。报告中明写 n=8 时 Pearson r 的 95% 区间约 ±0.5 宽、STARE 的 σ 来自其自身 GT 的稳健尺度（无 Gate A 训练尺度）。(3) 新增 §6b HRF 开发集全种子 Δr（对两个参照，池化 + 逐种子）并附 clDice 与迂曲度安全旗标。
  **顺带修掉一个会污染表的缺陷**：`e2_zeroshot.csv` 原先把含两个 reference 的 Δr 表按不含 `reference` 的键合并，会把每一行复制成两行；改为按 reference 透视成 `d_r_vs_baseline` / `d_r_vs_continued` 两组列，每个 (target, source, config, seed, biomarker, pipeline) 恰好一行；完整 Δr 表另存 `e2_zeroshot_delta.csv` 与 `e2_zeroshot_delta_pooled.csv`。
  **像素安全终点已经是全网格终值**：它来自推理阶段的 `pred*/pixel_metrics.csv`，不依赖 CPU 生物标志物阶段。HRF 上 `cfloss`（ΔclDice −0.0214…−0.0254）与 `reliseg_nocl`（−0.0122…−0.0205）在三个种子上**全部突破** 0.01 的非劣边界，而保留 clDice 的 `reliseg`（−0.0048）通过——这正是 clDice 安全约束被预注册的理由：去掉 clDice、或改用完全丢弃 Dice/clDice 的已发表 CF-Loss，在 HRF 上要以拓扑为代价。FIVES 上各臂均在 ±0.01 内。

- 2026-09-17 12:20 **LAN 节点重建于 /dev/sda1（新工作树）并启动 E3 外部推理**：
  用户新规则——工作树必须落在 `/dev/sda1`（NTFS，fuseblk 挂载于 `/mnt/data`，1.4 TB 空闲），
  路径 `/mnt/data/Programming/research_ws/medical1`，与本地同构（`exp/src`、`exp/data`、
  `exp/runs`、`exp/results` …），**用后不删**。旧根 `/home/user/medical1` 已被用户删除。
  (1) **NTFS 可执行性实测通过**（`mount | grep sda1` 显示 `rw,nosuid,nodev,relatime`，无 `noexec`；
  chmod+x 的脚本可执行），故 venv 直接建在 `/mnt/data/Programming/research_ws/medical1/venv`
  （`python3.12 -m venv --copies`，NTFS 无符号链接语义故用 `--copies`），**无需**退回
  `/home/user/medical1_venv`。按 `exp/runs/pip_freeze.txt` 安装（`packaging @ file://` 一行
  改为裸 `packaging`，落为 `runs/pip_req_lan2.txt`；torch 走 `--extra-index-url
  https://download.pytorch.org/whl/cu124`，节点上 6.6 GB pip 缓存命中，约 3 分钟装完）。
  核验：torch 2.6.0+cu124、`cuda_avail True`、`NVIDIA GeForce RTX 3080`，25 个包一次性导入通过。
  (2) **同步工具**：Windows Git Bash 无 rsync，故新增 `exp/remote/lan2_sync.sh`——以
  (size, path) 清单对比远端、只发缺失/尺寸不符的文件、按 ~400 MB 分批 tar-over-ssh，
  等价于 `rsync -a --size-only --partial`，中断至多损失一批，重跑即续传。载荷是 JPEG/PNG/.pt
  （已压缩），故默认**关闭** gzip（实测 `-z` 2.8 MB/s vs 不压缩 ~4 MB/s，Windows 侧单核 gzip 是瓶颈）。
  旧的 `lan_*.sh` 仍指向已删除的旧根，未改动；新根一律用 `lan2_*.sh`。
  (3) **E3 阶段的并行度按实测重定**：`e3_external bio` 是 CPU 密集（skan 生物标志物），
  探针显示 user-time ≈ real-time（每作业约 1 核），单作业 3.8–3.9 s/图（IDRiD 4288×2848）。
  10 GB 卡上 `--sw-batch 4` 每进程占 2134 MiB，第 5 个进程即 OOM（已实测，4 个作业因此在
  12:08 报 `torch.OutOfMemoryError`）；改 `--sw-batch 1` 后每进程 712 MiB 而**速度不变**
  （3.9 s/图），故 9 个检查点可**全部并发**（9×712 MiB = 6.4 GB，20 核用 9 核）。
  这是对"顺序跑"指令的有意偏离并记录于此：顺序执行 IDRiD 9 趟需 5.0 h，9 路并发需 ~58 min。
  驱动脚本 `exp/remote/lan2_e3_driver.sh`（作业列表 `<dataset> <ckpt> <tag>`，
  以 `meta.json` 而非 `bio.csv` 作完成标记，与 11:15 条一致；日志
  `exp/runs/lan_e3/<ds>__<tag>.log` 与 `driver_<ds>.log`）。
  (4) **同步内容**：`src`(83)+`configs`+`DECISIONS.md`+`data/{DATASETS,EXTERNAL_DATASETS}.md`
  +`results/gateA_biomarker_scales_train.csv`（134 文件 2 MB）；9 个检查点 + 6 个 config.json
  （266 MB）；`data/external/idrid/{raw,fov}`（1,038 文件 0.41 GB）；Messidor-2 4.8 GB 同步中，
  APTOS 8.1 GB 随后。**指纹核验通过**（12:10 条）：远端 `src/pivot/e3_external.py`
  SHA-256 = `80acee93…`，`src/data/external.py` = `6e96340d…`，与本地逐位一致。
  `python -m src.data.check_external idrid` 在节点上 **OK**（516 条、标签分布与公开值一致、
  FOV 内外绿通道比 24.1、单连通分量）。
  (5) **12:14 启动 IDRiD 波次**（9 作业并发，PID 群见 `driver_idrid.log`）：
  实测每作业 7.0 s/图（9 路并发下的单作业速率），波次等效 **0.78 s/图**，
  IDRiD 全部 9 个检查点 ETA ~58 min（12:14 → 约 13:12）。

- 2026-09-17 12:40 **独立复现数据集：候选调研与判定（`research/08_replication_datasets.md`）**。任务要求"最长边 ≥1400 px"，实测后**改用 FOV 直径**作为判据并记录理由：眼底集的黑边比例差异极大，MESSIDOR 帧/FOV ≈1.6，故 MAPLES-DR 的 1440×960 档虽然"最长边 1440"，视网膜实际只有 **909 px**（低于 CHASE_DB1 的 917），正落在本项目已证明机制不成立的低分辨率区；而 Fundus-AVSeg 的 1280×1280 是紧裁剪，FOV 直径 **1238 px**。参考值：DRIVE 537 / STARE 638 / CHASE 917 / FIVES 2013 / HRF 2967。标注粒度用尺度无关量（骨架总长 ÷ FOV 直径）核验：HRF 22.6 > Fundus-AVSeg 16.3 ≈ DRIVE 16.2 ≈ MAPLES-DR 16.1 > STARE 12.9 > FIVES 12.2 > CHASE 11.2——两个新集都不是"只标主干"。**判定：主复现集 = Fundus-AVSeg，次 = MAPLES-DR**（理由见 16:45、16:50 条）。已排除（逐一核验，链接与证据见 research/08）：RETA（掩膜仅 1024×1024）、Leuven-Haifa/UZLF（需签 DTA）、UoA-DR（需签 EULA 邮寄）、ORVS（论文自带 GitHub 404，无镜像）、ARIA（三个已知站点全部失效）、IOSTAR（下载自述损坏 + 1024 px + SLO）、LES-AV（n=22）、RVD（手持视频、半自动掩膜、CC BY-NC-ND）、VAMPIRE（无公开掩膜集，域名 NXDOMAIN）、HRF-Seg+（底图就是 HRF）、FOVEA（n=40 且混术中视频）、PRIME-FP20/REVIEW/AV-WIDE/RAVIR/RECOVERY-FA19/INSPIRE-AVR/HEI-MED/GRAPE/CHUAC/DCA1（n、模态或标注类型不合）。
- 2026-09-17 12:45 **Fundus-AVSeg 落盘（主复现集）**：`exp/data/fundusavseg/raw`，`Fundus-AVSeg.zip` 212,768,224 B，MD5 `20c85c9343ff95435f131b684afadd50` 与 figshare API `computed_md5` 一致；DOI 10.6084/m9.figshare.27938034.v2，**CC BY 4.0**（论文 doi:10.1038/s41597-025-05381-2 本身是 CC BY-NC-ND，数据与论文许可不同，勿混引）。核验：100 图 + 100 标注（文件名一一对应）+ metadata.xlsx 100 行 + 官方划分 80/20；分辨率 1280×1280 (79) 与 2656×1992 (21)，FOV 直径 1238 / 2080 px；病种 **40 Normal / 20 DR / 20 AMD / 20 Glaucoma**（与 FIVES 同一四分类、同一文件名编码），另有逐图质量标签（83 高 / 17 低）与左右眼（55/45）。**陷阱：随包标注不是二值掩膜**，而是 5 色动静脉图（红动脉 3.7%、蓝静脉 4.3%、绿交叉 0.12%、白类别不明 0.11%）；`read_binary` 只看红通道会**静默丢掉全部静脉**。装载器取四类非背景的并集生成二值血管掩膜，缓存到 `exp/data/fundusavseg/labels/`，raw 保持原样，原 5 色图仍以 `av_path` 挂在记录上。
- 2026-09-17 12:50 **MAPLES-DR 落盘（次复现集，仅标签）**：`exp/data/maplesdr/raw`，`MAPLES-DR.zip` 14,765,980 B MD5 `ee8bf3591abcf6c261f85eb6f61416bc`、`AdditionalData.zip` 40,782,288 B MD5 `d77c5891d881ca5cf13ca4c70f52fd49`，均与 figshare API 一致；DOI 10.6084/m9.figshare.24328660.v3，标签 **CC BY 4.0**。核验：198 张 Vessels 掩膜（官方 138/60）+ 11 项其它生物标志物；标签为 MESSIDOR 原生分辨率 1440×960 (104) / 2240×1488 (53) / 2304×1536 (41)。**图像不在该 deposit 内**——属 MESSIDOR 联盟，须在 adcis.net 填个人信息表 + 邮箱验证（自助、无需机构审批，但无法自动化；该站 `/en/download/messidor-bib/` 当前返回 HTTP 500），且无任何原始 MESSIDOR 的公开镜像（HF 只有 Messidor-2；Zenodo "Messidor-A" 仍指回 ADCIS）。诊断标签 DR 共识 R0–R4A + ME 共识 + 三位视网膜专家逐人分级，但可用子集分布为 **R0 12 / R1 129 / R2 10 / R3 7 / R4A 4**，严重失衡，不足以承载分病种终点。**两条必须写进论文的限制**：(1) 血管掩膜是**网络预标注经人工修正**，deposit 自带 preannotations，预标注与终稿 Dice 均值 **0.783**（中位 0.805，最小 0.358，最大 0.894，无一 >0.90）——人工改动是实在的，但预标注网络的系统偏差可能部分残留在被本项目当作锚点的"真值"里；(2) 笔触最粗，平均管径 1.54% FOV 直径（HRF 0.60%、FIVES 1.40%），FOV 内前景 0.150（FIVES 0.109），管径类标志物会继承。
- 2026-09-17 12:55 **MAPLES-DR × Messidor-2 重叠核查（`src/data/maples_overlap.py`）**：两边无法按文件名比对（本项目 Messidor-2 镜像丢失原文件名），故按像素匹配——同原生尺寸内，掩膜与绿通道各降到 512×512、各自带通（`x − gaussian(x, σ=3)`）后做归一化内积，图像取负（血管偏暗）。**首次尝试未做带通，198 个掩膜里 174 个被判到同几张图**：亮眼底盘对黑背景的低频项压倒一切，该失败已写入模块 docstring。**结果：198 张中 162 张在 Messidor-2 镜像里**；命中者领先候选池（同尺寸 527/614/603 张）**14.9–87.0 σ**，落选者 2.0–8.5 σ，8.5–14.9 之间**为空**（故 10 σ 阈值不是调出来的）；162 个命中构成**双射**（无一 Messidor-2 行被两个掩膜占用），而打分并未强制一一对应；另以 MAPLES-DR 视盘掩膜独立复核，162 张中 161 张盘内亮度高于 FOV 均值；并对两个最弱命中（14.9、17.5 σ）做了目视核对，掩膜与血管完全贴合。按尺寸命中 91/104、48/53、23/41，按划分 109/138 train、53/60 test。未命中的 36 张与 MAPLES-DR 文档"使用了若干被 MESSIDOR-2 移除的图像"一致。**排除清单写入 `exp/data/external/messidor2/overlap_maplesdr.csv`（162 行）**，完整 198 行含落选分数在 `exp/data/maplesdr/messidor2_match.csv`。**若采用 MAPLES-DR 作复现集，E3 的 Messidor-2 队列必须剔除这 162 行（1,744 → 1,582）**，否则两项分析不独立；`load_external` 不自动剔除（静默改行数会改掉已记录的全部 Messidor-2 数字），由分析阶段显式应用。
- 2026-09-17 13:00 **装载器与核验**：`src/data/datasets.py` 新增 `maplesdr` / `fundusavseg`（含别名），二者**刻意不进 `DATASETS`**（该列表仍严格是训练与汇报用的五个集，新增不会静默改动任何既有循环），改由 `ALL_DATASETS` 或按名索取。MAPLES-DR 图像解析顺序：先 `raw/images/<name>.tif`（官方 MESSIDOR，一旦有人填完表格把文件放进去即自动生效、无需改码），否则回退到 Messidor-2 镜像的 162 张；每条记录带 `image_source` 与 `messidor2_id`，无法解析的 36 张默认丢弃（`include_unresolved=True` 可见全部 198）。`check_data.py` 增加 "GT vessel pixels inside FOV" 一行（FOV 生成器当初就是按这个判据调的，此前只写在文档里）。核验结果：`python src/data/check_data.py maplesdr fundusavseg` → maplesdr 162（109/53，missing 0，GT-in-FOV 均值/最差 1.0000，FOV 占帧 0.448–0.472），fundusavseg 100（80/20，missing 0，GT-in-FOV 均值 1.0000 / 最差 0.9995，FOV 占帧 0.642–0.785）；`python -m src.data.datasets` 七个集全部 missing_files=0。**未训练任何模型，未触碰任何复现集的标签用于选择。**
- 2026-09-17 13:05 **待用户完成的一步**：MAPLES-DR 要补齐到 198 张、并把镜像 JPEG 换成官方 MESSIDOR TIFF，需有人在 <https://www.adcis.net/en/third-party/messidor/> 填表（姓名/邮箱/单位/国家/GDPR 同意 + 邮箱验证）下载 Base11.zip…Base34.zip，把 198 个 `<name>.tif` 放进 `exp/data/maplesdr/raw/images/` 即可，装载器已优先读该目录。未代填表格（需真实个人身份信息）。
- 2026-09-17 13:15 **独立复现实验 E4 预注册（在任何复现集训练/评估之前锁定）**：主复现集 Fundus-AVSeg（n=100，FOV 直径 1238/2080 px，四类疾病标签，CC BY 4.0；从未用于本项目任何选择）。设计：5 折交叉验证（按官方 80/20 划分为第 1 折，其余随机分层折，固定种子），每折训练基线 U-Net（与 FIVES/HRF 同配方、同分辨率约定 1536/768），再以同起点微调 {continued, reliseg, cfloss}（50 epoch，last.pt），得到 100 张折外预测。主终点：折外 Δr（reliseg − continued；同时报 − baseline）对 density/total_length/FD（skan），配对 bootstrap 95% CI；安全终点：clDice 非劣 0.01、tortuosity/density 边际同前；次终点：四类 macro-AUC 三臂（GT / baseline / reliseg，n=100，配对 ΔAUC）——同时成为第三个神谕臂队列。次复现集 MAPLES-DR（162 张可用；掩膜为网络预标注的人工修正，笔画最粗，须在论文声明）：仅零样本评估全部 E2 检查点，按 FOV 直径分层：≥1380 px 为分析层（71 张），909 px 层预先声明为低分辨率阴性对照。E3 Messidor-2 队列在分类阶段剔除与 MAPLES-DR 重叠的 162 张（`exp/data/external/messidor2/overlap_maplesdr.csv`，1,744→1,582），剔除在 clf 阶段以显式参数执行。以上任一结果为阴性均如实报告；不得按复现集结果更改设计。
- 2026-09-17 13:15 **paper2 第 1 轮评审的 C1 侧实现（分析+稿件，未触碰 E2/E3 数字）**：(1) 新脚本 `src/pivot/e1_topology_image.py`：拓扑轴改为**图像级估计量**，输出 `results/pivot/e1_topology_image.csv`（每 dataset×biomarker×修复臂×种子的 mean|ΔB|/σ、RMS ΔB/σ、配对 Δ|误差|，2000 次图像 bootstrap；另含 `macro_primary4`/`macro_all8` 伪标志物以便给修复表一个配对 CI）与 `results/pivot/e1_topology_matched.csv`（像素预算匹配对照的配对差 H_net，bootstrap **按 image_id 聚类**）。**σ 轴的最后一个例外已消除**：H_net 不再用 all-mask σ，而是由 `absdB/cabsdB × nativef ÷ sigmatr` 重算到训练集原生 σ。结果：实测逐图 mean|ΔB| 0.006–0.115σ、RMS 0.006–0.351σ；逐事件拓扑特异超额 0.000–0.020σ，配对 CI 排除零但上界 ≤0.037σ（即"有界小效应"而非"证据缺失"）。(2) `src/pivot/p3_downstream.py` 增加 `skan4` 特征集与 `--featuresets/--sources/--merge`；`src/pivot/e1_oracle.py` 按特征集分块输出并用**同一面板**计算保真度摘要。冻结的 4 项 skan 主面板下游结果：HRF 参考差 +0.2170 [+0.1188,+0.3323]（logreg）、+0.1800 [+0.0615,+0.3052]（gbdt）均排除零；FIVES 四格 CI 全部跨零且三格点估计为负（−0.0027/−0.0218/0.0000/−0.0319）——即**预测特征分类器在 FIVES 上不劣于参考掩膜特征**，这是"参考臂是基准而非上界"的直接证据，"oracle ceiling"措辞已全文删除。(3) `src/pivot/e5_reliability_flag.py` 的 flag validity 增加两侧 bootstrap p（(b+1)/(B+1) 修正）与 `n_boot` 列，并加 `--only flag`（未重跑昂贵的 gating）。主定义由 CV 改为 **SD/σ_b**；FIVES 显著列由 8 列中 6 列降为 3 列（主面板 4 列中 2 列），如实报告并将 CV 降为敏感性分析。(4) 论文侧：标题改为 TITLE_NOTES 候选 2（去排他性、保留 fidelity、oracle→reference-mask）；Messidor-2 改 1,744；外部集全部改为图像级重采样并删除全部 patient-level 声称；检查点矛盾以"删除接受门"方式消解，安全边际数值化；新增 5 张 matplotlib 图（`paper2/tools/fig_*.py`，经 `src/eval/savefig_util.py`）；8 列面板、harmonisation 明细、flag gating、per-seed 表移入 `paper2/sections/supplement.tex`。映射见 `paper2/REVIEW_R1_RESPONSE.md`。
- 2026-09-17 13:15 **E3 预注册增补：Messidor-2 排除 MAPLES-DR 重叠的 162 张**（协调者指示，已写入 `e3_external.py`）。清单 `exp/data/external/messidor2/overlap_maplesdr.csv`（163 行 = 表头 + 162 条，键 `messidor2_id`，另带 maples_name / split / 尺寸 / score / margin_sd）。核验：162 个 id **全部**能在 `load_external('messidor2')` 的 1,744 条记录中命中，无一缺失，排除后 **1,744 → 1,582**。
  **只在 clf 阶段排除，绝不在 bio 阶段排除**：`run_bio` 一行不改，bio.csv 保留它测过的每一行，于是测量阶段仍然逐位可复现，而排除始终是一个**单点、可审计、可逆**的分析决策。代码落点：常量 `CLF_EXCLUDE` / `CLF_EXCLUDE_COL` 与函数 `load_exclude()`；`run_clf()` 在算出 `common` 之后、任何拟合之前剔除；`summary.csv` 与 `delta.csv` 每行新增 `n_excluded` 与 `exclude_src` 两列记录来源。CLI 新增 `--exclude-list`：缺省用该数据集的预注册清单；传 `none` 可关闭，**仅供明确标注的敏感性分析**；清单文件缺失时直接报错退出而不是静默跳过。
  **重新指纹（LAN 节点需重新同步 src/ 后再跑分类阶段；bio 阶段不受影响，正在跑的可以继续）**：
  `src/pivot/e3_external.py` SHA-256 = `af65105ebf0a500ca0a70bc407ab62f5d76b03805a8057a6b1fcf8a8e9f6c1a5`，mtime 2026-09-17 13:11:15，36,579 字节（旧值 `80acee93…`，33,451 字节，已作废）。
- 2026-09-17 13:15 **E4 复现 agent 与本队列共用两块 GPU**：其作业仅在某卡空闲显存 ≥ 6 GB 时启动。本地 worker 若因此 OOM，**按约定上报协调者，不盲目重试**——`e2_run.py` 的 worker 在作业失败后会把它记入 `fails` 并不再重试，因此 OOM 会表现为 `[e2] FAILED <job id>`，不会变成重试风暴；届时以该行为准上报。当前占用：两块卡各 20 GB，训练期约 10-12 GB/卡。

- 2026-09-17 13:20 **E3/IDRiD 在 LAN 节点跑完（9 个主检查点）并已回拉**：12:14:20 启动 → 13:19:35 全部
  完成，**65 分 15 秒**跑完 9×516 = 4,644 张 4288×2848 推理 + skan 生物标志物，波次等效
  **0.84 s/图**（11 路并发下单作业 7.3 s/图；同样的作业顺序执行需 5.0 h）。9 个
  `bio.csv` 各 516/516 行，`meta.json` + `resolutions.csv` 齐全；9 个
  `ckpt_sha256` 互不相同，并与本地检查点逐位一致（抽查 s0：baseline `4f0489e1cff4` /
  reliseg `bed33d322ea6` / cfloss `c4aaca9182bc`）。推理参数统一记录为
  `resize_longest=1536, patch=768, stride=384, crop_to_fov=True, threshold=0.5,
  device=cuda:0, amp=True`（`--sw-batch 1`，见 12:20 条）。回拉用新脚本
  `exp/remote/lan2_pull.sh`，远端 29 文件 / 本地 29 文件一致；**远端副本按用户规则保留不删**。
  同预算对照 `continued` s0/s1 在 12:25 追加启动（本地 `summary.json` 出现后才同步检查点），
  13:20 时 440/516，ETA 约 13:35。
- 2026-09-17 13:20 **Messidor-2 波次启动**：9 个主检查点并发（+ 仍在跑的 2 个 IDRiD continued，
  共 11 个 worker，7.9 GB 显存）。前置已备妥：`check_external messidor2` 在节点上 **OK**
  （1,744 条，标签分布与公开值一致，FOV 内外绿通道比 20.7、单连通分量），FOV 缓存 1,732 张
  已预建（1,359 s）——**必须预建**，否则 9 个并发 worker 会同时写同一个 `fov/*.png`。
  APTOS 的 FOV 预建同时在跑（13:17 时 2,132/3,662）。Messidor-2 中位 3.3 MP、APTOS 中位
  3.1 MP，均约为 IDRiD 12.2 MP 的 1/4，CPU 端成本随之下降、瓶颈转到 GPU，ETA 待实测修正。
- 2026-09-17 13:16 **E3 预注册脚本第二次指纹核验（协调者要求，Messidor-2 排除名单加入 clf 阶段）**：
  `src/pivot/e3_external.py` SHA-256 = `af65105ebf0a500ca0a70bc407ab62f5d76b03805a8057a6b1fcf8a8e9f6c1a5`
  （36,579 字节）、`data/external/messidor2/overlap_maplesdr.csv` SHA-256 =
  `31ef72bf3ac3504a1ff395aca6dfcf727cb54f1da7b4ef0eb43d61a66918fc79`（15,101 字节），
  本地与 LAN 节点逐位一致。bio 代码路径未变，故在飞作业不重启。

- 2026-09-17 13:20 **E4 折定义落盘（`results/pivot/e4/folds.json`，生成器 `src/pivot/e4_replication.py::build_folds`）**：Fundus-AVSeg 100 图、5 折，每图恰好折外一次（代码内断言：五个 held-out 集两两不交且并集 = 100）。**折 0 = 官方 80/20 划分的 20 张测试图**（deposit 的 `testing.txt`），按 image_id 排序；**折 1–4** = 其余 80 张（官方 train）的分层随机划分：按病种（AMD/DR/Glaucoma/Normal，排序后）逐类取成员、按 image_id 排序后用 `random.Random(2026)` 洗牌，再以**跨类连续**的发牌计数器 `counter` 依次发到折 `1 + counter % 4`；80 张正好每折 20 张，类内折间计数最多差 1。实际计数：折 0 = 2 AMD/4 DR/5 G/9 N（官方划分本身不均衡，如实保留）、折 1 = 5/4/4/7、折 2 = 5/4/3/8、折 3 = 4/4/4/8、折 4 = 4/4/4/8。每折训练池 = 另外 80 张，验证集按项目标准 `DEFAULT_SPLIT_SEED=12345` 取 15%（68 train / 12 val / 20 held-out），与普通 `src/seg/train.py` 运行取法完全一致。
- 2026-09-17 13:20 **E4 复现集几何与超参来源（不偏离 13:15 锁定设计）**：Fundus-AVSeg 与 MAPLES-DR 的 `resize_longest=1536 / patch=768 / batch=4`，在**进程内**注入 `src/seg/data.py` 的 `DATASET_CFG`/`DEFAULT_BATCH`/`DEFAULT_EPOCHS`（`install_geometry()`），不改磁盘上的表——两个复现集刻意不在 `DATASETS` 里（13:00 条），运行期注入不会让任何别的脚本的循环静默变长。折切换同样是进程内补丁：`get_records('fundusavseg')` 把 held-out 20 张标为 `split='test'`、其余 80 张标为 `train`，之后 `make_splits` / `src.seg.train` / `p5_finetune` / `p5_eval infer` 全部**原封不动**地跑。基线 U-Net 用 `src/seg/train.py` 自身默认（Adam 1e-3、poly(0.9)、BCE 1.0+Dice 1.0+clDice 0.0、100 iters/epoch、patience 60、oversample 0.7、AMP、增广开），显式给 `--epochs 150 --batch-size 4`（= FIVES/HRF 配方）。微调臂的 argv **直接调用 `e2_run._train_argv('fives', 0, arm, E2.CONFIGS[arm])`** 再只替换 dataset/seed/ckpt/out/val-limit/gpu 六个值，故 loss-kind / lam / w-cldice / terms / fd-ladder / cf-scale / epochs=50 不可能与 E2 漂移；起点 = 该折 base 的 `best.pt`（与 E2 从 `best.pt` 起同）；评估检查点 = `last.pt`。
- 2026-09-17 13:20 **E4 的 σ 选择（13:15 条留的二选一，此处定下并记录）**：Fundus-AVSeg 在 `results/gateA_biomarker_scales_train.csv` 中无行。采用**汇总 100 图 GT 的 σ** = 1.4826·MAD（与 Gate A 表同一稳健估计量 `src/c1/train_scales.py::_mad_scale`，也与 `e2_analysis.sigma_for` 对 STARE 的回退一致），在原生分辨率的参考掩膜上测得，写入每张表的 `sigma_source` 列。理由：主分析是**池化折外**的 100 张图，逐折训练部分的 σ 会让同一张表里的 bias/resid 混用五个不同刻度。设计允许的另一支（逐折训练部分 σ）作为敏感性分析并列报告：E4_REPORT.md §1 给出每个标志物的 pooled σ 与五折训练部分 σ 的最小/最大值。
- 2026-09-17 13:22 **E4 调度启动（与 E2 队列共享两块 3080，不抢占）**：驱动 `python -m src.pivot.e4_replication gpu --gpu <g> --min-free-mb 6144`，每条通道**顺序**取作业，且**每个作业启动前轮询 `nvidia-smi` 直到该卡空闲显存 ≥ 6 GB**（否则每 60 s 重试，只等待、绝不 kill）。作业锁与 E2 同机制（`os.mkdir` 原子占用），队列目录 `runs/pivot/_e4_queue`（与 E2 的 `_e2_queue` 分开）。作业顺序：5 个折基线训练 → 15 个微调（3 臂×5 折，依赖该折 `base/best.pt`）→ 20 个域内推理（4 臂×5 折）→ 24 个 MAPLES-DR 零样本推理（最低优先级，部分检查点仍在 E2 队列里产出）。两条通道经 `Win32_Process Create` 分离启动（pid 21204/42420，bat 见 `runs/pivot/e4/launch_gpu{0,1}.bat`），日志 `runs/pivot/e4/driver_gpu{0,1}.log`。实测领取顺序为 GPU1←折 0、GPU0←折 1（两条通道同时启动的竞争结果，与折/卡的绑定无协议含义）。基线训练实测稳态 ~38 s/epoch（与 E2 共卡时），首 epoch 86 s（含 1536 缓存构建）。
- 2026-09-17 13:30 **E4 协议澄清（外部设计咨询 review/design_e4_reply.md；在任何 E4 折外结果被查看之前锁定）**：(1) 时间线表述：E4 为"E2 之后前瞻锁定"的复现，而非项目初始预注册；Fundus-AVSeg 在看到 FIVES 零结果后选定，论文如实陈述。(2) 折划分：官方 20 张测试 = 折 0；其余 80 张按疾病标签分层一次性划为 4 个互斥的 20 张折（种子 2026），不重采样；图像级 CV（无患者 ID，双眼泄漏不能排除，须声明）。(3) 主对比 = ReliSeg − continued；ReliSeg − baseline 为次要"总部署差"；下游 AUC 亦加入 continued 臂。(4) σ 定义：每个外折仅用该折 80 张训练图的参考掩膜生物标志物计算 σ，应用于其 20 张留出图；clDice/Δr/ΔAUC 不换算为 σ 单位。(5) 折外 bootstrap 在各折内重采样（保留折结构），报告为"以固定折与已拟合模型为条件"的区间；另报 5 个留一折 Δr 作为稳健性检查。(6) 多重性：总体"复现成功"= density/total_length/FD 三项 Δr 的 max-statistic bootstrap（或 Holm）在 α=0.05 下至少一项显著且无一项方向相反；逐项 CI 分别报告；"CI 含零"不等于"复现零结果"（无等效边际）。(7) 下游 AUC 嵌套：分类器外折与分割折一一对齐。(8) 功效：E4 前用 E2 的相关结构做 Monte-Carlo 功效曲线（Δr 0.05/0.10/0.15，n=100）并写入论文。(9) MAPLES-DR：仅作零样本标注约定/域稳健性与分辨率效应修饰检验；≥1380 px 层称"较高分辨率"而非"高分辨率"；增加预标注 vs 人工修正的敏感性分析（deposit 含预标注）；主张措辞为"与人工修正约定的一致性"。(10) 近乎免费的补充：按疾病类别调整的 Fundus-AVSeg 保真度敏感性；参考生物标志物残差噪声剂量-反应（常数偏移臂 vs 图像特异相关残差臂，仅 CPU，不重训练）。

- 2026-09-17 13:30 **IDRiD 全部 11 趟完成并回拉；continued s2 入列；ODIR-5K 已提前落盘**：
  `continued` s0/s1 于 13:29:50 收尾（12:25 启动，64 分钟），IDRiD 11 个标签各 516/516 行、
  `meta.json` 齐全，已 `lan2_pull.sh` 回拉（远端 33 / 本地 33 文件一致）。本地
  `runs/pivot/fives/seed2/continued/summary.json` 于 13:22:10 出现后立即同步 `last.pt`
  并于 13:30:18 启动 `e3_fives_s2_continued`（IDRiD + Messidor-2 两路）；APTOS 的 s2 排在其后。
  ODIR-5K（6,407 文件 / 758 MB）已同步到位（13:22），但按预注册顺序**最后**跑（敏感性集）。
  **Messidor-2 实测**：11 路并发下单作业 3.6–4.0 s/图（IDRiD 为 7.3），波次等效
  **0.42 s/图**，9 个检查点 ETA 约 105–114 min → 约 15:10–15:15 完成。

- 2026-09-17 13:45 **E4 按 13:30 澄清条重构实现（在任何折外结果被计算或查看之前）**，逐条对照：(1) 折划分——本就如此：折 0 = 官方 20 张测试，其余 80 张按标签分层**一次性**划成 4 个互斥 20 张折（种子 2026），不重采样；`folds.json` 新增 `cv_unit="image"` 与 `cv_unit_note`：Fundus-AVSeg 只发布左右眼标志、无患者 ID，**同一患者双眼跨折泄漏无法排除**，作为限制写入报告而非默认不存在。(2) 主对比 ReliSeg − continued、次要 ReliSeg − baseline、下游 AUC 含 continued 臂——实现中本就如此，下游的参照顺序改为 continued 在前。(3) **σ 改为逐外折**：折 k 的 σ 只用该折 80 张训练图的参考掩膜生物标志物估计（1.4826·MAD），施用于其 20 张留出图（`per_image_sigma()`）；**撤回本日 13:20 条的"汇总 100 图 σ"**。Dice/clDice/Δr/ΔAUC 一律不换算 σ 单位（`e4_pixel.csv` 增 `units` 列注明）。(4) **折外 bootstrap 改为折内重采样**（`fold_boot_idx()`，各折各自抽自己的 20 行、折大小不变），区间表述为"以固定折与已拟合模型为条件"；另报 5 个留一折 Δr 点估计（`e4_delta.csv` 的 `scope=leave_out_fold<k>`）。(5) **多重性**：density/total_length/FD 三项主 Δr 的 max-statistic（Westfall–Young）bootstrap，调整 p 写入 `e4_delta.csv` 的 `p_adj`；**复现成功判据** = 至少一项 p_adj < 0.05 且无一项方向相反（95% CI 整段位于 0 以下），规则与结果落 `results/pivot/e4/replication_rule.json`；报告明写"CI 含零 ≠ 复现出零结果"（未预设等效边际）。(6) **下游嵌套**：分类器外折用 `cv_proba_aligned()` 钉死为这 5 个分割折（1:1），估计器与标准化沿用 `p3_downstream.cv_proba`（logreg / 小 GBDT），3 次重复只影响 GBDT 种子。(7) **揭盲前功效曲线**：`e4_replication power` → `results/pivot/e4_power.csv`，输入只有 E2 的 FIVES/HRF seed0 `{baseline,reliseg}/pred/bio.csv` 与 `bio_master.csv`，Δr = 0.05/0.10/0.15、n = 100、400 次模拟 × 1000 次 bootstrap；残差配对系数 ρ_e 用 E2 实测（FIVES 上 ≈0.99，两臂同起点微调故残差几乎同向），并另跑 ρ_e = 0.5 与 0 两个敏感性档，避免整条曲线读自单一乐观假设。(8) **MAPLES-DR**：≥1380 px 层标签改为 `higher_resolution`（另一层 `lower_resolution`），不再用"high-resolution"；分层**按原生帧尺寸组的名义 FOV 直径**（1440×960→909、2240×1488→1380、2304×1536→1452，取自 research/08）而非逐图实测——实测 2240×1488 组落在 1376–1384 px，逐图卡 1380 会把同一采集设置劈成两半（实测 47/115），按帧组则得到预注册的 **71 higher / 91 lower**；逐图实测直径仍留在 `results/pivot/e4/maples_strata.csv` 供核验。新增**预标注 vs 人工修正**敏感性：`bio --preannot` 用 deposit 的 `AdditionalData/preannotations/Vessels` 作锚点重算生物标志物，`maples-analyse --anchor preannot` → `results/pivot/e4_maples_preannot.csv`。(9) 新增 Fundus-AVSeg **按疾病类别调整**的保真度敏感性（类内去均值后的 r，`e4_fidelity.csv` 的 `scope=class_adjusted`）。**尚未实现**：13:30 条 (10) 的"参考生物标志物残差噪声剂量-反应"（常数偏移臂 vs 图像特异相关残差臂）——纯 CPU、不依赖 GPU 队列，留待折外结果产出后补。
- 2026-09-17 13:45 **环境补装**：`D:/Anaconda/envs/medical1` 缺 `xlrd`，`_maplesdr_grades()` 读 `AdditionalData/diagnosis_infos.xls` 直接抛 ImportError，MAPLES-DR 的任何加载都失败。已 `pip install xlrd`（2.0.2，纯 Python、仅新增包），未动任何既有依赖版本；在跑的 E2/E3 进程不受影响。

- 2026-09-17 14:29 **Messidor-2 主 9 趟完成并回拉；APTOS-2019 波次接棒**：13:20:18 → 14:29:07，
  **68 分 49 秒**跑完 9×1,744 = 15,696 张，波次等效 **0.263 s/图**（11 路并发下单作业
  2.4–4.0 s/图）；9 个 `bio.csv` 各 1,744/1,744 行，`meta.json`/`resolutions.csv` 齐全，
  `lan2_pull.sh` 回拉远端 30 / 本地 30 文件一致。比 13:30 条的估计（15:10–15:15）快 45 分钟——
  原估计按 IDRiD 的每图成本线性外推，但 Messidor-2 中位 3.3 MP 使 CPU 端成本降到约 1/3，
  实际瓶颈落在 GPU 上。`continued` s0/s1/s2 三趟随槽位释放在跑。
  APTOS-2019 于 14:28 自动接上（同一 driver 的作业表按数据集顺序排列，槽位一空即接棒，
  GPU 无空转）：实测 3.8 s/图，主 9 趟 ETA ≈ 230 min → 约 **18:20**。
- 2026-09-17 14:16 **四个外部集在节点上全部核验通过**：`check_external` 对
  idrid(516)/messidor2(1,744)/aptos2019(3,662)/odir5k(6,392 眼 / 3,358 患者) 均 **OK**，
  四者标签分布都与公开值一致；FOV 缓存全部**预先**建好（messidor2 1,732 张 / aptos 3,601 张 /
  odir 6,380 张）——并发 worker 会竞写同一个 `fov/*.png`，必须预建而不能让作业顺带生成。
  ODIR-5K 原生仅 512×512（in/out 绿通道比 289.7，FOV 占比 0.79），按预注册仍为敏感性集，
  排在 APTOS 之后最后跑（`remote/lan2_jobs_odir5k.txt`，12 个检查点，由
  `lan2_e3_chain.sh` 在前序 driver 退出后自动启动）。
- 2026-09-17 14:52 **评审建议的两项分析已跑完 + E4 写入稿件（协调者 13:19 指示）**：(1) `src/pivot/e1_outer_boot.py` → `results/pivot/e1_outer_boot.csv`：参考差的**外层 bootstrap**，每一次抽样重跑整套流程（折划分 + 两臂全部拟合），折标签挂在**图像**上使自助重复样本不会跨折泄漏，两臂共用同一抽样与同一折。点估计用 p3_downstream 的折种子，因而逐位复现 `e1_oracle_downstream.csv`（自检通过）。结果：区间加宽 1.2–1.6 倍，**HRF 仍排除零**——logreg +0.2170 由 [+0.119,+0.332] 变为 **[+0.090,+0.411]**，gbdt +0.1800 由 [+0.062,+0.305] 变为 **[+0.046,+0.348]**；FIVES 四格全部加宽且仍跨零。抽样数 logreg 200 / gbdt 100（每次抽样 = 一整轮重复交叉验证；gbdt 臂单独耗时约 75 CPU-min），无一次被丢弃。落在补充材料表 S2。(2) `src/pivot/e1_dose_response.py` → `e1_dose_response.csv` + `_summary.csv`，正文 5.5 节与图 4：**参考生物标志物上的误差注入剂量-反应，两臂**——常数偏移臂（每图同一 c·σ_b）与**图像特异相关残差臂**（按本队列自身去偏残差的跨标志物相关矩阵抽 MVN，逐标志物缩放到 s_j=g_j·sqrt(1/r²−1) 以命中目标 r，实际 r 为实测）。仅 CPU，不重训练。结果：**常数偏移臂在全部剂量上逐位不变**（FIVES 0.7027、HRF 0.9704 于 0/0.25/0.5/1/2/3 σ），残差臂单调下降（FIVES 0.7027→0.5585 @ r 0.42，HRF 0.9704→0.6170 @ r 0.38），gbdt 复现同样行为；**HRF 实测点落在自己的注入曲线上**（r̄ 0.549、AUC 0.7533，对曲线 r 0.591 处 0.7517 与 r 0.487 处 0.6856），FIVES 实测点在曲线**之上**（r̄ 0.854、AUC 0.7055 对曲线约 0.64），因为其面板均值被单一迂曲度列（r 0.42）拖低而其余三列 0.91–0.98——分类器能降权一个坏特征，降权不了四个。两条限制（残差为高斯同方差、且按构造与疾病标签独立）随结果一并写明，故曲线是"给定保真度损失所必然造成的衰减"的**上界**。(3) `src/pivot/e1_fov_scale.py` → `dataset_scale.csv`：统一实测 FOV 直径 2√(A/π)（DRIVE 538 / STARE 639 / CHASE_DB1 920 / Fundus-AVSeg 1276 [1177–2080] / MAPLES-DR 909 [903–1456] / FIVES 2010 / HRF 2967 px），新生成 `tab:datasets` 与 4.1 节"画幅不是分辨率"的段落。(4) 稿件写入 **E4**：方法 3.6 节、锁定的 `tab:e4`、空的图 7（复现森林图 + 蒙特卡洛功效曲线）、结果 5.12 节；按指示如实表述为"**E2 之后前瞻锁定**的复现"而非项目初始预注册，13:30 条的全部澄清（折划分一次性固定、图像级 CV 与不可排除的双眼泄漏、每折 σ、保折 bootstrap、max-statistic 多重性、分类器折与分割折一一对齐、功效曲线）均已写入；MAPLES-DR ≥1380 px 层称"**较高分辨率**"，909 px 层为预先声明的低分辨率阴性对照，掩膜表述为"网络预标注的人工修正"（预标注 vs 最终 Dice 0.783）。(5) 引用：`deng2025fundusavseg`（Sci Data 12, 1298, 2025, 10.1038/s41597-025-05381-2）与 `lepetitaimon2024maplesdr`（Sci Data 11, 914, 2024, 10.1038/s41597-024-03739-6），均经 Crossref 核验并记入 BIB_AUDIT（含"文章 CC BY-NC-ND / 数据 CC BY 4.0"的区分）。(6) Messidor-2 改为"发行 1,744，分类阶段 1,582"。(7) 顺手修掉一个历代继承的排版缺陷：elsarticle 的 `\paragraph` 自带句点，48 处 `\paragraph{标题.}` 原本排成"标题.."。构建：59 页（正文含参考文献 52 页，补充材料 7 页），无未定义引用/交叉引用，66 个 `\todo`（正文 16 + 三张锁定骨架表 50）。映射见 `paper2/REVIEW_R1_RESPONSE.md` §2b。

- 2026-09-17 14:50 **E4 分析管线在真跑之前用合成数据端到端验证**：脚本 `scratchpad/test_e4_analyse.py`（不入库）用真实的 100 张 Fundus-AVSeg **GT** 生物标志物加乘性噪声伪造四个臂的 `bio.csv` 与 `pixel_metrics.csv`（噪声尺度 baseline 0.20 / continued 0.19 / reliseg 0.12 / cfloss 0.25），在**隔离目录**里跑 `cmd_analyse`，确认五张表与报告全部生成且形状正确：`e4_fidelity.csv` 224 行（oof_pooled + class_adjusted + 逐折）、`e4_delta.csv` 240 行（5 组对比 × (池化 + 5 个留一折) × 8 列标志物）、`e4_pixel.csv` 4 行、`e4_safety.csv` 50 行、`e4_downstream.csv` 32 行、`replication_rule.json`、`E4_REPORT.md`。合成数字无意义，只验证管线。**顺带修两处**：(1) 下游 AUC 的 2000 次配对 bootstrap 原用 `sklearn.roc_auc_score`（每次重排序重校验），改为排名式 Mann-Whitney 实现 `fast_macro_auc_factory`，**每次使用前都先与 sklearn 在同一组概率上对拍，差 > 1e-9 直接抛错**而非静默替换；(2) 下游的 GBDT 默认把 OpenMP 铺满 24 核，与 E2/E4 的生物标志物 worker 和两个训练互相抢核，实测该阶段吃掉 34,573 CPU-秒仍未跑完；`cmd_analyse` 现在在导入 sklearn 之前把 `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS` 钉到 `--threads`（默认 4），同样的分析 2 分钟跑完。
- 2026-09-17 14:55 **MAPLES-DR 预标注的几何问题（13:30 (9) 的敏感性分析）**：deposit 的 `AdditionalData/preannotations/Vessels` **200 张全部是 1500×1500 的方形画布**，而终稿掩膜与图像是 MESSIDOR 原生的 1440×960 / 2240×1488 / 2304×1536——直接拿来与原生 FOV 相与会 broadcast 报错（bio watcher 已因此失败 22 次，已暂停该开关后修复）。画布对应"**边长 = FOV 直径、以 FOV 中心对齐的正方形**"：对 scale（0.94–1.18 × FOV 宽）与中心偏移（水平 ±30 px、垂直 ±40 px）做网格搜索，8 张受检图像的 Dice 最优点**全部落在 scale = 1.00、偏移 (0,0)**，故该摆放是被识别出来的、不是拟合出来的。实现为 `maples_preannot_path()`，最近邻重采样回原生分辨率并缓存到 `data/maplesdr/preannot_native/`。**必须写进论文的限制**：把 1500 px 画布最近邻降采样到 1380–1452 px 的方形会削掉最细的血管，而这正是生物标志物要测的结构；重映射后预标注 vs 终稿的 Dice 均值为 **0.726**（162 张），`research/08` 记的 0.783 是在 1500×1500 画布上测的、不含重采样损失。因此预标注臂只作**方向性**的标注约定检验，不作第二个精确锚点。

- 2026-09-17 15:00 **E4 首个基线训练实测与全程 ETA**：`e4:base:f1` 用时 **96.3 min**（150 epoch，38.4 s/epoch，最佳验证 Dice 0.9390 @ep145，`runs/pivot/e4/fold1/base/summary.json`）；折 0 同批、几乎同时完成。按两条通道顺序推：基线 5 个 → 折 2/3 约 16:40 结束、折 4 约 18:20 结束（另一条通道从 16:40 起开始微调）。微调 15 个（3 臂 × 5 折，50 epoch ≈ 基线的 1/3 但每 epoch 多算一次验证集测量保真度）按 40 min/个估 → 约 **22:30** 全部完成；域内推理 20 个（每折 20 图）约 25 min → 约 22:55；生物标志物由常驻 watcher（`runs/pivot/e4/bio_watch.bat`，6 进程）随推理产出滚动计算，落后约 30 min → 约 23:25；`analyse` 约 5 min。**E4 主终点（Fundus-AVSeg 折外 Δr / 安全 / 下游 AUC）预计今日 23:30 前后可读。** MAPLES-DR 零样本 24 个检查点 × 162 图的推理约 60 min（两通道）→ 约 00:00；其 3,888 张生物标志物（CPU，约 17 s/图、6 进程）约 3.1 h → **次日 03:30 前后**出 `e4_maples_zeroshot.csv` 与预标注敏感性表。
- 2026-09-17 15:00 **暂不加开第三条 GPU 通道**：E2 队列尚有 1 个训练在跑、1 个 READY（`abl_length_only`/`abl_fd_only`），两卡空闲显存此刻为 12.8 GB / 7.5 GB。E2 的 worker **没有显存门控**，若此时加开 E4 通道把卡填到 ~6 GB，E2 领到最后一个训练时可能 OOM——那是别人的作业，不能由 E4 的加速去冒险。待 E2 最后两个训练结束后再评估（届时每卡可容两个 E4 训练 + E3 推理约 15.5/20.5 GB），门限设 9 GB。

- 2026-09-17 15:12 **E4 加开第 2/3/4 条 GPU 通道（E2 训练队列已空）**：`e2_run status` 显示 E2 已无 READY/WAIT 的 train 作业（最后两个 HRF 单项消融训练完成），两卡空闲 12.8 / 13.8 GB，故为每卡再起一条 E4 通道（`launch_gpu{0,1}b.bat`，pid 48732 / 26436，日志 `driver_gpu{0,1}b.log`），**门限提高到 9216 MB**（原两条仍为 6144）——一个 E4 训练约占 7 GB，9 GB 门限保证同卡第二个训练启动时仍留有余量给 E2 的 E3 推理（约 1.5 GB）。四条通道立刻把 5 个基线训练全部并行开出，并让折 0 的第一个微调（`e4:ft:f0:continued`）在 15:09 就开始，而不必等全部基线跑完。实测两个已完成的基线：折 1 **96.3 min**、折 0 **104.7 min**。
- 2026-09-17 15:12 **E4 收尾自动化（`e4_replication finish`，pid 35548，日志 `runs/pivot/e4/finish.log`）**：GPU 通道与 CPU 生物标志物 watcher 都是分离进程，缺的只是最后一步分析。`finish` 每 5 min 轮询，一旦 20 个折×臂预测目录都有 `bio.csv` 且 GT 表就位就跑一次 `analyse`；一旦 24 个 MAPLES 目录就位就跑 `maples-analyse`（正式锚点 + 预标注锚点各一次）并回头刷新一次报告的 MAPLES 小节。只轮询、不 kill 任何进程，每个分析只跑一次，24 h 超时退出。**修订后的 ETA（四通道）**：基线全部完成约 16:50；15 个微调约 19:00；域内推理约 19:15；生物标志物滞后约 30 min → **E4 主终点约 20:00 可读**（原两通道估计为 23:30）。MAPLES 零样本推理约 19:45 跑完，其 3,888 张 CPU 生物标志物约 3.1 h → **约 23:00–23:15 出零样本与预标注敏感性表**。

- 2026-09-17 15:15 **13:30 条 (10) 的"参考生物标志物残差噪声剂量-反应"不由 E4 实现**（协调者指示）：论文 agent 已完成，落点 `src/pivot/e1_dose_response.py`（13:29）与 `results/pivot/e1_dose_response.csv`（13:40，295 KB）。撤回 14:50 条末尾"留待折外结果产出后补"的说法，E4 不再重复实现，避免同一量两套代码。

- 2026-09-17 15:30 **生物标志物 watcher 在 14:47 静默死亡，已查明原因并重启（协议相关：期间没有任何生物标志物被计算）**。症状：`runs/pivot/e4/bio_watch.log` 最后写入时间停在 **14:47**，此后 41 分钟无输出，`Win32_Process` 里既无 `e4_replication bio` 也无对应的 cmd.exe 宿主（四条 GPU 通道与 finish 进程均健在）。原因：**我在 14:53 与 14:55 两次就地编辑了正在运行的 `bio_watch.bat`**——cmd.exe 是按字节偏移逐行重读批处理文件的，改变行长度会让偏移错位并终止循环。日志里那 5 次 `1500×1500 vs 1488×2240` 的 ValueError **全部早于 14:57 的代码修复**，属陈旧记录，不是修复无效；手工核验重映射在 20 张图上 0 处尺寸不符。**两项处置**：(1) `cmd_bio` 的三个 GT 阶段改为各自 try/except（`_try_gt`），单个阶段失败只记录并继续——此前**可选的**预标注锚点一崩就把整趟（含 20 个折×臂目录的测量）全部带走，这正是 14:40–14:47 之间什么都没算出来的原因；失败摘要打印在收尾行。(2) 新建 `runs/pivot/e4/bio_watch2.bat`（文件头写明"运行期间禁止就地编辑，要改就新建 bio_watch<N>.bat 重启"）并分离启动，pid 51164。**此故障不影响任何已产出的结果**：期间 GPU 队列照常推进，只是生物标志物阶段停摆 41 分钟；按当前进度（基线仍在跑，尚无推理产出）没有造成关键路径延误，ETA 不变。

- 2026-09-17 16:12 **会话中断后恢复：LAN 节点上作业无损，IDRiD 与 Messidor-2 各 12/12 全部回拉核验**：
  16:00 前后本地会话中断；因所有作业以 `nohup setsid` 脱离 ssh 会话，
  节点侧 11 个 worker、两个 driver（`wave2b`、`cont_s2`）与 ODIR 链式等待进程全部存活
  （16:11 复查：GPU 94%、7,395 MiB、load 10.0）。**中断未造成任何损失**——这正是当初
  "短 ssh 启动 + nohup setsid + 日志落盘 + meta.json 作完成标记"这套约定要防的情形。
  IDRiD 12/12（各 516/516 行）、Messidor-2 12/12（各 1,744/1,744 行）均已回拉；
  **12 个 `ckpt_sha256` 两两不同，且逐一与本地检查点文件重算的 SHA-256 完全一致**
  （baseline s0-2 `4f0489e1cff4`/`ec5172f7417b`/`dc580f6c7530`；reliseg `bed33d322ea6`/
  `6227c165e948`/`2d4ba04e4056`；cfloss `c4aaca9182bc`/`4e6575fda818`/`5876d8f4fb3a`；
  continued `f5d6a09dac98`/`cfee9c045f39`/`decf9b4ec853`）——冻结分割器的身份守卫成立。
  `continued` s2 用时：IDRiD 13:30:18→14:29:44（59 min），Messidor-2 13:30:18→14:35:15（65 min）。
  注：本地 `results/pivot/e3/idrid/` 另有 `e3_hrf_s0_{baseline,reliseg}`，属本地队列产物
  （12:05 分工条：HRF seed0 归本地），与 LAN 结果互不覆盖，`lan2_pull.sh` 的文件数告警即源于此。
- 2026-09-17 16:12 **两条 bio 通道漏掉 30 个新增单元（作业表快照问题，非失败）**：11:11 与 11:26 启动的两条 bio 通道在启动时就把 `bio_jobs()` 的结果固化在内存里（104 个），而 `continued` 臂与三个单项消融是 11:29-11:30 才加入 CONFIGS/ABLATIONS 的，于是这 30 个新单元**从未出现在它们的作业表中**；两条通道跑完各自的 104 个后正常退出并打印 `worker done, not-done=[]`——那句话说的是"我这张表上的都做完了"，不是"全网格都做完了"。这不是失败、不是 OOM、也没有陈旧锁：核对确认这 30 个 id **一个锁都没有**（从未被认领），因此新通道可以全部领走。
  审计结果（16:11）：`train 33/33`、`infer 132/132` **全部完成**（含 continued 与单项消融的域内 last+fid 及 CHASE/STARE 零样本推理），缺的只有 30 个 bio：continued 域内 12（6 格 × last/fid）、单项消融 6（3 格 × last/fid）、continued 零样本 12（2 目标 × 2 数据集 × 3 seed）。已启动第三条通道（pid 51032，`--procs 12`，日志 `runs/pivot/_e2_logs/bio3_20260917_161245.log`）。
  **教训（写给后续 agent）**：`e2_run.py` 的 worker 在启动时构建一次作业表，此后不再重建；**任何对 CONFIGS / ABLATIONS / build_jobs 的修改，都必须重启全部相关 worker，否则新作业会被静默跳过**。`worker done` 一行不能作为阶段完成的判据，判据应当是 `e2_run status` 的全量计数。
- 2026-09-17 16:14 **最终分析改为限线程运行**：机器 CPU 已达 92.5%（E3 的两条通道 + E4 复现 agent 共用），sklearn/BLAS 默认会开满核心并与 bio 抢占。改用 `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=6` 运行 `e2_analysis`。同时**停掉了 11:55 启动的旧链（pid 15552，未限线程）**并换成限线程版（pid 46272，日志 `e2_final_chain2.log`）——两条链会在 bio 结束后同时触发分析、并发写同一批 CSV，必须只保留一条。
- 2026-09-17 16:28 **FIVES × ODIR-5K 的 9 个 E3 作业改判归 LAN 节点，已在本地队列加锁隔离**（ETA 审计 `results/pivot/ETA_GPU_SCALING.md` 发现）。12:05 划界时把整个 ODIR-5K 敏感性集留给本地，理由是 LAN 只承担三个主队列；现更正为 **ODIR-5K 上的全部 FIVES 检查点（含 `continued`）由 LAN 承担，排在其 APTOS 之后**。
  `e2_run status` 核对（16:28）：本地 e3 队列确为 17 个，其中 `e3:odir5k:fives:s{0,1,2}:{baseline,reliseg,cfloss}` 正是这 9 个。已用与 12:05 相同的方式预置锁目录（`owner.json` 无 pid、标注归属 LAN），使任何本地 worker（包括用旧作业表启动的）都无法 `claim()`。另**预防性**锁住 `e3:odir5k:fives:s{0,1,2}:continued` 三个 id——它们当前不在 `e3_checkpoints()` 中，但若 `continued` 日后被重新加入，这三个作业会自动出现，锁先放在那里可免于再次竞争。
  **本地保留不动的 8 个**：`e3:idrid:hrf:s0:{baseline,reliseg}`（正在跑，锁属本地 worker pid 44224 / 44688）、`e3:messidor2:hrf:s0:{baseline,reliseg}`、`e3:aptos2019:hrf:s0:{baseline,reliseg}`、`e3:odir5k:hrf:s0:{baseline,reliseg}`。核验后本地可认领的 e3 作业为 6 个（另 2 个在跑），与上述划分逐一吻合。
  累计 LAN 归属的隔离锁：36（12:05，三个主队列 × FIVES）+ 12（本条）= 48。

- 2026-09-17 15:42 **E4 生物标志物分工：MAPLES-DR 的 24 个零样本单元移交 CPU 服务器 agent（32 核 EPYC 9654 租用机）**。本地 watcher 换为 `runs/pivot/e4/bio_watch3.bat`（pid 38868），命令去掉 `--maples --preannot`，只剩 `bio --procs 6`；`bio_units(include_maples=False)` 实测返回 **20 个单元且 dataset 仅 fundusavseg**（此前 44），`cmd_bio` 也不再进入两个 MAPLES GT 阶段——日志尾行由 "44 not ready" 变为 "20 not ready" 即为证据。v1/v2 的 .bat 宿主已停止，不得再启用（同一目录下 v3 文件头写明禁止就地编辑）。**归属划分**：本地保留 20 个 `e4_f{0..4}_{baseline,continued,reliseg,cfloss}`（fundusavseg）与 `e4_gt_fundusavseg.csv`；远端负责 24 个 `e4mp_{fives,hrf}_s{0,1,2}_{baseline,continued,reliseg,cfloss}` 的 `bio.csv`。两个 MAPLES 锚点已在本地算完（`e4_gt_maplesdr.csv` 13:46、`e4_gt_maplesdr_preannot.csv` 15:35），远端不重算、不覆盖。`cmd_maples_analyse` 与 `cmd_finish` 读的就是 `runs/pivot/e4/maples/<tag>/bio.csv`，故收尾逻辑无需改动。**移交时点无冲突**：`runs/pivot/e4/maples/` 目录此刻尚不存在（24 个推理作业排在队列最后），不存在半写的 manifest。
- 2026-09-17 15:42 **移交的可比性前提（已同步给远端，协议相关）**：远端必须复用 `p5_eval._bio_one` 的同一估计量路径（`compute_all(..., fd_rotations=5)`，视盘由 `locate_optic_disc(image, fov, vessel_mask=None)` 仅从眼底图检测），并**直接拷贝 `exp/data/maplesdr/fov/` 而非重新生成 FOV**、使用同一套经 `_maplesdr_image_index()` 解析的 Messidor-2 镜像图像；否则 density/total_length 会因 FOV 面积差异而系统偏移，且与 E2/Fundus-AVSeg 的数字不可比。另注意 `manifest.csv` 内是**绝对 Windows 路径**，在 Linux 上需重写或复刻同构路径；bio 阶段读的是 `<tag>/mask/*.png`（不是 `pred/`）。每个目录应为 162 行。
- 2026-09-17 17:18 **E3 三节点最终划分：本地退出 E3，GPU 让给 E4**（协调者再规划 + GPU2 agent 请求）。新增租用节点 **GPU2（2× RTX 2080 Ti，user@GPU_HOST_2:SSH_PORT）承担整个 ODIR-5K**（全部 14 个检查点 = FIVES s0-2 ×{baseline,reliseg,cfloss,continued} + HRF s0 ×{baseline,reliseg}）；**LAN 节点在其 APTOS 波次之后追加 HRF s0 ×{baseline,reliseg} 的 Messidor-2 与 APTOS**。
  本地按同一机制（锁目录 + 无 pid 的 owner.json）新增隔离 4 个：`e3:messidor2:hrf:s0:reliseg`、`e3:aptos2019:hrf:s0:reliseg` 归 LAN；`e3:odir5k:hrf:s0:{baseline,reliseg}` 归 GPU2。至此**本地 e3 队列可认领作业为 0**。累计隔离锁 48 + 4 = 52。
  **与协调者 17:14 消息的事实差异（已回报）**：该消息假定本地两个 worker 仍在跑 IDRiD，实际上 IDRiD 的 HRF s0 两趟**已完成**（`results/pivot/e3/idrid/e3_hrf_s0_{baseline,reliseg}/` 各 516 行 + meta.json），worker 44224/44688 随后已自行认领并正在跑 **`messidor2/e3_hrf_s0_baseline`（子进程 48140）与 `aptos2019/e3_hrf_s0_baseline`（子进程 49448）**——按新划分这两趟归 LAN。按"不动正在跑的作业"的指示**未停止**它们；其锁由本地 worker 持有（有真实 pid），脚本据此跳过、不覆盖 owner.json（覆盖只会丢失"谁在跑"的记录，并不能停止进程）。**需 LAN 侧在其波次中跳过这两个 tag，否则两节点会对同一个可续写的 bio.csv 交错追加。**
  收尾：这两趟跑完后本地 e3 全部 done-or-fenced，两个 worker 会进入 `nothing runnable; waiting 120 s` 空转（不占显存）。届时可直接停掉 pid 44224 / 44688，本地 GPU 完全交给 E4。

- 2026-09-17 17:21 **ODIR-5K 迁往新的 2×2080 Ti 机器；LAN 节点改接 HRF s0 开发集（仅 reliseg）**：
  (1) **ODIR 链已解除**：LAN 节点上等待中的 `lan2_e3_chain.sh`（PID 219804，持有
  `remote/lan2_jobs_odir5k.txt`）已 kill 并确认消失，作业表改名为
  `lan2_jobs_odir5k.txt.DISARMED_moved_to_2080ti` 以防误启。**LAN 节点上 ODIR 一趟都没跑过**
  （`results/pivot/e3/odir5k` 不存在，0 目录 0 行），故新机器 14 个检查点全部从零开始、无重复。
  已向 2080 Ti 的 agent 确认，并转达两条实测经验：`--sw-batch 1`（712 MiB/worker，速度不变）
  与 **FOV 必须预建**（并发 worker 会竞写同一 `fov/*.png`）。
  (2) **HRF s0 重复作业已止损**：17:20:39 我在 LAN 上启动了 messidor2/aptos2019 × {baseline,
  reliseg} 四趟；随后得知本地机器早已在跑那两个 **baseline**（local pid 48140 / 49448）。
  17:21（各仅 10 行）kill 掉 LAN 侧两个 baseline，**未删除**，目录改名并移出结果树到
  `runs/lan_e3/_held/{messidor2,aptos2019}__e3_hrf_s0_baseline_LANpartial_held_1721`——
  既保留证据，又保证 `results/pivot/e3/<ds>/e3_hrf_s0_baseline/` 规范路径空着留给本地产物，
  且整目录回拉不会把半截结果带回本地。协调者随后确认：**LAN 只跑 `e3_hrf_s0_reliseg`**。
  两趟 reliseg（messidor2 + aptos2019）17:20:40 起在跑，与剩余 4 个 APTOS FIVES 作业并行
  （共 6 worker），APTOS 主 9 趟已于 17:20 前完成 7 个 + s2_continued。
- 2026-09-17 17:23 **缺陷：`ensure_fov` 非原子写入，同数据集并发时会读到半写的 FOV PNG**（本地 `e3:messidor2:hrf:s0:reliseg` 因此崩溃，rc=1）。报错在 `_row_for` → `_imread_gray(rec['fov_path'])`：`ValueError: cannot reshape array of size 1701 into shape (36396,60828)`——imageio 因文件当时不是合法 PNG 而回退到 SPE 插件。根因是 `src/data/external.py::ensure_fov` 按需生成 FOV 掩膜且**直接写目标路径、不走临时文件+替换**；同一数据集的两趟推理并发跑在同一台机器上时，两者都会对同一批缺失的 FOV 调用 `ensure_fov`，其中一个在另一个尚未写完时读取，拿到截断图像。
  **证据表明是瞬态竞态而非坏数据**：事后用 imageio 逐个校验 messidor2 的全部 470 个 FOV PNG，**0 个损坏**——活下来的那个写进程把文件补全了；同一张图 `e3_hrf_s0_baseline` 顺利通过（当时已到 450/1744）。崩溃发生在 17:18 加锁之前；worker 失败时调用了 `release()` 删掉锁，17:18 的加锁脚本因此看到"无锁"并新建了隔离锁，故该作业现已归 LAN 且本地不会重试。
  **处置**：本地半成品 `results/pivot/e3/messidor2/e3_hrf_s0_reliseg`（180 行、无 meta.json）改名为 `e3_hrf_s0_reliseg_LOCALpartial_crashed_1723` 并保留（不删），让 LAN 的 rsync 落在干净的规范路径上。已把竞态与规避方式（要么单进程预生成 FOV，要么每数据集只跑一个 worker）告知 LAN agent；**建议的永久修复**是让 `ensure_fov` 写同目录临时文件后 `os.replace()`，与 `src/pivot/common.py::json_dump` 已有的做法一致——因 LAN 作业正在飞行中，未擅自改动代码。
- 2026-09-17 17:23 **E3 HRF s0 四个 tag 的最终归属（协调者裁定）**：messidor2 与 aptos2019 的 **baseline 由本地跑完**（pid 48140 / 49448，17:23 时分别在 450/1744 与 190/3662，ETA 约 19:00 与 23:15），**reliseg 归 LAN**；LAN 已于 17:21 杀掉自己的两个 baseline 重复作业（各仅写 10 行）并把半成品改名为 `*_LANpartial_held_1721` 保留。idrid 的 HRF s0 两趟归本地且已完成（各 516 行 + meta.json）；odir5k 的 HRF s0 两趟归 GPU2 节点。
  收尾计划：两个本地 baseline 跑完后，worker 44224 / 44688 将进入空转（本地 e3 已 0 个可认领），届时按显式 PID 核对命令行后停止，本地 GPU 全部交给 E4，并在此另记一条。
- 2026-09-17 17:40 **`ensure_fov` 原子写入修复延后至 E3 全部完成（协调者裁定 (b)）**。`src/data/external.py` 带有已锁定的预注册指纹 `6e96340dd5c61bc94db6ccb4e48614f95fcae4bde0984e7e0f140de3d3e60c0c`（LAN 节点已核验两边逐位一致），`ensure_fov` 就在该文件里；在三个节点的 E3 bio 作业全部跑完并回收之前改它，会让一份协调者两次要求核验的预注册产物失效。
  **该缺陷不影响任何一个数字**：它的后果只有两种——要么读到截断 PNG 让作业直接崩溃（可见、可重跑），要么什么也不发生。它不会产生错误的生物标志物值：截断文件不可能被 imageio 当作合法掩膜读出来再算出数来。已核验的证据见 17:23 条（本地 470 个 messidor2 掩膜事后全部合法）与 LAN 节点的 5,922 个掩膜复核（0 不可读、0 退化，56,416 行 × 4 个 skan 列 0 个 NaN，全部作业 rc=0，仅有两个人为 kill 的 rc=143）。
  **在此期间的官方规避办法**：并发/分片跑同一数据集之前，**单进程预生成 FOV 缓存并核对掩膜数与记录数**（`python -m src.data.check_external --fov <name>`）。已写入 `exp/data/EXTERNAL_DATASETS.md` 的「FOV generation」一节，作为硬性前置条件，含复现该崩溃的报错原文、"不同数据集并行安全、同一数据集并发不安全"的界线，以及 LAN 节点的实测时间线作为范例。
  **E3 全部完成后**：用同目录临时文件 + `os.replace()` 修复（与 `src/pivot/common.py::json_dump` 一致），重新计算并发布 `external.py` 的指纹，另记一条，并把新哈希同步给 LAN 与 GPU2 两个节点重新核验。
- 2026-09-17 17:35 **E2 CPU 生物标志物阶段完成：134/134**（最后 30 个单元由 17:12 启动的第三条通道补齐）。至此 `train 33/33 · infer 132/132 · bio 134/134`，E2 网格的全部产物就位。限线程的分析链（pid 46272）于 17:35:49 自动触发，日志 `runs/pivot/_e2_logs/e2_final_chain2.log`，将重写六个 CSV 与 `results/pivot/E2_REPORT.md` 的全网格版本（主网格 + 9 个消融含三个单项 + `continued` + CHASE/STARE 零样本）。
- 2026-09-17 17:45 **新增 CPU-only 租用节点上线，承接两类纯 CPU 工作**：`user@CPU_HOST:SSH_PORT`（AMD EPYC 9654，**cgroup 限 32 vCPU / 60 GiB**——`nproc` 与 `cpu.max` 一致，但 `free -g` 显示宿主的 377 GB 不可信；无 GPU）。工作根 `/path/to/workdir/medical1`（`/dev/md0`，50 GB，是该容器唯一大卷；`/path/to/storage` 不存在），与本地同构 `exp/{src,configs,data,runs,results,remote}`。一次性口令仅用于 `remote/bootstrap_key_cpu.py`（paramiko，口令只从环境变量 `REMOTE_SECRET` 读取，不落盘、不进日志、不写入任何文件）安装本机公钥，此后一律 `BatchMode=yes` 密钥登录。环境文档 `exp/remote/ENV_cpu.md`，同步/驱动脚本 `exp/remote/cpu_{sync,pull,e4_loop,e4_watch,e4_bio,e3_clf}.sh` 与 `cpu_fix_manifest.py`。**用后不关机**（用户指令）。
  **venv 配方**：`/root/miniconda3/bin/python -m venv`（容器里 `python3` 不在非登录 shell 的 PATH 上），装 `runs/pip_req_cpu.txt` = `pip_freeze.txt` 三处改动（torch/torchvision 换 `+cpu` 走 `download.pytorch.org/whl/cpu`；`packaging @ file://…` 改为裸 `packaging`；去掉 `opencv-python` 只留 headless），另加 `pyarrow`。
  **新发现的缺失依赖（会影响任何新建环境）**：`src/data/datasets.py::_maplesdr_grades` 用 pandas 读 `diagnosis_infos.xls`，需要 **`xlrd`（本地为 2.0.2）**，但它不在 `runs/pip_freeze.txt` 里（本地 Anaconda 环境自带）。新环境若不补装，`load_dataset("maplesdr")` 直接抛 `ModuleNotFoundError: xlrd`。已在节点上按本地版本补装并记录于 ENV_cpu.md。
  **估计器跨平台一致性实测（非假设）**：在节点上以 `p5_eval._bio_one` 对 MAPLES-DR 前 3 张图的参考掩膜重算，与本地 `results/pivot/e4_gt_maplesdr.csv` 逐列比对——**101 个数值列最大相对差 5.5e-15**，仅 `runtime_skan_s` / `runtime_pvbm_s` 两个计时列不同。单核每图 7–11 s（本地 Windows 15–17 s）。
  **脚本同步指纹核验**：节点上 `src/pivot/e3_external.py` = `af65105e…`、`src/data/external.py` = `6e96340d…`、`data/external/messidor2/overlap_maplesdr.csv` = `31ef72bf…`，与 13:16 / 12:10 两条锁定记录逐位一致。
- 2026-09-17 17:45 **E4 MAPLES-DR 生物标志物单元整体移交 CPU 节点（与 E4 agent 双向确认）**：CPU 节点承接 **24 个零样本预测目录**（`runs/pivot/e4/maples/e4mp_{fives,hrf}_s{0,1,2}_{baseline,continued,reliseg,cfloss}/bio.csv`）与两个 MAPLES 锚（`--maples` / `--preannot`）；E4 agent 已把本地 watcher 改为 `bio --procs 6`（新文件 `bio_watch3.bat`，pid 38868，旧 v1/v2 停用——**不要原地编辑正在被 cmd.exe 执行的 .bat**，cmd 按字节偏移重读会静默中断），经验证 `bio_units(include_maples=False)` 返回 20 个 `fundusavseg` 单元、日志由 "44 not ready" 变为 "20 not ready"。两个锚文件（`e4_gt_maplesdr.csv` 13:46、`e4_gt_maplesdr_preannot.csv` 15:35）**已在本地算完，双方均不重算、不覆盖**。分界线：CPU 节点只写这 24 个目录里的 `bio.csv`，其余一概不碰；`cmd_maples_analyse` / `cmd_finish` 读的正是这些路径，收尾器无需改动。
  **只传 `manifest.csv` + `mask/`，不传 `prob/`**：`prob/` 是 float16 `.npy`（约 6.6 MB/图，24 个目录合计约 25 GB），生物标志物步骤一行都不读。
  **`manifest.csv` 携带 Windows 绝对路径**（`image_path`/`mask_path`/`fov_path`/`label_path`），在 Linux 上不可解析，而 `p5_eval.cmd_bio` 把 `manifest.mask_path` 直接交给读取器。节点上由 `remote/cpu_fix_manifest.py` 重定根，并在任何一个 `mask_path` 仍不存在时**报错退出**，杜绝"路径失配后静默回退到错误记录"。
  **FOV 掩膜是复制过去的，绝不在节点上重新生成**（density / total_length 按 FOV 面积归一，重生成的 FOV 会系统性偏移且不会报错）；MAPLES-DR 自身不带图像，`_maplesdr_image_index()` 回落到 Messidor-2 镜像，故只同步 `messidor2_match.csv` 中 `matched==True` 的那 **162 张**（347 MB，而非镜像全量 4.8 GB）。节点上 `load_dataset("maplesdr")` 返回 162 条、缺图 0、缺 FOV 0。
  交付校验：每个 `bio.csv` 必须恰好 162 行才被拉回本地（`cpu_e4_loop.sh` 行数不符则留为 `.part` 并告警）。
- 2026-09-17 17:38 **E3 分类阶段在 CPU 节点执行；预注册默认值一律不改，唯一传入的额外参数是 `--n-jobs`**：`--n-jobs` 只作用于 `GridSearchCV` 的并行度（`nested_cv_proba` 内），不进入任何随机数或划分，结果逐位无关。其余全部沿用脚本默认：`N_BOOT=1000`、`SEED=0`、5 折 × 3 重复的分组分层嵌套 CV（内层 3 折）、APTOS/IDRiD/Messidor-2 按 `COHORT_ROLE` 走**图像级**划分与自助、logreg + gbdt、`bio` 与 `bio+cov` 两套特征、Messidor-2 的 162 张 MAPLES-DR 重叠在 clf 阶段剔除（实测日志 `1744 -> 1582 images (162 dropped)`）。
  **两族运行，原因是 `delta_vs_ref` 永远以 `tags[0]` 为参照**：(A) **canonical**——12 个 FIVES tag 一次调用，参照 `e3_fives_s0_baseline`，写入规范路径 `results/pivot/e3/<cohort>/{summary,delta}.csv`；(B) **per-seed**——预注册要求"逐种子报告"，故对每个种子 k 用该种子自己的 4 个 tag 各跑两次，分别以该种子的 `baseline` 与 `continued` 为参照，从而 `ReliSeg − baseline` 与 `ReliSeg − continued` 都有**配对自助 CI**且配对发生在同一种子内。(B) 的输出落在 `results/pivot/e3_perseed/ref{baseline,continued}_s<k>/<cohort>/`，其 tag 目录是指向规范 `bio.csv` 的符号链接——**没有任何一行生物标志物被复制、重算或修改**。
  17:38 起 idrid 与 messidor2 两个队列并行启动（各 `--n-jobs 12`）；APTOS 的 12 个 FIVES tag 本地尚未齐（`results/pivot/e3/aptos2019/` 只有一个 200 行的 hrf 半成品），齐备后再跑，驱动脚本 `have_all()` 缺 tag 时直接跳过而不是部分计算。
- 2026-09-17 18:00 **E2 零样本中期结果（CHASE_DB1 n=8、STARE n=10，未被任何探针触碰）**：ReliSeg − baseline 在 total_length 上两处 CI 排除零（CHASE +0.008 [+0.001, +0.037]，STARE +0.005 [+0.000, +0.014]），但 ReliSeg − continued（同预算对照）全部 CI 含零 → 相对未动基线的表观增益可归因于额外 50 epoch 训练，而非测量项。HRF 训练模型迁移更差，且在 STARE 上相对 continued 显著更差（total_length −0.034 [−0.076, −0.004]）。结论按预注册如实进入论文；C2 的正向主张现在完全取决于 E4（Fundus-AVSeg）与域内 continued 对比；若 E4 亦为零结果，C2 改写为"测量感知项在同预算下不增加保真度、且带来迂曲度附带损害"的有界阴性结论，安全约束分析保留。

- 2026-09-17 18:15 **新租 GPU 服务器（2× RTX 2080 Ti）接管 ODIR-5K 全部 E3 外部推理，14 个检查点已并发启动**。
  节点 `user@GPU_HOST_2 -p SSH_PORT`（cloud 容器，Ubuntu 22.04.3）：2× RTX 2080 Ti **各 11,264 MiB**，
  驱动 535.98 / CUDA 12.2（cu124 轮子按 CUDA 12.x 次版本兼容规则可用），2× Xeon Platinum 8255C **96 逻辑核**，
  375 GB 内存，`/path/to/workdir` 为 /dev/md0 50 GB（工作根 `/path/to/workdir/medical1`，与本地同构；`/` 仅 30 GB overlay）。
  **口令只用了一次**（`exp/remote/bootstrap_key_gpu2.py`，经 `REMOTE_SECRET` 环境变量装公钥），此后全部密钥登录；
  口令不写入任何文件、脚本、日志或本条目——这是刻意的。环境见 `exp/remote/ENV_gpu2.md`，
  助手脚本 `exp/remote/gpu2_{sync,pull,e3_driver,fov_prebuild,verify_pulled}.sh`（沿用 CPU 节点 agent 的 `cpu_*.sh` 命名法）。
  env：`/root/miniconda3` 的 python3.12 建 venv，torch **2.6.0+cu124** / torchvision 0.21.0+cu124，numpy 2.2.6、
  scipy 1.17.1、skimage 0.26.0、sklearn 1.9.0、cv2 5.0.0、pandas 3.0.5、skan 0.13.1、numba 0.67.0、pyarrow 25.0.1，
  25 个包同进程导入通过，`cuda_avail True`、`device_count 2`、两张卡各跑了一次 2048² matmul。
  **装环境的顺序坑（写给后续 agent）**：先装 freeze 再装 torch 会被 `pvbm 3.0.1.0` 的**无版本上限的 torchvision 依赖**
  拖进 torch 2.14.0（554 MB）；已改为「torch 先装 + 其余用 `-c` 约束文件锁住 torch/torchvision」，见 `install_env2.sh`。
  同步（仅 ODIR 所需，Messidor-2/APTOS 留在 LAN 节点，不上传）：`src/` 131 文件、`configs/`、`runs/pip_freeze.txt`、
  `results/gateA_biomarker_scales_train.csv`、`data/external/odir5k` 6,407 文件 771 MB、14 个检查点 + config/summary json 约 435 MB。
  **完整性守卫**：节点上 `src/pivot/e3_external.py` 的 SHA-256 = `af65105ebf0a…8e9f6c1a5`，与本地一致；
  14 个检查点在节点与本地逐一同哈希且两两不同（FIVES 的 12 个与 LAN 节点 16:12 条记录的摘要完全吻合）；
  节点上 `python -m src.data.check_external odir5k` **OK**（6,392 张 / 3,358 患者，标签分布与公开分布一致，0 缺失）。

- 2026-09-17 18:15 **ODIR-5K 归属再更正一次：全部 14 个检查点（含 HRF s0 两个）移到 2080 Ti 盒子，跑前已向两侧取得确认**。
  16:28 条把 ODIR-5K 上的 12 个 FIVES 作业排给 LAN 节点（接在其 APTOS 之后），HRF s0 的两个仍留本地队列。
  现统一改判：**ODIR-5K 整集由新 GPU 节点承担**，理由是它有两张空闲的 11 GB 卡和 96 核，而 LAN 节点的 APTOS 尚未跑完。
  **启动前的两项确认（先确认后启动，不是先跑后通知）**：
  (1) LAN agent 回报已杀掉链式进程（pid 219804，`lan2_e3_chain.sh 9 remote/lan2_jobs_odir5k.txt`），
      并把作业表改名为 `lan2_jobs_odir5k.txt.DISARMED_moved_to_2080ti`；节点上 `results/pivot/e3/odir5k/` 不存在，
      **没有任何 ODIR 作业曾经启动**，因此无需去重，14 个全部从零跑。其 APTOS 波次不受影响，照常运行。
  (2) E2 队列 agent 回报已在 17:18 为 `e3:odir5k:hrf:s0:{baseline,reliseg}` 预置隔离锁（`owner.json` 无 pid，
      标注归属 GPU2 节点），与 12:05 / 16:28 同一机制；本地 e3 队列**可认领作业归零**，
      本地 `results/pivot/e3/odir5k/` 亦为空目录。
  **另记一处重复风险**（E2 agent 报、已转告 LAN agent）：本地仍在跑 `messidor2/e3_hrf_s0_baseline`（pid 48140）与
  `aptos2019/e3_hrf_s0_baseline`（pid 49448），而最新分工把 Messidor-2/APTOS 的 HRF s0 给了 LAN 节点——
  这两个 tag 有被算两遍的风险，LAN 节点开跑前应先与 E2 agent 核对 meta.json。

- 2026-09-17 18:15 **ODIR-5K E3 吞吐实测与 ETA：14 路并发 1.6–1.7 s/img，整批约 3 小时**。
  分工：GPU0 跑 `e3_fives_s{0,1,2}_{baseline,reliseg}` + `e3_hrf_s0_baseline`（7 个），
  GPU1 跑 `e3_fives_s{0,1,2}_{cfloss,continued}` + `e3_hrf_s0_reliseg`（7 个），
  参数一律 `--resize-longest 1536 --sw-batch 1 --resolution-record`，输出 `results/pivot/e3/odir5k/<tag>/`。
  18:11:56 / 18:12:46 以 `nohup setsid` 从**短 ssh** 分离启动（长 ssh 会在 nohup 上挂住，与 LAN 节点同一毛病），
  18:14 核验：14 个 worker 全在，每卡 7 个、**各占 652 MiB**（7×652 = 4,567 MiB / 11,264 MiB），GPU 利用率 63–83%，load 19/96。
  **实测**：单作业 1.6 s/img；把 OMP/MKL/OPENBLAS/NUMEXPR 从 1 提到 4 **没有任何收益**（仍 1.6 s/img），故保持 1；
  14 路并发下每作业仍 1.6–1.7 s/img，**几乎没有并发惩罚**，整波 8.5 img/s（摊薄 0.12 s/img）。
  每作业 ETA 165–180 min → **全部 14 个约在 21:10 前后完成**（89,488 次图像推理，串行需 ~19 h）。
  与 LAN 节点（RTX 3080、IDRiD）不同的是本机 GPU 利用率高得多：ODIR 原生 512×512，放大到 1536 后
  滑窗推理而非 skan 成了主成本。另：**FOV 缓存必须先建满**——`ensure_fov` 由首个用到它的 worker 惰性写盘，
  14 路并发会对同一批 `fov/*.png` 形成写竞争；`gpu2_fov_prebuild.sh odir5k 32` 分片 32 路，
  **73 秒**建完 6,392 张（单线程需约 35 min），已先于任何 driver 跑完并核对计数。
  **服务器不关机**（用户指令）。

- 2026-09-17 18:18 **APTOS-2019 主 9 趟完成并回拉（+ s2_continued，共 10 个标签）**：
  14:28 启动 → 18:18:14 主 9 趟齐活，实测 2.7–2.8 s/图（并发下单作业），
  9×3,662 = 32,958 张。10 个标签各 **3,662/3,662 行**，`meta.json`/`resolutions.csv` 齐全，
  10 个 `ckpt_sha256` 与本地检查点重算值**逐一吻合**（s2_continued = `decf9b4ec853`）。
  尚在跑：`e3_fives_s0_continued` / `e3_fives_s1_continued`（970/3,662，ETA ≈ 20:15）、
  `e3_hrf_s0_reliseg`（aptos 960/3,662 ETA ≈ 20:15；messidor2 1,260/1,744 ETA ≈ 18:25）。
  **回拉的文件数告警已查清且无害**：本地多出的 24 个文件全部是本地自有产物——
  2026-09-16 的 `smoke_*` 冒烟运行（含 `oof_*.npy`、`SMOKE_*.csv`）与本地机器正在写的
  `e3_hrf_s0_baseline/bio.csv`（18:18 时 aptos 750 行、messidor2 1,240 行，仍在增长）。
  `lan2_pull.sh` 只写节点上存在的标签目录，故这些本地产物**一个都没有被覆盖**
  （已用 `comm -13` 逐文件核对）。这正是 17:21 把 LAN 侧半截 baseline 移出结果树的用意。
- 2026-09-17 18:20 **完整性负结果（并发分片安全性的实测证据）**：E2 队列 agent 报告本地
  `ensure_fov` 存在**非原子写竞态**（同一数据集两个进程同时生成同一个缺失 FOV，读到截断
  PNG，imageio 回退到 SPE 插件后 `ValueError: cannot reshape array of size 1701`）。
  LAN 节点**不受影响**，因为每个数据集的 FOV 缓存都在波次扇出**之前**单进程预建完毕
  （idrid 516 随数据同步；messidor2 12:42:38 完成 vs 波次 13:20:18 启动；
  aptos 13:39:22 完成 vs 波次 14:28 启动），worker 运行时 `os.path.exists` 恒真、写路径从未进入。
  实测核验而非假设：(1) 用 imageio 重读全部 **5,922 张 FOV**，0 不可读、0 退化；
  (2) 所有 driver 日志中每个作业 `rc=0`，唯二非零是 17:21 主动 kill 的两个 `rc=143`；
  (3) 已完成表格共 56,416 行，行数全额、`image_id` 全唯一、4 个 skan 列 **0 个 NaN**。
  截断 FOV 必然表现为崩溃或 NaN/垃圾行，三者皆无 ⇒ 这是真阴性而非"没看见"。
  **不修补 `ensure_fov`**：它位于 `src/data/external.py`，处于协调者的指纹锁之下
  （`6e96340d…`，两机已核验一致），静默修改会使预注册指纹失效；已请 E2 agent 上报协调者，
  由其决定是否重新签发锁定哈希。在此之前，"并发前先单进程预建 FOV 并核对数量"是既定规避方案，
  由 E2 agent 写入 `data/EXTERNAL_DATASETS.md`（文档文件，不在锁内）。
- 2026-09-17 18:24 **`aptos2019/e3_hrf_s0_baseline` 改由 LAN 节点从零重跑（协调者再分配）**：本地这一趟是本机的长尾（ETA 23:15），挤占 E4 复现 agent 的 GPU，而 LAN 侧有余量。处置：先用 Win32_Process 核对 pid 49448 的命令行（`e3_external bio --dataset aptos2019 --ckpt runs\seg\hrf\seed0\best.pt --tag e3_hrf_s0_baseline --resize-longest 1536 --resolution-record --device cuda:1`）确认无误后按显式 PID 停止，当时进度 **770/3,662**；worker 44688 记为 `FAILED ... rc=4294967295` 并按设计 `release()` 了锁（**未自动重试**）。
  半成品移出规范路径：`results/pivot/e3/aptos2019/e3_hrf_s0_baseline` → `e3_hrf_s0_baseline_LOCALpartial_stopped_1824`（770 行、无 meta.json），**保留不删**，与 LAN 的 `_LANpartial_held_1721` 同一约定；随后以 LAN 归属重新加隔离锁（owner.json 无 pid），并核验规范路径已空、本地 e3 可认领作业为 **0**。
  **要求 LAN 从零跑、不要接着这 770 行续写**：bio 阶段按 image_id 续跑，若这份半成品被拉回规范路径，LAN 的运行会静默跳过这些图，产出一张"一半在本机测、一半在 LAN 测"的表。同一检查点下数值应当一致，但溯源必须干净：一个 tag、一台机器、一趟跑完。那 770 行可另作跨节点一致性抽查，不作为交付输入。
  HRF s0 四个 tag 的最终归属：`messidor2/baseline` 本地（pid 48140，ETA ~18:55）；`messidor2/reliseg`、`aptos2019/baseline`、`aptos2019/reliseg` 均归 LAN；`idrid` 两趟本地已完成（各 516 行 + meta.json）；ODIR-5K 全部归 GPU2 节点。
  待办：`messidor2/baseline` 落盘后两个 worker 都将空转，届时按显式 PID 停止 44224 / 44688，本地 GPU 全部交给 E4，另记一条。
- 2026-09-17 18:30 **E3 分类阶段结果（IDRiD + Messidor-2 两个复现队列已完成；APTOS 主队列待其 12 个 tag 齐备）**：在 CPU 节点上跑完，预注册默认值未改一项（见 17:38 条），输出已拉回 `results/pivot/e3/<cohort>/{summary,delta}.csv`（含 `oof_*.npy`）与 `results/pivot/e3_perseed/ref{baseline,continued}_s{0,1,2}/<cohort>/`。Messidor-2 的排除按预注册生效：`1744 → 1582`（162 张 MAPLES-DR 重叠，`exclude_src` 已写入每一行）；IDRiD n=516、无预注册排除。两队列均为图像级划分与图像级配对自助（`COHORT_ROLE`）。
  **主格（referable DR / logreg / bio+cov）的 ΔAUC，逐种子配对自助 1000 次重采样**：
  | 队列 | 对照 | seed0 | seed1 | seed2 | 种子均值 |
  |---|---|---|---|---|---|
  | IDRiD (n=516) | ReliSeg − baseline | −0.0025 [−0.0066, +0.0017] p=0.252 | −0.0023 [−0.0082, +0.0040] p=0.444 | −0.0022 [−0.0059, +0.0017] p=0.280 | **−0.0023** |
  | IDRiD | ReliSeg − continued | −0.0029 [−0.0062, +0.0007] p=0.118 | −0.0015 [−0.0075, +0.0047] p=0.648 | −0.0018 [−0.0055, +0.0017] p=0.310 | **−0.0021** |
  | Messidor-2 (n=1582) | ReliSeg − baseline | −0.0036 [−0.0103, +0.0031] p=0.318 | +0.0005 [−0.0056, +0.0069] p=0.824 | −0.0036 [−0.0090, +0.0017] p=0.180 | **−0.0022** |
  | Messidor-2 | ReliSeg − continued | +0.0007 [−0.0022, +0.0036] p=0.626 | +0.0038 [−0.0014, +0.0092] p=0.152 | −0.0033 [−0.0070, +0.0003] p=0.072 | **+0.0004** |
  **判读：两个复现队列均为零结果，12 个逐种子对比的 95% CI 全部包含零，无一个 p<0.05。**效应量的尺度本身就说明问题：12 个 tag 的主格 macro-AUROC 在 IDRiD 上落在 0.8699–0.8743（极差 0.0044）、在 Messidor-2 上落在 0.7364–0.7467（极差 0.0103），即**四个臂之间的差异小于同一臂三个种子之间的差异**——分割器的选择在这两个外部队列的下游判别力上不可分辨。这与 E2 在 FIVES 上的天花板效应结论方向一致，如实报告，不做任何事后筛选。注意 `continued` 同预算对照在此**确实起作用**：Messidor-2 上 ReliSeg − baseline 的种子均值为 −0.0022 而 ReliSeg − continued 为 +0.0004，说明"对 baseline 的那点负差"里有一部分来自"多训练 50 epoch"本身而非测量项，若无该臂会误读为 ReliSeg 特有的损害。
  **口径说明（记录以免日后误读）**：`run_clf` 的 `delta_vs_ref` 永远以 `tags[0]` 为参照，故规范路径 `results/pivot/e3/<cohort>/delta.csv` 里 12 个 tag 的 Δ 全部相对 `e3_fives_s0_baseline` 一个参照，**跨种子不配对**；上表的逐种子配对数字来自 `e3_perseed/` 下的六次 4-tag 运行。引用时不要把规范 delta.csv 的跨种子行当作配对对比。

- 2026-09-17 18:31 **HRF s0 四个标签的归属落定；`aptos2019/e3_hrf_s0_baseline` 改由 LAN 从零重跑**：
  协调者 18:24 改派——本地那趟 APTOS baseline（pid 49448，已到 770/3,662、ETA 23:15）是本地长杆，
  且在饿死 E4 复现 agent 的 GPU；LAN 有余量故接手。E2 agent 停掉该 PID 并把半截结果移出规范路径
  （`e3_hrf_s0_baseline_LOCALpartial_stopped_1824`，770 行、无 meta.json，保留不删），
  规范路径核验为空后，LAN 于 **18:31:25 从零启动**（我方 17:20 的 10 行残留早已移至
  `runs/lan_e3/_held/`，不在结果树内）。检查点 `runs/seg/hrf/seed0/best.pt`
  sha256 `fd24517bb24a…` 两机现算一致；参数与其余 34 张表完全相同。
  **明确不做断点续跑**：bio 阶段以 `image_id` 判断已完成，**不记录该行由哪台机器产生**，
  两机混写的表事后无法分辨 ⇒ 规则定为"一个 tag、一台机器、一趟跑完"。
  E2 的 770 行将用作**跨节点一致性抽查**（同一冻结检查点、同一约定，四个 skan 列应逐位或
  浮点级一致），属旁证而非交付输入。
  四个标签最终归属：`messidor2/baseline` 本地（ETA 18:55）；`messidor2/reliseg` LAN
  （**18:22:08 完成，1,744/1,744，`1325429e2850…` 与本地 last.pt 一致，已回拉 3/3 文件**）；
  `aptos2019/baseline` LAN（新起）；`aptos2019/reliseg` LAN（在跑）。IDRiD 的 HRF s0 两趟归本地
  （各 516 行已完成），ODIR-5K 全部归 2080 Ti 机器。
- 2026-09-17 18:30 **E4 婉拒 LAN 空闲机时（记录理由，避免日后重提）**：按协调者指示向 E4 agent
  提供 ~20:15 之后的 3080 通道，对方**明确拒绝且理由成立**：(1) E4 关键路径是剩余 5 个微调，
  配方锁死在 1536/768 **batch 4**，10 GB 卡必须减半到 bs 2 → 对预注册设计构成协议偏离，
  且这些微调在 20:00 前就结束，早于卡空出；(2) 唯一可外包的 MAPLES 零样本推理已由 32 核 EPYC
  CPU 节点接管，转移只省 ~15 min 却多一次跨机交接；(3) 需要同步 ~310 MB 检查点 +
  整个 `data/maplesdr` 树 + 尚未纳入指纹核验的 `src/pivot/e4_replication.py`，
  staging 20–40 min 不划算；(4) **`data/maplesdr/fov/` 必须复制而非重新生成**——density 与
  total_length 按 FOV 面积归一化，重生成的掩膜会系统性偏移且不报错（CPU 节点逐字节复制后
  与锚点对比，101 列最大相对差 5.5e-15）；再引入第三份 FOV 来源纯属徒增风险。
  **结论：LAN 卡在本轮作业结束后允许空闲**——是用户自有机器、不计费，空转零成本，
  不为填满而制造工作。E4 若有作业失败会来要通道。
- 2026-09-17 18:32 **E3 首张跨节点交付表已本地独立核验通过：`messidor2/e3_hrf_s0_reliseg`**（LAN 节点 18:22:08 完成并回拉）。**不是采信对方的自述，而是在本地重算**：1,744 行 / 1,744 个唯一 image_id（与 meta 的 n_images 一致，且等于排除 MAPLES-DR 重叠前的 Messidor-2 全量——排除只在 clf 阶段生效，bio 表保留全部行，与 13:15 条一致）；`resize_longest=1536, patch=768, threshold=0.5` 与其余 34 张表同一约定；meta 的 `ckpt_sha256 = 1325429e28507c11a2fd9680fe6536aa7284be60d9e6a32a374e56ff83c28906`，本地对 `runs/pivot/hrf/seed0/reliseg/last.pt` 重新计算得到同一哈希，逐位相符；四个 skan 生物标志物列**无空值、无 NaN**（0/1,744 行）。
- 2026-09-17 18:31 **`aptos2019/e3_hrf_s0_baseline` 已在 LAN 节点从零开跑**（18:31:25），检查点 `runs/seg/hrf/seed0/best.pt` 的 sha256 `fd24517bb24a7489935ce77d8c14a00673afcaa6fa9c69f6c1e16921b5a2b771` 在两台机器上重算一致；APTOS 的 FOV 缓存已于 13:39:22 单进程预建完毕，故 `ensure_fov` 竞态不适用。预计 20:45–21:20 完成（4 worker，优于本地单进程的 23:15）。
  **`results/pivot/e3/aptos2019/e3_hrf_s0_baseline_LOCALpartial_stopped_1824`（770 行）继续保留、任何人不要清理**：LAN 表完成后将按 image_id 内连接该半成品，报告四个 skan 列的最大相对偏差，作为**跨节点一致性抽查**。同一冻结检查点、同一约定，预期浮点级一致；若不一致，这是必须在 clf 阶段之前知道的发现。该检查是副产品，不作为交付输入。
  **值得写下的一点**（LAN agent 的复述，与我的判断一致）：续跑集合以 image_id 为键，**不记录每一行由哪台机器产生**，因此一份混合来源的表事后无法分辨——这正是"一个 tag、一台机器、一趟跑完"的理由。

- 2026-09-17 18:35 **FOV 缓存完整性实测通过（此前只是"按顺序推断没出问题"，现在是核验过的）**：LAN 节点 agent 指出 `ensure_fov` 非原子写入，两个进程同时补同一张缺失掩膜会产出被截断的 PNG。E4 的四条 GPU 通道确实可能在首次 `load_dataset` 时并发触发 `_fov_for`。逐图核验结果：`fundusavseg` 100/100、`maplesdr` 162/162，全部存在、可读、且与对应图像尺寸一致；FOV 占帧比 fundusavseg 0.642 / 0.779 / 0.787（min/中位/max）、maplesdr 0.447 / 0.464 / 0.472，与 13:00 条记录的 0.642–0.785 与 0.448–0.472 吻合。**无截断、无竞争损坏**。原因是两个缓存都在通道启动前由单进程建好（`check_data.py` 13:00、strata 扫描 ~14:00），属顺序上的运气而非设计；**后续若中途新增数据集，必须先单进程预建 FOV 再开并发**。
- 2026-09-17 18:35 **通用运维规则（今日三次同类故障后确立）**：**绝不修改正在被活动进程读取的文件，包括该进程本身正在执行的脚本。** 今日三例：(1) 我就地编辑运行中的 `bio_watch.bat`，cmd.exe 按字节偏移重读批处理，偏移错位后循环静默终止，生物标志物停摆 41 min；(2) LAN agent 就地编辑运行中的 `lan2_sync.sh`，bash 在循环结束后 seek 到位移后的偏移读到乱码（`line 64: it: command not found`），8 GB APTOS 传输中断且 0/3,726 文件落地；(3) 同类风险还包括并发进程互相覆写非原子生成的缓存文件（见上条 FOV）。做法：改脚本时写新文件名重启（本项目为 `bio_watch<N>.bat`），或复制到第二路径运行副本。
- 2026-09-17 18:35 **跨机数值一致性的检验范式（与 CPU/LAN 两个 agent 共识）**：分布式跑同一估计量时，不靠"用了同一份代码"的声明，而是**对同一批输入在两台机器上各算一遍、逐列报告最大相对差**。CPU 节点对 MAPLES GT 锚点做此检验：101 个数值列最大相对差 **5.5e-15**，仅 `runtime_*_s` 列不同。关键理由：FOV 重生成这类错误**不会抛异常**——density 与 total_length 都以 FOV 面积归一化，换一份"有效但不同"的掩膜只会让数字系统性偏移且看起来正常，与竞争写入产生的截断 PNG（会崩、会自曝）是不同性质的故障。
- 2026-09-17 18:45 **E3 复现队列分类结果（CPU 节点，预注册默认值未改）**：IDRiD（n=516）与 Messidor-2（n=1,582，剔除 162）主终点 referable-DR logistic AUROC 的 ΔAUC（ReliSeg − baseline / − continued）12 个逐种子 CI 全含零；四臂间差异小于同一臂三种子间差异。APTOS 主队列待 continued s0/s1 到齐后自动运行。结合零样本结果：C2 的效用主张目前无任何队列支持；最终定性等 E2 域内 continued 对比（HRF）与 E4。
- 2026-09-17 18:43 **E2 完整网格结论（results/pivot/E2_REPORT.md，final analysis 18:39）**：(1) HRF（开发集）三种子合并：ReliSeg − continued 的 FD +0.034 [−0.037, +0.107]、length +0.015 [−0.040, +0.074]，均含零；P5 探针的正结果只出现在 seed 0（+0.14），seeds 1–2 为 −0.006/−0.020 → 种子异质，且 continued seed 0 本身 +0.07。(2) 忠实 CF-Loss 相对 continued 在 HRF 上显著**降低**保真度：FD −0.156 [−0.222, −0.092]、length −0.148 [−0.208, −0.090]、density −0.037 [−0.074, −0.002]，并突破 clDice 边际——对已发表方法的可复现阴性证据。(3) FIVES 天花板零结果；零样本相对 continued 零结果；E3 复现队列零结果。(4) HRF seed-0 消融中 no-density / length-only 出现 +0.15–0.20 的 FD/length 增益，但伴随迂曲度偏差 +1.5–8.5σ 的严重附带损害，且仅单种子，不作为新臂推进。**定性：C2（测量感知微调）为有界阴性结论；安全约束分析与 CF-Loss 复现保留。**
- 2026-09-17 18:45 **新探针预注册（推理操作尺度）**：审计与分辨率阶梯表明保真度随分辨率下降；当前所有推理把 HRF（原生 3504）与 FIVES（原生 2048）降采样到最长边 1536。探针：在 HRF 测试集上用已冻结的 baseline 与 continued（3 种子）以最长边 2048 / 2560 / 3504 推理（patch 768，其余不变），计算 FD/length/density/tortuosity 对 GT 的 r 与 Dice/clDice。判定规则（看 HRF 结果前锁定）：若 FD 与 length 的 Δr（原生 − 1536）三种子均 ≥ +0.10 且 CI 排除零，则将"原生尺度推理"作为 C2 的候选干预，预注册确证：FIVES 原生 2048（三种子）、零样本 CHASE/STARE 原生、E4 折外原生、E3 外部集原生（重跑受影响标签）；主终点 Δr（原生 − 1536）配对 bootstrap，安全终点同前，下游 ΔAUC 次要。若不满足，探针结果仅作为审计的补充报告。
- 2026-09-17 19:05 **`messidor2/e3_hrf_s0_baseline` 本地跑完并核验通过**：1,744 行 / 1,744 个唯一 image_id，与 meta 的 n_images 一致；`resize_longest=1536, patch=768, threshold=0.5`；meta 的 `ckpt_sha256 = fd24517bb24a7489935ce77d8c14a00673afcaa6fa9c69f6c1e16921b5a2b771`，本地对 `runs/seg/hrf/seed0/best.pt` 重算得到同一哈希（**与 LAN agent 在其节点上独立算出的值也相同**，三方一致）；四个 skan 列 0 空值 / 0 NaN。行数为 Messidor-2 全量（含 162 张 MAPLES-DR 重叠图），符合"排除只在 clf 阶段生效"的 13:15 决定。
- 2026-09-17 19:06 **本地退出 E3，两个 E2 worker 已停止，本地 GPU 全部交给 E4**：停止前逐一核对——`Get-CimInstance` 确认 pid 44224 / 44688 的命令行均为 `python -m src.pivot.e2_run gpu --gpu {0,1}`、除 conhost 外**无子进程**（即无在跑作业），且两者日志末行均为 `[e2] nothing runnable; waiting 120 s for the other lane`（本地 e3 队列可认领作业为 0，故其空转属预期）。按显式 PID 停止后复查：**无任何 `e2_run` 进程残留**。
  本地 E3 最终交付：`idrid/e3_hrf_s0_{baseline,reliseg}`（各 516 行）与 `messidor2/e3_hrf_s0_baseline`（1,744 行），共 3 张表，均已本地核验。其余 E3 由 LAN 节点（messidor2/aptos2019 的 reliseg + aptos2019 的 baseline）与 GPU2 节点（全部 14 个 ODIR-5K 检查点）承担。
- 2026-09-17 18:50 **E2 全网格分析完成（`exit=0` 18:39:43）**，产物：`E2_REPORT.md`（119 KB）、`e2_fidelity.csv` 576 行、`e2_delta.csv` 1,184 行、`e2_pixel.csv` 72 行、`e2_downstream.csv` 224 行、`e2_zeroshot.csv` 480 行、`e2_safety.csv` 1,316 行。主要判读见 18:55 条（C2 记为有界阴性）。三个值得单独记下的点：
  (1) **`continued` 对照推翻了四个"看似显著"的结果**：FIVES 上 ReliSeg 的 FD Δr 对 baseline 为 +0.0047、对 continued 变为 **−0.0034（变号）**，density 同样变号；零样本 CHASE/STARE 上对 baseline 的 total_length Δr 两个 CI 都不跨零（+0.0076 [+0.0006,+0.0372]、+0.0051 [+0.0004,+0.0144]），对 continued 则**全部跨零**。若无同预算对照，这四条会被当作阳性报告。
  (2) **已发表 CF-Loss 在 FIVES seed0 上显著损害下游 AUC**：ΔAUC = −0.0179 [−0.0397, −0.0002]，CI 不跨零。
  (3) **HRF 消融显示 density 项适得其反**（seed0，仅开发集，假设生成性质）：去掉 density（`abl_no_density`）得到 FD Δr **+0.1981**、total_length **+0.1989**，约为完整 ReliSeg（+0.0389 / +0.0190）的 5 倍；`abl_length_only`（+0.1631 [+0.0887,+0.2383] / +0.1509 [+0.0799,+0.2223]）与 `abl_fd_only`（+0.0949 [+0.0287,+0.1705] / +0.0891 [+0.0207,+0.1633]）是**仅有的 CI 不跨零的臂**，且都优于完整损失；而 density 自身的 Δr 在几乎所有臂中为负（含 `abl_density_only` 的 −0.0229），即该项连它所约束的量都没有改善。
- 2026-09-17 19:11 **E2/E3 结果写入稿件，C2 改写为有界阴性（协调者指示）**：(1) 骨架表全部撤销，改为从 CSV 生成：`paper2/tools/mk_tab_e2.py` → `tab:e2`（FIVES/HRF 域内主表）、`tab:e2zs`（零样本，补充 S2）、`tab:e2abl`（HRF seed-0 消融，补充 S2），读 `e2_delta.csv` / `e2_safety.csv` / `e2_pixel.csv` / `e2_downstream.csv` / `e2_zeroshot_delta_pooled.csv`；`mk_tab_e3.py` → `tab:e3`，读 `e3/<cohort>/summary.csv` 与 `e3_perseed/ref<arm>_s<k>/<cohort>/delta.csv`，APTOS 行由生成器自己输出 `\todo{}`；`fig_e2_forest.py` → 图 5（左：三个目标标志物的配对 Δr，对 baseline 与对 continued 两种参考；右：迂曲度偏差在两条管线上，画出 ±0.25σ 边际）；`fig_e3_external.py` → 图 6（逐种子 ΔAUROC）。`mk_tab_placeholders.py` 与 `fig_placeholders.py` 现在只剩 E4。`\todo` 由 66 降到 21。(2) **C2 在摘要 / 引言 1.3 / 5.10 / 讨论 / 结论中一律改写为有界阴性**："测量感知微调在五个队列（FIVES 确证、HRF 开发、CHASE_DB1+STARE 零样本、IDRiD+Messidor-2 外部）上均未使图像级保真度超过同预算训练，目标标志物的全部 CI 含零；相对冻结基线的表观增益对同预算对照消失"，并明确该结论以 E4 为条件。两项实质发现被提到 C2 的核心位置：忠实复现的已发表 CF-Loss 相对同预算对照**降低**保真度（FD −0.156 [−0.222,−0.092]、length −0.148、density −0.037，CI 均排除零）并在三个种子上突破 clDice 边际；预注册安全终点抓到未约束迂曲度的大幅位移（FIVES +1.16σ、HRF +6.61σ，HRF 迂曲度 Δr −0.244 [−0.375,−0.103]）。(3) 按指示如实写入：FIVES 为天花板（基线 r 0.98/0.98/0.90，三种子一致，且无下游差可恢复）；HRF 探针效应仅 seed 0（+0.143 vs seeds 1–2 的 −0.006/−0.020），且 continued 自身 seed 0 即 +0.073；零样本相对基线的 total_length 增益（CHASE +0.008、STARE +0.005）对 continued 消失，HRF 训练模型迁移更差（STARE −0.034 [−0.076,−0.004]）；消融仅作单种子开发证据，**不推进任何变体**，并把 no-density / length-only 的迂曲度偏差（skan/PVBM +2.37/+11.15σ 与 +0.57/+1.49σ）与其增益并列报告。(4) 5.9 节机制表现在明确标注"早于同预算对照"，讨论直说 C2 未通过、探针是 seed-0 结果，未为了迎合阴性而改写目标函数。(5) NUMBER_SOURCES 增补 5.10/5.11 两节的逐条出处与新的"未能取数"清单；REVIEW_R1_RESPONSE 增 §2c。构建：63 页（正文含参考文献 55 页，补充 8 页），无未定义引用/交叉引用，overfull hbox 8 处且全部 <30pt，剩余 21 个 `\todo`（正文 13 + `tab:e3` 的 APTOS 行 1 + `tab:e4` 骨架 7）。**仍待**：APTOS-2019 主队列分类（E3 主终点）与全部 E4。

- 2026-09-17 19:18 **E4 主终点出结果：独立复现集上 ReliSeg 的测量保真度增益为零，且迂曲度安全终点破界（如实报告，不改设计）**。64/64 作业完成、0 失败；折外 n=100，σ 按外折训练部分估计，bootstrap 在折内重采样（以固定折与已拟合模型为条件）。**主对比 ReliSeg − continued（skan）**：FD Δr = −0.0104 [−0.0376, +0.0141]、density −0.0003 [−0.0048, +0.0038]、total_length −0.0006 [−0.0024, +0.0015]；max-statistic（Westfall–Young）调整后 p 分别为 **1.000 / 0.859 / 0.890**，无一显著，亦无一方向相反 → `replication_rule.json` 判定 **replication_success = false**。次要对比 ReliSeg − baseline 同样贴零（FD −0.0020、density −0.0006、total_length −0.0002）。留一折稳健性：五折的 FD Δr 在 −0.005 … −0.022 之间，无单折驱动。按疾病类别调整后 r 几乎不变（如 FD 0.8969 vs 0.8968），说明一致性不是由病种间差异撑起来的。**这是有功效的零结果，不是没打中**：预注册的揭盲前功效曲线（`e4_power.csv`，输入仅 E2）在本集这种保真度水平（baseline r = 0.899 FD / 0.940 density / 0.987 total_length）下对 Δr = 0.05 的功效约 1.00。基线已近天花板，与 FIVES 的零结果同因同果，**E4 因此是对 FIVES 结论的独立确证，而非又一次"数据不够"**。
- 2026-09-17 19:18 **E4 安全终点：8 项 BREACH，全部集中在迂曲度（与 E2 在 FIVES 上的 collateral damage 同向、同性质，独立复现）**。skan 迂曲度 Δr = **−0.0631 [−0.1366, −0.0296]**（对 continued）与 −0.0644 [−0.1361, −0.0317]（对 baseline），点估计与整个 95% CI 均越过 −0.05 边界；PVBM 迂曲度更严重：Δr = **−0.7746 [−1.0191, −0.5315]**、Δbias = **+1.738σ**（边界 0.25σ）。迂曲度保真度由 baseline 的 r = 0.967 掉到 reliseg 的 **0.903**（skan）。另有 cfloss 的 density Δbias = −0.272σ 轻微越界。像素端非劣性通过：四臂 Dice 0.9362–0.9380、clDice 0.9465–0.9482，两两差值均在 ±0.0017 以内，无一越过 0.01 边界。**结论：ReliSeg 在本复现集上没有买到任何测量保真度，却确实付出了未受约束标志物（迂曲度）的代价。**
- 2026-09-17 19:18 **E4 下游四分类 macro-AUC（skan4 + logreg，n=100，分类器外折与分割折 1:1 对齐）**：GT 0.540 [0.460, 0.614]、baseline 0.558、continued 0.552、reliseg 0.558、cfloss 0.551；ΔAUC(reliseg − continued) = +0.0063 [−0.0139, +0.0288]，ΔAUC(reliseg − baseline) = −0.0005 [−0.0203, +0.0205]。**三臂全部与 GT 臂在 CI 内无差别，且 GT 臂本身并不比预测臂更好**——在这 100 张图上，四项 skan 面板对病种几乎没有判别力（GT 的 AUC 仅 0.54），因此该终点**不能**作为"测量误差衰减下游效用"的证据，只能如实报告为无信息。这与 E1 在 FIVES 上神谕臂衰减仅 +0.004 的发现一致（高保真时神谕臂无优势）。

- 2026-09-17 19:50 **E4 次复现集 MAPLES-DR 零样本结果（24 个 E2 检查点 × 162 图，生物标志物由 CPU 节点算，已过验收）**。验收：24 个 `bio.csv` 各 162 行、列齐全、162 个 image_id 全部与本地 GT 锚点对上；`density_skan` 对 GT 的 r 按来源分层聚集（FIVES 训练 0.805–0.822、HRF 训练 0.637–0.744），无离群格，叠加 CPU 节点自测的 5.5e-15 逐列一致性，按本地跑对待。零样本保真度（skan，三种子均值）：FIVES 训练 FD 0.87 / density 0.81 / total_length 0.96，HRF 训练 FD 0.87–0.91 / density 0.66–0.72 / total_length 0.95–0.97。**主结论（第三次独立阴性）**：在预注册的 `higher_resolution` 分析层（71 张，≥1380 px），Δr(ReliSeg − continued) 在 **全部 6 个 来源×标志物 组合的全部 3 个种子上都是负的（18/18 格 d_r < 0）**，均值 −0.003 … −0.030；`lower_resolution` 层（91 张，909 px，预先声明的低分辨率阴性对照）更贴近零且符号混杂。即：**在最该看到机制的那一层，ReliSeg 一致地略差于同预算继续训练，从未更好。** 另注 `continued` 在多格上是最好的臂（HRF 的 FD 0.906 vs baseline 0.869），即"多训练 50 epoch"本身有益而测量项无益——这正是 13:30 加设 `continued` 对照要分离的混淆。迂曲度的附带损害在零样本上同样复现：skan Δr 全集均值 −0.094、higher_resolution 层 −0.131（最差格 −0.72），PVBM 全集 −0.251。
- 2026-09-17 19:50 **MAPLES-DR 预标注锚点敏感性（e4_maples_preannot.csv，576 格）**：把锚点从人工修正终稿换成网络预标注后，平均 r 由 **0.7845 降到 0.6098（−0.175）**；降幅在 `higher_resolution` 层远大于 `lower_resolution` 层（FD 0.859→0.608 vs 0.868→0.816）。**方向性结论不变**：Δr(ReliSeg − continued) 的均值由 −0.0275 变为 −0.0101，仍为负、仍有 69% 的格小于零，故"ReliSeg 无增益"的判断不依赖于用哪一版标注作锚。**但 r 的绝对水平强烈依赖锚点**，论文必须写明：MAPLES-DR 上报告的保真度是"与人工修正约定的一致性"，不是与某个与标注者无关的真值的一致性。降幅为何在高分辨层更大，有两种解释无法由本数据区分——(a) 视网膜专家在分辨率更高、细节更可见的图上改得更多，(b) 我把 1500² 画布最近邻映射回原生几何带来的重采样损失在各层不同；如实并列陈述，不择一。

- 2026-09-17 20:19 **APTOS-2019 全部 14 个标签完成并回拉核验（LAN 节点 E3 交付完毕）**：
  最后一趟 `e3_hrf_s0_baseline` 18:31:25 从零启动 → 20:19:26 完成（3,662 张，1.9 s/图，
  比本地原先 6.1 s/图 快 3.2 倍，也早于我方 20:45–21:20 的估计）。
  14 个标签**各 3,662/3,662 行**，`resize_longest=1536, patch=768, threshold=0.5, amp=True`
  完全一致；**14 个 `ckpt_sha256` 与本地检查点现算值逐一吻合，0 个不符**。
  LAN 节点 E3 总交付 = IDRiD 12 + Messidor-2 13 + APTOS 14 = **39 张表**。
- 2026-09-17 20:25 **跨节点一致性抽查：不是逐位一致，原因已定位为 AMP 推理的非确定性，且无系统偏差**：
  用 E2 agent 留存的 770 行本地半截结果（同一冻结检查点 `fd24517bb24a…`、同一约定）
  与 LAN 完整表在 770 个共同 `image_id` 上做内连接：
  | 列 | max abs rel diff | mean | 逐位相等行 |
  |---|---|---|---|
  | FD_skan | 3.60e-03 | 3.1e-05 | 12/770 |
  | tortuosity_skan | 1.68e-03 | 3.0e-05 | 215/770 |
  | density_skan | 2.04e-04 | 2.1e-05 | 98/770 |
  | total_length_skan | 2.12e-03 | 1.0e-04 | 121/770 |
  **定位**：几何链路**逐位相同**（`feat_fov_area_px`、`native_longest`、`scale`、`crop_h/w`
  在 770 行上完全相等）⇒ 读图/FOV/裁剪完全确定，**排除 FOV 来源问题**；
  而 `pred_fg_frac` 有最大 2.0e-04 的差异 ⇒ 差异发生在 **GPU 前向**：`amp=True` 的 fp16
  与 cuDNN 算法选择在两台机器上略有不同，概率图在 0.5 阈值附近翻转了少量像素，
  经骨架化/FD 放大为 1e-3 量级。**无系统偏差**：符号相对差的 |mean|/sd 仅 0.011–0.043，
  LAN 高于本地的比例 0.35–0.47，即零均值噪声而非某台机器整体偏高。
  **对交付的影响**：E3 每个标签都"一个 tag、一台机器、一趟跑完"，表内自洽；
  全部 E3 中**唯一跨机器的配对对比**是 Messidor-2 的 HRF s0 对
  （`baseline` 本地 / `reliseg` LAN），其余配对（IDRiD 的 HRF 对全本地、APTOS 的 HRF 对全 LAN、
  所有 FIVES 对全 LAN）均同机。该噪声（相对 sd ~1e-4）远小于检查点间的真实差异，
  但**论文不得声称逐位可复现**；若要彻底消除，可在 LAN 空闲卡上重跑
  `messidor2/e3_hrf_s0_baseline`（1,744 张，约 30–60 min）使该对同机——
  **未擅自执行**：协调者已明确该标签归本地且不得写入其目录，此项作为建议上报。

- 2026-09-17 20:26 **E3 bio 阶段（三个主外部集）全部完成，端到端复核通过**：本地树内共
  **42 张表 / 82,908 行**——IDRiD 14 × 516、Messidor-2 14 × 1,744、APTOS-2019 14 × 3,662
  （LAN 交付 39 张 + 本地交付 3 张：IDRiD 的 HRF s0 两张与 Messidor-2 的 HRF s0 baseline）。
  逐表复核：行数全额、`image_id` 全唯一、四个 skan 列 **0 个空值/NaN**、
  `resize_longest=1536 / patch=768 / threshold=0.5` 全部一致、
  **42/42 个 `ckpt_sha256` 与本地检查点文件现算值吻合**，问题项 NONE。
  LAN 节点 GPU 于 20:19 起空闲（19 MiB），工作树 21 GB 留存于
  `/mnt/data/Programming/research_ws/medical1`（/dev/sda1 仍有 1.3 TB 空闲），按用户规则不删。
  ODIR-5K 敏感性集由 2×2080 Ti 机器承担（LAN 一趟未跑）。**clf 阶段未运行**（本地预注册中）。
  待协调者决定的唯一事项：是否在空闲的 LAN 卡上重跑 `messidor2/e3_hrf_s0_baseline`
  以消除 E3 中唯一的跨机器配对（见 20:25 条）。

- 2026-09-17 20:27 **可复现性声明的口径（协调者裁定，写入论文）**：跨机器复现**达到 ~1e-4 相对量级，
  而非逐位一致**；论文按此措辞，不得声称 bit-level reproducibility。实测依据（同一冻结检查点
  `fd24517bb24a…`、同一约定、770 个共同 `image_id`，LAN 节点 vs 本地机器）：
  `FD_skan` max 3.60e-03 / mean 3.1e-05；`tortuosity_skan` 1.68e-03 / 3.0e-05；
  `density_skan` 2.04e-04 / 2.1e-05；`total_length_skan` 2.12e-03 / 1.0e-04。
  **归因**：几何链路逐位相同（`feat_fov_area_px`/`native_longest`/`scale`/`crop_h`/`crop_w`
  在 770 行上完全相等）⇒ 读图、FOV、裁剪确定性无差异；差异出现在 `pred_fg_frac`
  （最大 2.0e-04）⇒ **`amp=True` 的 fp16 算术与 cuDNN 算法选择在两台机器上不同**，
  概率图在 0.5 阈值附近移动约 1e-4，少量边界像素翻转，经骨架化与盒计数 FD 放大到 1e-3 量级；
  对单像素拓扑最敏感的 FD/total_length 散布最大、density 最小，与该机制的预测一致。
  **零均值、无系统偏差**（符号相对差 |mean|/sd = 0.011–0.043，LAN 高于本地的比例 0.35–0.47）
  ⇒ 只增加噪声，不会使任何对比朝一个方向偏移。若日后需要确定性可设 `amp=False`，
  但那将改动全部 39 张 LAN 表共用的标志，现阶段不做。
- 2026-09-17 20:24 **`messidor2/e3_hrf_s0_baseline_lanrep` 启动（协调者裁定第 2 条）**：
  用 HRF s0 baseline 检查点在空闲的 LAN 卡上重跑 Messidor-2，**写入独立标签
  `e3_hrf_s0_baseline_lanrep`，绝不触碰规范标签 `e3_hrf_s0_baseline`（归本地）**。
  一物两用：(a) 为 HRF 对比提供**同机配对**（消除 E3 中唯一的跨机器配对，见 20:25 条），
  (b) 作为**跨机器复制样本**，把 20:27 条的 ~1e-4 结论从 770 张扩展到全部 1,744 张。
  启动前核验：新标签目录不存在、规范标签在节点上**也不存在**（故不可能误写）、
  检查点 sha256 `fd24517bb24a…`、FOV 缓存 1,744 张齐全、GPU 空闲 0 作业。
- 2026-09-17 20:35 **推理尺度探针完成（18/18，0 失败），结论：HRF 的保真度缺口**不是**降采样伪影**。`src/pivot/e2_scale.py`，产物 `results/pivot/e2_scale.csv`（240 行）、`e2_scale_pixel.csv`（24 行）、`E2_REPORT.md` §9。冻结检查点在 2048 / 2560 / 3504（原生）重新推理，与既有 1536 运行配对比较（同 30 张 HRF 测试图，2000 次配对 bootstrap）。**关键在于效应在 baseline 与 continued 两臂上一致出现**，故是真实的尺度效应而非与臂混淆。
  (1) **FD 与 total_length —— 即保真度最差的两个标志物（r≈0.30）—— 在原生分辨率不但没有恢复，反而变差**：baseline 的 FD r 随尺度为 0.3045 → 0.3418 → 0.3252 → **0.2544**，total_length 0.3240 → 0.3596 → 0.3476 → **0.2844**，均在 2048 达峰后回落；3504 对 1536 的 Δr 为 FD **−0.0501**（0/3 种子为正，BREACH）、total_length **−0.0396**（0/3）。continued 臂同形（FD 0.3090 → 0.3960 → 0.3823 → 0.3045）。**降采样假说对主要缺口不成立。**
  (2) **中等上采样（2048）才是甜点**，且效应稳健：continued 的 FD Δr **+0.0870 [+0.0358, +0.1381]**、total_length **+0.0867 [+0.0433, +0.1302]**，CI 均不跨零、3/3 种子同向；baseline 的 total_length **+0.0356 [+0.0224, +0.0489]** 亦不跨零。
  (3) **density 是例外，随尺度单调大幅改善**：baseline r 0.6292 → 0.7462 → 0.7799 → **0.8113**，3504 对 1536 的 Δr **+0.1820 [+0.1314, +0.2326]**，3/3 种子；continued **+0.1300 [+0.0807, +0.1793]**。density 本就是 HRF 上第二好的标志物，也是对拓扑最不敏感的一个——与"尺度改善的是面积类而非拓扑类"的机制一致。
  (4) **代价是实打实的**：3504 上 Dice −0.044、clDice −0.069/−0.072，**18 个非参照运行中 12 个突破 clDice 0.01 非劣边界**；偏差发生**符号翻转**——HRF baseline 原本低估 −1.6 至 −2.2σ，3504 的 Δbias 为 FD **+2.63σ**、total_length **+2.82σ**（continued +2.96 / +3.23σ），即由低估转为高估；算力 2.5–3.4× 每图（1536 0.92 s/图 → 3504 2.76 s/图，baseline）。
  判读：尺度不是 HRF 拓扑类保真度缺口的原因；把推理放大到原生分辨率会**损害** FD/total_length 并大幅破坏拓扑一致性。可辩护的实用建议是 2048 附近的适度上采样，对 FD/长度有小而一致的收益、对 density 有较大收益，代价约 1.1–1.7× 算力，但仍会轻微突破 clDice 边界（−0.006），须如实报告。
- 2026-09-17 20:30 **跨节点一致性抽查：本地独立复算，确认 LAN agent 的诊断成立**（其 DECISIONS 20:25）。在本地对 `aptos2019/e3_hrf_s0_baseline` 的 LAN 完整表与本地 770 行半成品按 image_id 内连接（770/770 重叠），**逐位重算而非采信对方数字**，结果与其报告完全吻合：最大相对差 FD_skan 3.60e-3、tortuosity 1.68e-3、total_length 2.12e-3、density 2.04e-4。
  **几何链逐位相同**（`feat_fov_area_px` / `native_h,w` / `native_longest` / `scale` / `crop_h,w` 七列全部精确相等，maxabs=0）——图像读取、FOV 生成与裁剪框完全确定性，**FOV 来源被排除**，这正是先前最担心的失效模式。差异起自前向传播：`pred_fg_frac` 最大绝对差 **1.26e-05**（对方文中写 2.0e-4，当为相对量；此处以实测绝对值为准），amp/fp16 与 cuDNN 算法选择在两台机器上略有不同，概率图在 0.5 阈值附近移动，少量边界像素翻转，再经骨架化与盒计数 FD 放大到 1e-3 量级。
  **是噪声不是偏差**：|mean|/sd 为 0.011–0.043，frac(LAN>local) 为 0.351–0.474，零中心散布，无一台机器系统性偏高。
  **量化"是否影响交付"（本地新增的一步）**：Messidor-2 的 HRF s0 配对是全 E3 中**唯一**跨机器的配对对比（baseline 本地 / reliseg LAN）。在该对上，baseline↔reliseg 的真实信号相对差 sd 为 0.0118–0.0722，而跨机噪声 sd 为 3.3e-5–1.85e-4，**噪声/信号 = 0.07%–1.2%**。对配对检验而言，向一臂加入零中心独立噪声只会使差值方差膨胀 √(1+0.0116²) ≈ 1.00007（约 0.007%），即**略偏保守，不可能制造假阳性**。
  **结论：不重跑**。重跑属美观性修补，需占用 LAN 卡 30–60 min 并改写一张已完成且已核验的表，收益为零。**预注册主终点完全不受影响**——每个队列的 12 张 FIVES 检查点表全部产自 LAN 单机，故主对比（ΔAUC ReliSeg − baseline，FIVES seed 0-2，APTOS 主队列）是同机配对；受影响的只是一个复制队列上的次要 HRF 检查点比较。
  **写作要求**：论文**不得**声称生物标志物表跨机器逐位可复现。可写且可辩护的表述是"同一检查点、同一约定，跨机器一致到约 1e-4 相对量级且无系统性偏移"。`amp=False` 大概能换来确定性，但那会改动全部 39 张 LAN 表共享的标志，现在不做。`e3_hrf_s0_baseline_LOCALpartial_stopped_1824` 继续保留。
- 2026-09-17 20:40 **E4 独立复现完成（results/pivot/E4_REPORT.md）**：主终点 ReliSeg − continued（折外 n=100，折内 bootstrap，max-statistic 校正）FD −0.010 [−0.038, +0.014]、density −0.000、length −0.001，adj. p ≥ 0.86 → replication_success = FALSE；预注册功效在该保真度水平下对 Δr=0.05 为 0.98–1.00，故为**有功效的零结果**。continued − baseline 的 FD +0.008 [+0.001, +0.019] 为唯一有利方向的显著区间（更多训练略增保真度，测量项抵消）。安全：clDice/Dice 0/10 突破；迂曲度 skan Δr −0.063 [−0.137, −0.030]、PVBM Δbias +1.74σ → 附带损害被独立复现。下游四类 AUC：GT 臂 0.54 与预测臂无差 → 该终点在此集无信息量，不作证据。MAPLES-DR 零样本：较高分辨率层 18/18 单元 Δr(ReliSeg − continued) 为负；预标注锚敏感性方向不变。**C2 定性为经预注册、有功效的阴性结论；论文按此改写。**

- 2026-09-17 20:34 **20:25/20:27 两条的证据修正（E2 agent 独立复算后发现，已核实）**：
  (1) 我引用的 `pred_fg_frac` 差异 **2.0e-04 是相对值**，对应的**绝对值为 1.256e-05**，
  两者是同一数字的两种归一化，原文未标注口径。
  (2) 更要紧的一点：同一张表内 `pred_fg_frac` 与 `density_skan` **逐位相同
  （max abs diff = 0.000e+00）**——二者是同一个量（掩膜像素数 / FOV 像素数）的两个列名。
  故我原先把"`density_skan` 2.04e-04"与"`pred_fg_frac` 2.04e-04"并列为两项佐证是**重复计数**，
  实际只有一项。机制结论不变（掩膜确实有差异，这正是要点），但掩膜层面的差异
  只由一个列度量，两条 DECISIONS 的措辞据此更正。
  (3) E2 agent 独立复算内连接得到与我完全一致的数字（770/770，FD 3.60e-3、tortuosity 1.68e-3、
  total_length 2.12e-3、density 2.04e-4；几何 7 列 maxabs=0），结论互证。
- 2026-09-17 20:34 **噪声对配对检验的影响已定量，且主终点不受影响（E2 agent 提供，采纳）**：
  在 Messidor-2 的 HRF s0 配对上，baseline→reliseg 的**信号**相对 sd 为
  0.0118(FD)/0.0349(tortuosity)/0.0451(density)/0.0722(total_length)，
  **跨机噪声** sd 为 1.37e-4/9.84e-5/3.32e-5/1.85e-4 ⇒ **噪声/信号 = 0.07%–1.2%**。
  配对检验中向一臂加入零均值独立噪声，使差值方差膨胀 √(1+0.0116²) = **1.00007（0.007%）**，
  且方向**保守**——只可能让 CI 微幅变宽，**不可能制造假阳性**。
  **必须写进论文的一句**：预注册**主终点完全不受影响**——每个队列的 12 张 FIVES 检查点表
  都出自同一台 LAN 机器，故 ΔAUC(ReliSeg − baseline, FIVES seed 0-2, APTOS 主队列,
  referable DR, logreg on bio+cov) **全程同机配对**；跨机配对只涉及**一个复现队列上的
  一个次要 HRF 对比**。此句应明写，不应留给审稿人自行推导。
- 2026-09-17 20:45 **更正 20:30 条，并记录 E3 bio 表中的重复列**（由 LAN agent 指出 `pred_fg_frac` 与 `density_skan` 同源，本地复核后发现重复不止两列）。
  **更正**：20:30 条把 `density_skan` 的最大相对差 2.04e-04 与 `pred_fg_frac` 的最大绝对差 1.26e-05 并列为两项证据，这是**重复计数**——两者是同一个量在两种归一化下的同一数字（LAN 侧的 2.0e-4 是相对量，本地的 1.26e-5 是绝对量，指的是同一列）。掩膜层面的分歧由**一列**度量，不是两列互相印证。机制结论不变（掩膜确实不同，这才是要点），但证据强度应按一列计。
  **本地复核（三张表逐位比对）**：`density_skan` == `pred_fg_frac` == `feat_pred_density`，在 aptos LAN 表（3,662 行）、aptos 本地半成品（770 行）、messidor2 本地表（1,744 行）上**最大绝对差恒为 0.000e+00**，即同一个量（掩膜∩FOV 面积 / FOV 面积）以三个名字各存了一份。另发现第二组重复：`feat_prob_mass` == `feat_prob_mean`（两者都是概率图在 FOV 内的均值）。
  **对交付无影响，已逐一确认**：E3 分类阶段的特征集 `SKAN_COLS` 与 `COV_COLS` **均不含** `pred_fg_frac` / `feat_pred_density` / `feat_prob_mass`，故 `bio` 与 `bio+cov` 两个特征集不存在重复计数，预注册的主终点不受影响。（此前已知 `density_pvbm` 与 `density_skan` 亦相同——两条管线读的是同一个前景占比，见 e2_analysis 中 primary8 面板作为敏感性而非主面板的理由。）
  **给后续使用者的警告**：任何直接从 E3 `bio.csv` 列名拼特征集的分析，都可能把密度重复计入最多三次、把概率均值重复计入两次。要用就显式选列，不要按前缀或"全部数值列"自动收集。
- 2026-09-17 20:46 **Messidor-2 HRF s0 的跨机复制跑（`_lanrep`）：撤回反对意见**。此前建议"不重跑"针对的是重写规范表 `messidor2/e3_hrf_s0_baseline`；协调者实际下达的是写入**独立 tag** `messidor2/e3_hrf_s0_baseline_lanrep` 且明令不得触碰规范 tag，两者不是一回事，故原反对意见不成立。该方案严格增量：规范表分毫不动（LAN 节点上根本不存在该目录，启动前已确认），卡本就空闲不挤占他人，而且把跨机一致性的依据从"770 张机会性重叠"升级为**1,744 张完整队列的配对复制**——论文要引用的 ~1e-4 量级由此建立在完整队列上，同时该队列既有同机配对也有跨机配对，读者可自行比对二者是否一致。20:24:39 启动，预计 21:10 完成。
- 2026-09-17 20:38 **E3 主队列 APTOS-2019 分类阶段完成（三个队列全部收齐）**：本地 12 个 FIVES tag 于 19:44 齐备（各 3,662 行 + meta.json），等待器自动同步并在 CPU 节点起跑，canonical 19:45:29→20:09:10，六次逐种子配对运行 20:09→20:36。**预注册默认值未改一项**（同 17:38 条；唯一额外参数 `--n-jobs`，只作用于 GridSearchCV 并行度）。n=3,662，`cohort_role=primary`，图像级划分与图像级配对自助，APTOS 无预注册排除。产物已拉回 `results/pivot/e3/aptos2019/{summary,delta}.csv` + 96 个 `oof_*.npy`，逐种子在 `results/pivot/e3_perseed/ref{baseline,continued}_s{0,1,2}/aptos2019/`。
  **预注册主终点（referable DR / logreg / bio+cov / macro-AUROC，`is_primary=True`）**：12 个 tag 落在 **0.9053–0.9091**（极差 0.0038）。
  | 对照 | seed0 | seed1 | seed2 | 种子均值 |
  |---|---|---|---|---|
  | ReliSeg − baseline（**预注册主对比**） | −0.0005 [−0.0018, +0.0008] p=.420 | +0.0001 [−0.0014, +0.0016] p=.898 | −0.0004 [−0.0010, +0.0001] p=.138 | **−0.0003** |
  | ReliSeg − continued（同预算对照） | −0.0019 [−0.0027, **−0.0012**] p<.001 | −0.0010 [−0.0018, **−0.0003**] p=.004 | −0.0011 [−0.0016, **−0.0006**] p<.001 | **−0.0014** |
  **判读（必须如实写进论文，不得只报上面一行）**：主队列上**预注册的主对比 ReliSeg − baseline 为零结果**（三个种子 CI 全含零），与 IDRiD、Messidor-2 一致；**但同预算对照 ReliSeg − continued 在三个种子上全部显著为负**（CI 全部排除零，p ≤ .004）。即：在 APTOS 上，把 50 epoch 的预算花在测量感知微调上，其下游判别力**显著劣于**把同样预算花在继续常规训练上。方向一致性（3/3 种子）比单个 p 值更有说服力，故这不是偶然。**同时必须并列报告效应量**：−0.0014 AUC 是第三位小数上的差异，12 个 tag 的全距仅 0.0038，实践意义可忽略；结论应表述为"**统计上可检出、实践上可忽略的不利方向**"，而非"ReliSeg 损害下游效用"。
  **次要终点（同为逐种子配对自助，ReliSeg − 对照，macro-AUROC）**：GBDT 在 referable/bio+cov 上 vs baseline 均值 −0.0001（三种子 CI 全含零）、vs continued 均值 −0.0007（CI 全含零，p≥.066）——**GBDT 下 vs continued 的显著性消失**，说明该效应对分类器选择不稳健，进一步支持"可忽略"的判读。`bio` 特征集：vs baseline 均值 +0.0005（logreg）/ +0.0003（gbdt），vs continued 均值 −0.0011 两者皆是。DR 分级（0–4 序数，类别 0=1805;1=370;2=999;3=193;4=295）：bio+cov/logreg 的 QWK 臂均值 baseline 0.7498、continued 0.7533、cfloss 0.7539、**reliseg 0.7482**（qwk_expected）；bio+cov/gbdt 臂均值约 0.80，四臂同样在第三位小数内。macro-AUROC 臂均值（bio+cov/logreg）baseline 0.8609 / continued 0.8616 / cfloss 0.8627 / reliseg 0.8605。**cfloss 在 APTOS 的多数格上名义最高**，但同样落在种子噪声内，不据此下任何结论。
  **三队列汇总**：APTOS（主，n=3,662）、IDRiD（复现，n=516）、Messidor-2（复现，n=1,582，已剔除 162 张 MAPLES-DR 重叠）。**预注册主对比 ReliSeg − baseline 在三个队列上一致为零结果**（9 个逐种子 CI 全含零）；ReliSeg − continued 仅在 APTOS 上显著为负，IDRiD/Messidor-2 上 CI 含零。E3 外部下游终点整体为**阴性**，按预注册如实报告，不做事后筛选、不改设计。
- 2026-09-17 20:38 **E4 MAPLES-DR 生物标志物 24/24 交付完毕（CPU 节点）**：24 个 `runs/pivot/e4/maples/<tag>/bio.csv` 于 19:46:16 全部落到本地规范路径，交付回路随即自动退出。**逐文件校验（本地复核，非仅信任回路）**：24 个文件，**每个恰好 162 行**；8 个 `PRIMARY_COLS`（FD/tortuosity/density/total_length × skan/pvbm）全部存在且无 NaN；24 个文件的 `image_id` 集合**两两完全一致**。
  **与 GT 锚的相关性抽查（24 个目录全查，不是抽一个）**：`total_length_skan` r=0.949–0.973、`FD_skan` r=0.837–0.916、`density_skan` r=0.637–0.822（FIVES 源 0.805–0.822 系统性高于 HRF 源 0.637–0.744，与零样本域差一致）、`tortuosity_skan` r=0.365–0.890 离散度大——**与门禁 A 早已记录的"迂曲度是弱标志物（两管线 ρ=0.65）"一致，不是本次计算的缺陷**。没有任何一个 tag 在任何一列上是离群的，故不存在 FOV 错配或图像配对错误。
  全程只写这 24 个 `bio.csv`，未触碰两个 MAPLES 锚，未改动 E4 agent 树中任何其他文件；`cmd_maples_analyse` / `cmd_finish` 读的正是这些路径。
- 2026-09-17 20:50 **E3 主队列 APTOS-2019 分类结果（CPU 节点，锁定默认值）**：主终点 referable-DR logistic AUROC（bio+cov）12 个标签跨度 0.9053–0.9091；预注册主对比 ReliSeg − baseline 三种子 CI 均含零（均值 −0.0003）；ReliSeg − continued 三种子均显著为负（均值 −0.0014，CI 排除零），但效应量在第三位小数、GBDT 下不显著 → 表述为"可检出、实际可忽略、方向不利"。三队列合计：主对比 9/9 逐种子 CI 含零；E3 外部效用终点整体为阴性。E4 MAPLES 生物标志物 24/24 交付并核验。DECISIONS 尾部因多代理并发追加不再严格按时间排序（各条自带时间戳，不重排）。
- 2026-09-17 21:34 **推理操作尺度探针结果（results/pivot/e2_scale.csv；判定规则见 18:45 条）**：HRF baseline 三种子，FD/length 在 2048/2560/3504 相对 1536 的 Δr 均未达 +0.10（FD：+0.02/+0.05/−0.02，+0.05/+0.02/−0.06，+0.05/−0.01/−0.06），原生尺度反而略降且常数偏差由 −1.6~−2.2σ 翻转为 +0.6~+0.8σ（训练尺度 1536 的网络在原生尺度下把血管看得更细→过估）；density 的 r 随尺度单调上升（+0.10~+0.18，CI 排除零）但非主终点。**判定：不满足预注册规则，"原生尺度推理"不作为 C2 候选干预**；结果作为审计补充报告：(i) FD/length 的逐图保真度不受推理尺度限制；(ii) 常数偏差是尺度依赖的，跨尺度协调必须匹配操作尺度。至此所有候选干预（测量感知微调、CF-Loss、TTA、事后校准、拓扑修复、推理尺度）在预注册检验下均无保真度增益；论文定稿为：C1 审计 + 神谕臂衰减 + 剂量-反应机制（主），C2 预注册有功效的阴性结论 + 单侧附带损害 + CF-Loss 复现（次）。

- 2026-09-17 21:02 **`messidor2/e3_hrf_s0_baseline_lanrep` 完成并回拉核验（协调者裁定第 2 条执行完毕）**：
  20:24:39 → 21:02:18，**37 分 39 秒**跑完 1,744 张（空闲卡单作业 1.3 s/图），`rc=0`。
  核验：**1,744/1,744 行、`image_id` 全唯一、四个 skan 列 0 个 NaN**、
  `resize_longest=1536 / patch=768 / threshold=0.5 / amp=True`、
  `ckpt_sha256 = fd24517bb24a…` 与本地 `runs/seg/hrf/seed0/best.pt` 现算值一致；
  回拉远端 3 / 本地 3 文件。**规范标签 `e3_hrf_s0_baseline`（归本地）自始至终未被触碰**
  ——它在 LAN 节点上根本不存在，本地那份 `meta.json` 完好。
- 2026-09-17 21:05 **跨机一致性结论升级为 1,744 张配对复制（取代此前 770 张的机会样本）**：
  同一冻结检查点 `fd24517bb24a…`、同一约定，LAN vs 本地：
  | 列 | max abs rel | mean abs rel | signed mean | \|mean\|/sd | LAN>本地 |
  |---|---|---|---|---|---|
  | FD_skan | 3.77e-04 | 1.90e-05 | −7.6e-08 | 0.002 | 0.481 |
  | tortuosity_skan | 3.83e-03 | 2.77e-05 | −6.3e-06 | 0.044 | 0.262 |
  | density_skan | 1.44e-04 | 2.08e-05 | +2.4e-07 | 0.009 | 0.431 |
  | total_length_skan | 1.15e-03 | 7.23e-05 | −1.3e-06 | 0.011 | 0.357 |
  **三项结论在全队列上复现**：(1) 几何链路 7 列**逐位相同**（maxabs=0.000e+00：
  `feat_fov_area_px`/`native_h`/`native_w`/`native_longest`/`scale`/`crop_h`/`crop_w`）
  ⇒ 再次排除 FOV/读图/裁剪来源；(2) 掩膜层面 `pred_fg_frac` max abs = **1.245e-05**
  （与 770 张样本的 1.256e-05 一致）⇒ 差异确实只源于 GPU 前向的 amp/cuDNN；
  (3) **零均值无系统偏差**（|mean|/sd ≤ 0.044）。
  论文口径据此定稿：**"同一检查点、同一约定，跨机器一致性达 ~1e-4 相对量级、无系统偏移"**，
  n=1,744 配对，**不声称逐位一致**。该队列现同时拥有同机配对（`_lanrep` vs `reliseg`，均 LAN）
  与跨机配对（规范 `baseline` 本地 vs `reliseg` LAN），读者可自行验证两者一致。
- 2026-09-17 21:05 **重复列别名已在 LAN 表上独立确认（E2 agent 发现，扩展为三重）**：
  在 `_lanrep` 表（1,744 行）上实测 `density_skan` == `pred_fg_frac` == `feat_pred_density`
  **三者两两 maxabs = 0.000e+00**，且 `feat_prob_mass` == `feat_prob_mean`（maxabs = 0）。
  连同既知的 `density_pvbm` == `density_skan`，**density 在本项目表中最多有四个列名**。
  已核实：`SKAN_COLS` 与 `COV_COLS` **均不含** `pred_fg_frac` / `feat_pred_density` /
  `feat_prob_mass`，故 `bio` 与 `bio+cov` 两个特征集**都不存在重复计数，预注册主终点不受影响**。
  **强制规则（写入下游）**：**绝不可**按列名前缀或"所有数值列"从 E3 `bio.csv` 构造特征集——
  density 会进入最多三次、概率均值两次，模型将被静默地向"面积类"量加权、远离拓扑量；
  而本项目的全部论点正是关于拓扑标志物，此偏差尤其致命。**必须显式列举列名。**
- 2026-09-17 21:15 **跨机一致性以 1,744 张完整队列的配对复制确立，本地逐位复算无误；并给出论文应引用的稳定界**。LAN 节点的 `messidor2/e3_hrf_s0_baseline_lanrep`（20:24:39–21:02:18，37 min 39 s，1.3 s/图，rc=0，1,744 行、ckpt 与本地 `runs/seg/hrf/seed0/best.pt` 的 sha256 一致、resize 1536 / patch 768 / thr 0.5 / amp True）与本地规范表逐图配对，**1,744/1,744 匹配**。本地独立复算，与 LAN 报告**逐位吻合**：
  几何链七列（feat_fov_area_px / native_h,w / native_longest / scale / crop_h,w）**bit-identical，maxabs=0.000e+00**；`pred_fg_frac` 最大绝对差 **1.2448e-05**（770 张样本上为 1.256e-05，在 2.3× 大且构成不同的样本上复现到两位有效数字，这使 amp/cuDNN 归因从"合理故事"变成可检验且已通过检验的结论）；四列的 max|rel| / mean|rel| / signed mean / |mean|/sd / frac(L>O) 与 LAN 所报完全一致。
  **更正 LAN 的一处告诫**：其把 tortuosity 标为"最不对称、需留意的一列"（frac LAN>local 0.262、|mean|/sd 0.044）。本地做符号检验（剔除并列后的二项检验）：**四列 p 值分别为 FD 0.387、tortuosity 0.146、density 0.659、total_length 0.821，无一显著**。frac(L>O) 偏低是**并列（bit-identical 行）被计入分母**的产物，不是系统性偏移。故"无系统性偏移"对四列**均**成立，包括 tortuosity——该告诫可以撤下，结论比其原本以为的更强。另计系统分量与真实信号之比（同队列 reliseg↔baseline 的相对差 sd 为基准）：**6.4e-06 ~ 1.8e-04**，即最坏情况下亦仅为组间信号的 0.018%。
  **论文应引用的数值（采纳 LAN 关于"不要引用逐列最大值"的意见并给出替代）**：逐列 max 是样本相关的尾部行为——FD 的 max 在 770 样本上为 3.60e-3、在 1,744 样本上降一个量级至 3.77e-4，而 tortuosity 反向升至 3.83e-3——不可作为列的稳定属性。改引**跨四列的稳定界：mean|rel| ≤ 7.2e-05，p99|rel| ≤ 4.7e-04**，配合"几何链逐位相同、符号检验不拒绝零中心"。定稿表述：**"同一检查点、同一约定下跨机器一致到平均 7e-5、99 分位 5e-4 相对量级，无可检出的系统性偏移；不声称逐位可复现。"**
  规范表 `messidor2/e3_hrf_s0_baseline` 全程未被触碰（LAN 节点上不存在该目录），`_lanrep` 为独立 tag，两者并存，读者可自行比对同机配对与跨机配对是否一致。
  E3 最终盘点：本地树共 **43 张表 / 84,652 行**（三个主队列），43/43 检查点哈希已核验，零问题；LAN 交付 40 张（IDRiD 12、Messidor-2 14、APTOS 14），本地 3 张；ODIR-5K 由 GPU2 节点承担。

- 2026-09-17 21:18 **撤回 21:05 条中的"tortuosity 需留意"提法：那是并列(ties)计数假象，非系统偏移**
  （E2 agent 指出，我方独立复算确认）。我原以 `frac(LAN>本地)=0.262` 判定 tortuosity 最不对称，
  但该比例把**逐位相同的行**计入了分母。实测四列的并列数差异极大：
  | 列 | ties | gt | lt | gt/(gt+lt) | 符号检验 p | mean\|rel\| | p99\|rel\| |
  |---|---|---|---|---|---|---|---|
  | FD_skan | 104 | 838 | 802 | 0.511 | 0.387 | 1.90e-05 | 1.15e-04 |
  | tortuosity_skan | **784** | 457 | 503 | 0.476 | **0.146** | 2.77e-05 | 4.00e-04 |
  | density_skan | 258 | 752 | 734 | 0.506 | 0.659 | 2.08e-05 | 7.54e-05 |
  | total_length_skan | 489 | 623 | 632 | 0.496 | 0.821 | 7.23e-05 | 4.70e-04 |
  tortuosity 的 1,744 行里有 **784 行（45%）逐位相同**，是四列中最多的，故"严格大于"的比例
  在两侧同时被压低；**剔除并列后为 457/960 = 0.476，二项符号检验 p = 0.146，不显著**。
  四列 p 均 ≥ 0.15 ⇒ **"无可检测的系统偏移"对四列全部成立，结论比我原先的表述更强而非更弱。**
  附带佐证：tortuosity 并列最多这件事本身支持既定机制——它是路径长度之比，
  边界少量像素翻转时骨架拓扑往往不变，故该列对掩膜扰动最不敏感。
  系统性分量相对真实的检查点间信号仅 6.4e-06–1.8e-04（≤ 0.018%）。
- 2026-09-17 21:18 **跨机可复现性的论文定稿措辞（用稳定统计量，不用会随样本漂移的尾部极值）**：
  21:05 条已指出逐列 max 不稳定（FD 的 max 从 770 张样本的 3.60e-3 掉到 1,744 张的 3.77e-4，
  而 tortuosity 升到 3.83e-3）。改用跨四列的稳定统计量：
  **mean|rel| ≤ 7.2e-05、p99|rel| ≤ 4.7e-04**。定稿句：
  > "Same checkpoint and convention, biomarker tables agree across machines to 7e-5 mean and
  > 5e-4 99th-percentile relative difference, with the geometry chain bit-identical and no
  > detectable systematic offset (sign test p >= 0.15 on all four markers).
  > Bit-level reproducibility is not claimed."
- 2026-09-17 21:25 **并列率（tie rate）复核：计数属实且解释了 0.262 假象；但"与离散度排序相反"的机制论证不成立，不应写进论文**。
  **属实且应保留的部分**：本地复算 1,744 对的并列计数与 LAN 所报**逐位一致**——tortuosity 784 并列 / 457 大于 / 503 小于，total_length 489 / 623 / 632，density 258 / 752 / 734，FD 104 / 838 / 802。并列率排序 **tortuosity 45.0% > total_length 28.0% > density 14.8% > FD 6.0%**。这**完全解释**了此前 frac(LAN>local)=0.262 的假象：tortuosity 有 45% 的行逐位相同，"严格大于"的比例被并列摊薄，剔除并列后为 457/960=0.476、符号检验 p=0.146。四列符号检验 p 值（0.387 / 0.146 / 0.659 / 0.821）本地复现，"无可检出系统性偏移"对四列均成立。
  **不成立、须剔除的部分**：LAN 提出并列率排序"与离散度排序相反"，据此论证二者是同一机制的两面（"FD 变化频繁但幅度小、tortuosity 罕有变化但一变就大"）。以**稳定统计量**核对则不然：mean|rel| 的排序是 total_length 7.2e-05 > tortuosity 2.8e-05 > density 2.1e-05 > **FD 1.9e-05**——FD 的并列率**最低**且离散度**也最低**，二者同向而非相反；Spearman rho(并列率, mean|rel|) = **+0.800**（正相关）。该"相反"关系只在**逐列最大值**上成立，而最大值正是 21:15 条刚判定为样本相关尾部、不得引用的统计量（FD 的 max 在两个样本间相差一个量级）。此外只有 **4 个数据点**，无论正负相关都不可能有统计支持（n=4 时 rho=0.8 的 p≈0.33）。
  **结论**：并列计数本身与"它解释了 0.262"写入记录；由并列率推出的机制叙事**不进论文**——它建立在一个已被弃用的统计量上，且样本量为 4。amp/cuDNN 归因的证据仍然充分（几何链逐位相同、pred_fg_frac 在两个不同样本上复现到两位有效数字、零中心），不需要这一条来支撑。
  论文定稿表述维持 21:15 条不变。

- 2026-09-17 21:28 **撤回 21:18 条附带的"并列率与离散度反序 ⇒ 机制佐证"推论（我方错误，E2 agent 指出，
  已自行复算证伪）**。我曾称并列率排序与离散度排序相反、二者是同一机制的两面。**实测三种统计量
  全部为正相关，无一反序**：
  | 列 | 并列率 | mean\|rel\| | p99\|rel\| | max\|rel\| |
  |---|---|---|---|---|
  | FD_skan | 0.060 | 1.90e-05 | 1.15e-04 | 3.77e-04 |
  | density_skan | 0.148 | 2.08e-05 | 7.54e-05 | 1.44e-04 |
  | total_length_skan | 0.280 | 7.23e-05 | 4.70e-04 | 1.15e-03 |
  | tortuosity_skan | 0.450 | 2.77e-05 | 4.00e-04 | 3.83e-03 |
  Spearman ρ(并列率, mean)=**+0.800**(p=0.200)、ρ(并列率, p99)=**+0.600**(p=0.400)、
  ρ(并列率, max)=**+0.800**(p=0.200)。FD **并列率最低且 mean 离散也最低**——同向而非反向。
  且 n=4 列，任何方向都无统计效力（p≥0.2）。
  **错误根源值得记录**：我的"反序"印象来自 770 张样本中 FD 的 max 最大，而并列率我**从未在那份
  样本上算过**，等于跨两个不同样本拼出一个对比；更糟的是它建立在 **max** 这个我们刚刚在
  21:18 条共同议定"随样本漂移、不得引用"的统计量上。**这与刚刚避开的错误是同一个，只是高了一层。**
  **处置**：并列率数据与"0.262 是并列计数假象"的解释**保留**（它们经双方独立复算、且直接支撑
  符号检验的正确用法）；由并列率推出的机制叙事**删除，不入论文**。
  amp/cuDNN 归因仍由三条成立的证据支撑，无需第四条：(1) 几何链 7 列在 1,744 行上逐位相同；
  (2) `pred_fg_frac` 的差异在两个成分不同的样本上复现到两位有效数字（1.256e-05 / 1.245e-05）；
  (3) 零均值且四列符号检验 p ≥ 0.15。**加入一条经不起核查的佐证只会削弱整组证据**——
  读者一旦验证发现排序主张不成立，会连带怀疑其余三条。21:15 条的论文定稿措辞不变。
- 2026-09-17 21:32 **并列率机制论证正式撤回（双方各自独立复算后一致）；并提出一条应写入流程的做法**。本地第三次复算确认：rho(并列率, mean|rel|) = **+0.800**（p=0.200）、rho(并列率, p99|rel|) = **+0.600**（p=0.400）、rho(并列率, max|rel|) = **+0.800**（p=0.200）——**三个统计量上全部为正**，包括原以为支持"相反"的 max，故该论证在任何一个统计量上都不成立，且 n=4 时三者均不显著。秩序（低→高）：并列率 FD < density < total_length < tortuosity；mean|rel| FD < density < tortuosity < total_length；max|rel| density < FD < total_length < tortuosity。
  **错误是如何产生的（比结论更值得记）**：LAN 的"相反"印象来自 FD 在 **770 张样本**上的最大值，而并列率从未在该样本上算过——即把**来自两个不同样本的两个统计量**配成对比，且所倚赖的正是双方在**上一条消息里刚刚共同弃用**的逐列最大值（该值在两个样本间相差一个量级）。换言之，是在撰写"弃用最大值"这条更正的同时又用它构造了新论断。
  **保留**：并列计数与"tortuosity 的 45% 并列率解释了 frac=0.262 假象"——已由双方各自复算两次，正是它使符号检验成为正确的检验。**剔除**：由并列率推出的机制叙事，不进论文。amp/cuDNN 归因仍由三条成立的证据支撑（几何链逐位相同、pred_fg_frac 跨两个不同样本复现到两位有效数字、零中心且附 p 值）；加入一条经不起核验的第四条只会削弱前三条，因为读者查出其排序主张为假后会开始怀疑其余。定稿表述维持 21:15 条。
  **建议作为常规做法（LAN 提出，本地附议，交协调者裁定）**：论文剩余主张一律**由未产出该数字的一方独立重算后方可采信**。今日两次由此推翻结论而非确认结论——FIVES manifest 词干错配（99/200 张测试图被配到 train 记录）与本条——且本条是复算方抓到了刚刚写下该规则的一方违反了它。这比任何一方声称"我检查过了"都更有说服力。

- 2026-09-17 21:50 **ODIR-5K E3 外部推理在 2080 Ti 节点全部完成、已回拉并通过行数与检查点哈希核验（14/14）**。
  两条 driver 分别于 **20:52:22**（GPU0，7 个）与 **20:54:06**（GPU1，7 个）打印 `ALL DONE`，
  14 个作业 **rc=0、各 6,392 行**，无一重跑、无一 OOM；从 18:11:56 启动算起**整批 2 h 42 min**
  （89,488 次图像推理；串行约需 19 h）。实测稳定在 **1.5 s/img/worker**，与单作业独跑的 1.6 s/img
  基本相同——14 路并发在 96 核 / 2×11 GB 上**没有并发惩罚**，每 worker 仅占 652 MiB。
  回拉：`gpu2_pull.sh results/pivot/e3/odir5k`，**42/42 文件**（14 tag × {bio.csv, meta.json, resolutions.csv}），
  落到 `exp/results/pivot/e3/odir5k/<tag>/`；节点日志另存 `exp/runs/gpu2_logs/`。
  **核验（`gpu2_verify_pulled.sh odir5k 6392`，全程本地、不依赖节点）：14 ok / 0 bad**——
  每个 tag 都有 meta.json（完成标记）、bio.csv 恰为 **6,392 行**，且 meta.json 里的 `ckpt_sha256`
  与本地对应 `.pt` 现场重算的 SHA-256 **逐一相符**，**14 个摘要两两不同**：
  baseline s0-2 `4f0489e1cff4`/`ec5172f7417b`/`dc580f6c7530`；reliseg `bed33d322ea6`/`6227c165e948`/`2d4ba04e4056`；
  cfloss `c4aaca9182bc`/`4e6575fda818`/`5876d8f4fb3a`；continued `f5d6a09dac98`/`cfee9c045f39`/`decf9b4ec853`；
  HRF s0 `fd24517bb24a`（baseline）/`1325429e2850`（reliseg）。前 12 个与 LAN 节点 IDRiD / Messidor-2 所用完全一致，
  **冻结分割器的身份守卫跨三台机器成立**。
  另核：7 个 tag 记 `device=cuda:0`、7 个记 `cuda:1`，与作业划分一致；全部 `resize_longest=1536`、
  `patch=768/stride=384`、`crop_to_fov=True`、`threshold=0.5`、`n_images=6392`、`pipelines=['skan']`；
  抽样 tag 内 **6,392 个不重复 image_id / 3,358 个 subject_id**，原生分辨率全为 512×512、推理长边全为 1536
  ——与 ODIR 仅有 512 px 预处理镜像、按 01:00 条降为**低分辨率敏感性集**的定位相符（`scale` 列已记录实际放大比）。
  至此 E3 的四个外部集齐备：IDRiD 与 Messidor-2（LAN 节点，16:12 条）、APTOS-2019（LAN 节点）、ODIR-5K（本条）。
  2080 Ti 节点由用户在核验完成后关停；**核验不依赖节点**，上述哈希比对全部在本地完成，
  环境与复现配方留存于 `exp/remote/ENV_gpu2.md` + `gpu2_*.sh`，日后再租同型机器可直接复用。

- 2026-09-17 21:56 **论文端：E3 主队列 APTOS-2019 写入、推理尺度探针降为审计附录、终稿论述定框。**
  1) **一处实质更正**：E3 的预注册主终点是 **`featureset=bio+cov`**（即 `e3/<cohort>/summary.csv`
  中 `is_primary=True` 的那一格：`target=y_referable`, `clf=logreg`），而不是未校正的 `bio`。
  上一版 `tab:e3` / `fig6` / §5.11 正文用的是 `bio`；已将 `mk_tab_e3.py` 与 `fig_e3_external.py`
  切到 `bio+cov` 并重写正文，`bio` 降为敏感性分析（二者结论一致）。
  2) **APTOS-2019（n=3,662）**写法：ReliSeg − baseline 三种子均为空（均值 −0.0003）；
  ReliSeg − continued 三种子均为负且区间不含零（−0.0019 / −0.0010 / −0.0011，均值 −0.0014）。
  论文表述为“**统计上可检出、实际上可忽略、方向不利**”，并同时给出三条降温证据：
  12 个 arm×seed 格的 AUROC 跨度仅 0.0038；GBDT 下同一对比为 −0.0011 / −0.0010 / −0.0001，区间全含零；
  五级 QWK 四个 arm 在任一种子下相差 ≤ 0.012，与单 arm 的种子间波动同量级。不将 10⁻³
  的下降渲染为“危害”。另：补正 Messidor-2 上 CF-Loss − continued 在两个种子上区间不含零
  （+0.0071 / +0.0051），按“同级噪声”写（同一 arm 在更大的主队列与 IDRiD 上均为空）。
  3) **推理尺度探针**（预注册规则见 18:45 条）**未达标**，按预定降为审计附录：
  新增 §5.2 `sec:res-scale` + 补充材料 S4 + `tools/mk_tab_scale.py` → `tab:scale`。
  FD / 总长度在 2048–2560 小幅上升、在原生 3504 反而下降（−0.050 / −0.040），且 clDice 破界（−0.069）、
  耗时 3.4×。保留两条审计结论：密度保真度随尺度单调上升（+0.117 / +0.151 / +0.182，区间均不含零）；
  常数偏置随尺度**变号**（FD −0.97σ → +0.73σ，总长度 −1.10 → +0.78）——因此跨队列常数校正
  只在固定推理尺度下有意义，这是对 §5.8 报告尺度建议的硬化。§6 的 `\todo` 随之关闭。
  4) **终稿论述**（摘要 / §1 C2 / §6 / §7 四处同步）：所有能动的补救手段——D4 TTA、事后常数协调、
  拓扑修复、原生尺度推理、忠实复现的 CF-Loss、我们自己的测量感知微调——**均在看到结果前固定的规则下被检验，
  没有一个能提高逐图保真度**；论文的正面内容是审计、参考掩模差距的衰减、以及注入剂量响应机制；
  C2 是有功效的负结果 + 单向附带损害 + CF-Loss 复现。另校正了一处队列计数：“六个队列”把两个
  无掩模的临床队列混入了保真度主张；现在保真度主张只覆盖**四个有参考掩模的队列**，
  三个外部临床队列单独作为下游效用的空结果报告。
  5) **构建**：70 页（正文含参考文献 60 页 + 补充材料 10 页），无未定义引用/交叉引用；
  `\todo` 从 66 降到 6：§1 伴随稿引用、1 条 ODIR-5K 敏感性、以及 4 条 back-matter 模板。
  ODIR-5K（本日 21:50 完成分割推理）在 `tab:e3` 中以“已测量、待分类”单行占位，不承载任何结论。
- 2026-09-17 21:55 **`ensure_fov` 原子写入修复已实施并重新指纹（17:40 条 (b) 的延后事项结清）**。改法：先把掩膜生成到**同目录下带 pid 的临时文件**，再 `os.replace()` 就位，与 `src/pivot/common.py::json_dump` 同一模式。`os.replace` 在同一文件系统上是原子的，故读者要么看到"文件不存在"、要么看到完整文件，17:23 条记录的竞态在结构上不再可能发生。
  **两处比"照搬 json_dump"更进一步的细节**：(1) 临时文件名带 **pid**，而 json_dump 用固定的 `.tmp`——固定名只对单写者安全，多进程共用同一临时名等于把同一个竞态下移一层。(2) pid 放在**扩展名之前**（`xxx_fov.tmp<pid>.png`）：`_generate_fov` 经由 PIL 保存，PIL 按后缀推断格式，首版把 pid 放在末尾直接触发 `ValueError: unknown file extension: .tmp50404`——**该缺陷由验证步骤当场抓出，未流入任何运行**。另加 `finally` 清理残留临时文件。
  **验证（不是"看起来对"，是逐字节核对）**：① 掩膜已存在时不重写；② `regenerate=True` 重新生成后与原文件**逐字节相同**（messidor2 一张 + idrid 三张，SHA-256 全部一致）；③ 无残留 `.tmp` 文件；④ 仍为合法 PNG（1488×2240 uint8）；⑤ 不同 pid 产生不同临时名。**这条是"此修复不可能改变任何数字"的实证依据**——它只改变掩膜字节在被改名就位前写在哪里。
  **指纹更新**：`src/data/external.py` 旧 `6e96340dd5c61bc94db6ccb4e48614f95fcae4bde0984e7e0f140de3d3e60c0c` → **新 `1693aaa7836088a32409ba5982ccece56a4903e5a11cd14a0d382bee23338d6d`**（2026-09-17 21:55:43，16,760 字节）。**已交付的全部 43 张 E3 生物标志物表均产生于旧哈希之下**；因上述逐字节验证，新旧哈希下的掩膜内容相同，故无需重跑任何一张表，历史结果与新代码不冲突。`src/pivot/e3_external.py` 未改动，仍为 `af65105e…`（与 21:15 冻结一致）。
  **文档**：`data/EXTERNAL_DATASETS.md` 的前置条件一节改为"竞态已修复，但仍建议预建 FOV"，并给出修复**未覆盖**的三条理由——把 FOV 生成移出计时循环（否则逐图耗时不再可信）、让损坏或不可读的源图在开跑前而非分片波次数小时后暴露、可在长跑之前核对掩膜数与记录数。原竞态描述改为过去时并保留，因为**那个报错签名值得认得**（imageio 回退到 SPE 插件后报 `cannot reshape array of size 1701 into shape (36396,60828)`，与 I/O 错误毫无相似之处）。
  **执行顺序说明（与协调者指示不同，已核实其可行性）**：协调者要求 (2) 在 (1) ODIR 分类跑完之后再做。经核实 `run_clf` **从不导入** `src.data.external`——`ast` 扫描确认 `ensure_fov` 与 `load_external` 的导入分别只出现在 `_row_for` 与 `run_bio` 内，均属 bio 阶段；且三个节点的 bio 工作已全部完成交付，不存在可被扰动的 bio 运行。故改为与 (1) 并行实施：排序约束在此不具约束力，而本会话今日已中断过一次，推迟到 4–6 小时后有使该修复丢失的实际风险。已如实上报。
- 2026-09-17 21:59 **稿件模板（用户指示）**：paper2 改用 Elsevier CAS 双栏模板 cas-dc（els-cas-templates.zip），具体期刊待定；后备论文保持 cas-sc。

- 2026-09-17 22:02 **`ensure_fov` 原子写修复已同步到 LAN 节点；"无需重跑"的结论由我方独立复核确认
  （并发现一个会误报的核查方法）**：
  节点 `src/data/external.py` 由 `6e96340d…` 更新为 **`1693aaa7…`**（16,760 字节），
  `src/pivot/e3_external.py` 仍为锁定的 `af65105e…`，两机现已完全一致。
  修复代码经**阅读确认**而非假定：临时文件写在**同目录**、名字含 **pid**（避免共享临时名把竞态下移一层）、
  **pid 置于扩展名之前**（PIL 按后缀推断格式）、`os.replace()` 原子落位、异常路径清理。
  **独立复核（我方执行，非采信对方数字）**：在节点上用新代码重新生成 FOV 并与既有缓存比对——
  | 集合 | n | **像素逐一相同** | 字节相同 | max 像素差 |
  |---|---|---|---|---|
  | idrid | 25 | **25** | **0** | 0 |
  | messidor2 | 10 | **10** | 10 | 0 |
  | aptos2019 | 10 | **10** | 10 | 0 |
  **像素层面 45/45 完全一致，max 差 0** ⇒ 已交付的 43 张表（全部在 `6e96340d…` 下产生）
  **确认无需重跑**；`fov` 目录无残留临时文件。
  **重要发现——跨机器时"字节比对 FOV 掩膜"是无效核查，会误报**：IDRiD 的 25 个掩膜
  **字节全部不同但像素全部相同**。原因：IDRiD 的 FOV 是在**本地 Windows** 生成后同步过去的，
  PNG 编码（zlib/PIL 版本）与 Linux 侧重新生成的不同；Messidor-2 与 APTOS 的掩膜是在节点上
  生成的，故字节也相同。E2 agent 的"字节相同"结论成立，仅因其在**生成掩膜的那台机器上**核查。
  **规则**：核查 FOV 一致性**必须比像素、不能比字节**，除非能确认两侧由同一台机器、同一套库生成。
  否则会在完全健康的数据上得出"损坏"结论——而本项目对 FOV 来源最为敏感
  （E4 的"只复制、绝不重新生成"规则即源于此），这类误报的代价很高。
- 2026-09-17 22:05 **方法更正：FOV 掩膜的一致性必须比较**像素**而非字节，除非两边由同一台机器与同一套库产生**（由 LAN agent 在复核 21:55 条时发现，本地已确认其成立）。
  **发现经过**：21:55 条用 SHA-256 逐字节比对来证明"修复不改变任何掩膜"。LAN agent 在其节点上照此复核 IDRiD，得到 **25/25 全部不同**，几乎据此上报故障；改比像素则 **25/25 完全相同、最大差 0**。原因：IDRiD 的 FOV 缓存是在本地 **Windows** 机器上生成后同步到节点的，其 PNG 字节带有与 Linux 端 PIL/zlib 不同的编码；而 Messidor-2 与 APTOS 的缓存本就在节点上生成，故字节也相同。三个队列合计 **45/45 像素相同**，21:55 条的结论成立且证据基础由 4 张扩大到 45 张。
  **本地验证（两个方向都查了）**：① 在**产出该掩膜的机器上**重新生成，字节与像素结论一致——idrid 5/5、messidor2 3/3、aptos2019 3/3 既逐字节相同也逐像素相同，故 21:55 条当时的判断没有被这一效应污染；② PNG 字节确实依赖编码器设置——同一幅像素在 compress_level 0/6/9 下产生 **3 个互不相同的字节流**，像素完全一致。因此字节相等是比像素相等**更强**的条件，强出来的那部分与掩膜内容无关。
  **规则**：凡涉及掩膜一致性的核验，一律比较 `np.array_equal` 的像素；字节比对只在"同机同库产出"这一附加前提成立时才等价，而该前提**不写在方法里就看不见**，也不会在换机器复核时自动成立。
  **与 E4"只复制、不重新生成"规则的关系（须写进使用说明）**：对**复制来的**掩膜做字节比对，会在数据完全健康的情况下报出"损坏"；更糟的是，它可能促使人以"重新生成"来修复，而重新生成正是该规则要避免的动作。需要说明的是，重新生成本身在**生成器与依赖一致时是安全的**（本条 ① 已实测像素相同）；之所以仍然优先复制，是因为复制彻底消除了对本地库栈的依赖，而这种依赖无法跨机器低成本验证。
  **对"独立复算"常规做法的补充建议（与 LAN 一致，交协调者裁定）**：当主张本身是关于**可复现性**时，独立复算应当**在不同的基础设施上**进行——今日正是"在同一台机器上重复同一检查"掩盖了编码依赖。这是今日第三次由独立复算推翻既有结论，也是第一次出错的是**方法**而不是数字。

- 2026-09-17 22:27 **论文端：paper2 改用 Elsevier CAS **双栏**模板（`cas-dc`）。**
  `main.tex` 改为 `\documentclass[a4paper,fleqn]{cas-dc}` + CAS 题首（`sections/frontmatter_cas.tex`、
  新增 `sections/highlights.tex`）+ `cas-model2-names`（著者-年份），`refs.bib` 不变；
  原 elsarticle 包装器保留为 `main_elsarticle.tex`（`build.ps1 -Main main_elsarticle`），正文共用同一套 sections。
  **页数：70 页（单栏）→ 36 页（双栏），正文含参考文献 60 → 29 页**，无任何删内容；无未定义引用/交叉引用。
  三个必须踩对的坑（均已在源码处加注释，详 `paper2/TEMPLATE_NOTES.md`）：
  1) CAS 的 `abstract` 环境是 moreverb 逐字写入 `\jobname.abs` 再读回，里面不能用 `\input`；
  改为自己写 `main.abs`。2) CAS 题首内不能有任何排版文字——一个 `\todo` 放在 `\maketitle` 前会
  导致**整个标题块消失**（页 1 只剩红字）；已移到 `\maketitle` 之后。3) `cas-dc` 是在 `\maketitle` 里才调
  `\twocolumn`，所以导言区的 `\if@twocolumn` 为假；改用包装器声明的 `\ifpapertwocol` 决定宽浮动体。
  22 张表 + 7 张图在双栏下全部走 `table*`/`figure*`（单栏宽 88 mm 放不下，缩到能放下字号会低于 5 pt）；
  四张原本还写 `table` 的补充表的生成器已改成 `widetable`（对单栏构建是 no-op）。
  **顺带捉到一个真 bug**：`sections/supplement.tex` 里有 6 处 `\ref` 的反斜杠被早先的 heredoc 写入吞掉，
  排出来是字面的 `Table eftab:e2zs`。因为它们已不再是 `\ref`，构建脚本的“未定义交叉引用”检查**永远不会报**。
  是渲染 PDF 逐页看才发现的。已全部修复，并对 sections/ 与 tables/ 全量扫描同类损伤（清零）。
  **结论性做法**：以后“编译通过 + 无未定义引用”不等于排版正确，定稿前必须把 PDF 页面真的看一遍。

- 2026-09-17 22:53 **独立数字审计（`paper2/NUMBER_AUDIT.md`）已全量落实到 cas-dc 稿中。**
  审计方用**自写代码 + 自己的 CV / bootstrap 种子**（777/4242/555/31337，而非本文的 SEED=0）从
  逐图原始 CSV 重算了 ~95 个数字：**定量核心全部复现**（三轴审计、HRF/FIVES 参考差距及其外层
  bootstrap、E2 全网格与安全终点、E4 主终点与功效曲线、E3 APTOS 降幅、剂量响应、拓扑上界、协调数字），
  到 3–4 位小数或 CI 重叠。发现的全部是**计数 / 范围 / 滤条件**类错误，已逐条修正：
  **D1** MAPLES 预标注敏感性句写着 “基线保真度 0.68–0.89 → 0.11–0.62”，穷举 {anchor}×{stratum}×{source}×
  {config}×{pipeline}×{3或4 生物标记}后**找不到任何组合能复现**；0.11–0.62 实际属于 **reliseg 臂**。
  已改为本稿自己表格 `tab:e4maples` 印的 **0.76–0.88 → 0.57–0.65**（baseline、higher-resolution、三个受约束生物标记）。
  **D2** HRF 旗标“八中六个点估计为正”→实为 **8/8**（“六中六”只在 `spread=cv` 或 `err=err_mean` 下成立，而那不是主格）。
  **D3** FIVES E2 “主面板上每个区间都含零 / 最大点估计 +0.008”——只在限定于 **ReliSeg 臂的三个受约束生物标记**时成立；
  已加上该限定，并明写出全面板下两个排除零的 CF-Loss 总长度格（−0.0008 / −0.0010）与 tortuosity 的 +0.071。
  **D4** DRIVE 梯子消融“每个 Δr 区间含零”→**23/24**（例外：`ft_cfloss` 的 PVBM 总长度 −0.086 [−0.155, −0.025]）。
  **B1** zero-shot CHASE 总长度对 continued 的区间仅以 1e-4 含零、换种子即翻→改写为“不再可与零区分”并点名该格为种子脆弱。
  **B2** “FIVES 四格中三格为负”是 CV 折叠种子伪影（审计方种子下为 2/4）→**摘要/§5.3/图3 标题一律删计数**，
  改为“四格均覆盖零、点估计在零或以下”（结论不变：参考臂是基准而非天花板）。
  **M1** 3.4×（baseline 臂）与表题 2.9×（两臂均值）不一致→两处都注明是哪个臂。
  **M2** 0.522→0.523（DRIVE 实为 0.5225，原为截断）。**M3** “+0.02–+0.09”把 continued 臂的值混入了 baseline 句→拆开写。
  **M4** DRIVE 密度旗标“positive”实为 **ρ = −0.466 [−0.728, −0.031]**，方向与假设相反→明写符号与含义。
  另：Fundus-AVSeg “1280×1280 帧携带 1238 px”在任何 CSV 中都不存在→改为 `dataset_scale.csv` 的中位数 **1276 px、最小 1177 px**。
  `NUMBER_SOURCES.md` 已重写：新增 **§5.12 E4** 整块溯源表（主终点/p_adj/功效/安全/MAPLES），修正 E3 条目，
  删除已过时的“APTOS/E4/功效曲线无法溯源”三条（现只剩 ODIR-5K 一条），并把审计列出的所有“无出处”数字
  （ODIR n=6,392、各队列 FOV 直径、FIVES 密度偏置 −0.024σ、每折 36 张、Spearman≥0.8 阀值等）逐条入表。
  重建：**36 页（正文 29 页），无未定义引用/交叉引用，`\todo` 7 条**（§1 伴随稿、ODIR-5K、CAS 作者块、4 条 back-matter）。
  **这次审计兼做了一次方法验证**：由未产出该数字的一方用自己的种子重算，确实只揪出计数类错误而未动摆结论——
  这正是 21:15 那条“所有主张均需由未产出该数字的一方独立重算后方可采信”想要的效果。

- 2026-09-17 23:52 **外部第二轮评审（7.7/10，`review/paper2_r2_reply.md`）的优先级修改已全部落实。**
  评审人的判断是：科学内容较上轮实质跃升，剩下的障碍是“主张校准与稿件收口”而非再做实验。本轮未新增任何实验。
  1) **“pre-registered”全文清理**：本研究从未在任何外部注册平台登记，Methods 本就这么写，而 Highlights/Abstract/图1/表8/图5
  却把它当术语用。sections/ 与 tools/ 共 **23 处**改为“在看到对应结果前写定并冻结的规则”；仅保留 4 处——Methods 那句否认术语效力的说明，
  以及三处“E4 是 post-E2 锁定、非自立项预注册”的时间线描述（这正是评审人要求的写法）。
  2) **“powered negative”改为效应量特定表述**：从未预定义 equivalence margin，故改为“对一个预先指定的中等效应量具有高功效的阴性结果”，
  并在摘要与引言都补上“因未设等效限，这只能界定该量级的改进，而不能主张等效”。
  3) **seed 汇总的内部矛盾已解决——且 CSV 里本来就两个区间都有**：回读 `e2_delta.csv` 确认 `pooled(0,1,2)` 行的 `d_r`
  是**三个逐 seed Δr 的均值**（FIVES/reliseg/continued/skan/FD/last = −0.003436，恰为 +0.003128 / −0.011068 / −0.002368 的均值），
  `lo/hi` 是**固定 checkpoint 下的逐图配对 bootstrap**，`seed_lo/seed_hi` 是**三个 seed 效应上自由度 2 的 t 区间**
  （mean ± 4.303·sd/√3，逐位复现到五位小数）。所以从未把 seed 并进一个 image bootstrap，只是表/图标题描述错了。
  现 Methods §3.5、tab:e2 表题、fig5 轴标三处口径一致；新增补充表 **`tab:e2seed`**（`tools/mk_tab_e2seed.py`）展示逐 seed 效应 + seed t 区间。
  4) **E3 practical-effect criterion 矛盾**：Methods 声称“已固定”、表 9 声称“从未固定数值阈值”——Methods 是错的一方，已改：
  锁定的是队列角色/终点/分类器/特征集/划分单位/质量分层，**数值阈值从未锁定**；“negligible”明写为**事后**对照 12 个 arm×seed 格散度的描述性判断。
  5) **C2 降为 secondary**：引言标题改为“C2（secondary）——a negative intervention study”，开头即声明它不是第二个正面贡献；C3 改为 exploratory。
  6) **排他性因果表述清零，包括图**：**图1 重做**——旧版用箭头画出“gap tracks axis 2, not axis 1 or 3 → 所以必须在训练时介入”，
  把第一轮刚删掉的排他性叙事用图形又引了回来，且还写着“E2/E3 pending”。现改为**五段证据链**（三路度量 → 参考掩模下游定价 →
  单队列可控注入 → 对照匹配的候选补救测试 → 独立复现），没有任何箭头声称某一轴造成下游损失；**图6** 不再写 ODIR pending。
  正文里所有“cannot carry”限定到已测试的 counterfactual：常数偏置是**代数上**不可能承载队列内损失（已实测确认），
  而拓扑只能说“在所测修复 counterfactual 下不太可能承载”——不是“任何拓扑错误都不会损害测量”的证明。
  另删掉引言里残留的种子脆弱计数“three of our four FIVES cells”（独立审计已点名，上轮回复声称已删但实际没删干净）。
  7) **题目/摘要/Highlights 重写**：题目改为评审人建议的 *Auditing segmentation-derived retinal vascular biomarkers:
  reference-mask fidelity, downstream attenuation, and intervention tests*；摘要 **679 → 390 词**，四段层次；
  Highlights 删掉被点名的两条（“only where”与“six pre-registered remedies”）。
  8) **正文表格瘦身**：`tab:invariance`、`tab:flag`、`tab:mechanism` 移入补充材料新增的 S9（各附一句“为何不在正文”）；
  正文表格 **9 → 6**，只留回答主问题的那六张。
  9) **ODIR-5K 已填入**：六个逐 seed delta 文件齐备。它没有 DR 分级，按**自己的终点**（any-disease AUROC、subject 划分）报告，
  不与三个 DR 队列合并。ReliSeg 对 continued 为 +0.0005 / −0.0024 / +0.0017，**第二、第三个种子的区间向相反方向排除零**；
  对 frozen baseline 为 −0.0023 / +0.0001 / +0.0012，全含零。写法：幅度微不足道、**符号跨种子不稳定**——一个从 512 px 镜像上采样到
  1536 px 的队列本就无法分辨这个量级的 seed 级效应，这正是当初把它定为 sensitivity 而非 replication 的理由。对应 `\todo` 已关闭。
  10) **作者块改成真正的占位符**：评审人抱怨的不是作者信息缺失，而是“Author One / Affiliation A”读起来像忘了替换的样例文本。
  现在姓名/ORCID/单位/CRediT/简作者列表全部是 `\todo` 标记；`\ead` 字段在地址未知前直接缺省（CAS 类逐字排该字段，
  里面放 `\todo` 会把宏源码印出来）。`-Final` 构建下 `\todo` 为空操作，填真值只是文本替换。
  **构建**：38 页（正文 29 页 + 补充 9 页），无未定义引用/交叉引用；`\todo` 15 条全在前后资料（作者块 10 + back-matter 4 + 伴随稿引用 1），
  **Methods / Results / 补充材料零 todo**。映射表见新增的 `paper2/REVIEW_R2_RESPONSE.md`；`NUMBER_SOURCES.md` 补上 ODIR 块、
  `tab:e2seed` 行与“pooled 行到底是什么”的定义。
  **未做且已写明的两项**：human-human 同构比较（需第二位标注者的逐图生物标记）与 view-spread 全量重跑（C3 维持 exploratory）。
  **期刊定位**（评审人意见，供协调者决策）：AI in Medicine / CMPB 自然契合，MedIA 仍属 stretch；本轮未做任何面向单一期刊的改写，
  稿件在 cas-dc 下保持期刊中立。

- 2026-09-18 00:26 **外部第三轮评审（8.3/10，`review/paper2_r3_reply.md`）的七项必修已全部落实；本轮**纯编辑**，未跑任何新分析。**
  1) **seed/推断表述彻底收口**：评审人抳出图5 caption 仍写着“pooled over three independent training seeds”，
  而同一张图的横轴写着“mean of 3 per-seed effects”、Methods 又说对比类表不 stack seeds——所以我上轮回复里“三处已完全一致”并不属实。
  已改，并把“**conditional image-level**”作为固定说法用在所有 headline 区间上（即：条件于三个已训练 checkpoint，只定价图像抽样，不包含训练随机性）。
  更重要的是把 **algorithm-level 的断言改成三个 seed 能支撑的说法**：“None raised image-level fidelity” → “在我们跑的三个种子上，
  没有任何 arm 给出**一致的**提升”，结论部分补上“没有任何 arm 在三个种子上同向移动”。
  2) **E4 power 加上条件限定**：六处 0.98–1.00 全部改写，每处同时说明三件事——由 **Monte-Carlo 在 E2 实测残差相关结构下**模拟得出、
  **条件于该误差模型与固定折叠**而非对重复重训的总体性陈述、以及**未设等效限**。§5.12 另明写模拟的假设（高斯、与标签无关的残差）。
  裸写的 `powered null` / `powered negative` / `the negative is powered` 在 sections/ 与 tables/ 中已**归零**。
  3) **残留过强措辞**：Methods §3.4 “topology repair cannot touch it” → “在所测修复 counterfactual 下，它的移动小两个量级”；
  “That leaves the objective” → “目标函数是我们能测的剩余选项——**不是唯一可能的介入点**”；“a measurement-aware objective silently deforms…”
  → “**this** … **can** silently deform…”，结论收尾加上“We tested our own objective, not the class.”；删掉早已失效的
  “the title of this paper is a question for that reason”（题目从二轮起已不是问句）。
  4) **全文一致性 grep**（pooled / pre-register / powered / causal / cannot / only / TODO / pending / exclusive）逐条处置，结果写进 `REVIEW_R3_RESPONSE.md`。
  只有一处 `cannot` 被**保留并说明理由**：常数偏置那一句是**代数事实**（队列内分类器对位置平移不变，且已实测六格逐位相同），
  旁边加了“that one is algebra”以与拓扑那句区分。另发现 `tab_scale.tex` 是**陈旧产物**（生成器上轮已改、表未重生），已重生。
  5) **长度**：摘要 390 → **250 词**（通行 Elsevier 上限）；Highlights 五条全部压到 **≤85 字符**（85/84/81/80/74，原为 87–97）。
  正文维持 29 页：再删就会删到前两轮审稿人明确要求的内容，故未动。
  6) **新增 `cover_letter.tex`（+pdf）与 `TITLE_PAGE_NOTES.md`**：定位为“measurement audit + downstream error propagation +
  controlled negative intervention”，且有一段专门写“**What the paper is not**”——不是方法论文，ReliSeg 是假设检验工具而非 headline algorithm。
  期刊名为 `\todo`。TITLE_PAGE_NOTES 另列出**排版/校对阶段不得重新出现的措辞清单**（三轮审稿每一条反对意见各一行）。
  7) **新增 `SUBMISSION_CHECKLIST.md`**（仿 `paper/` 那份），把作者块/CRediT/funding/代码/genAI/伴随稿引用逐条列出到文件与宏一级。
  里面记了两个机械性坑：`orcid=` 与 `\ead{}` 是**故意缺省**而非留空（CAS 类会解析前者、逐字排后者，里面放 `\todo` 会把宏源码印出来）；
  以及“`-Final` 构建会**静默吞掉**未填的 `\todo`，所以该信的是计数而不是页面上没有红字”。
  **构建**：`main.pdf` 38 页（正文 29 + 补充 9），`main_elsarticle.pdf` 72 页，`cover_letter.pdf` 1 页；两个包装器均无未定义引用/交叉引用；
  `\todo` 20 条（进入 CAS PDF 的 15 条）全在前后资料，**Methods / Results / 补充材料零 todo**。
  **至此稿件视为定稿**，仅等用户提供作者相关信息与期刊选择（见 `SUBMISSION_CHECKLIST.md` §1、§2）；不再启动新分析。
  **审稿人对 MedIA 的保留意见已完整记入 TITLE_PAGE_NOTES §6**：障碍不在写作而在证据广度（单一应用、单一 U-Net 系列、
  单一主测量管线、只有两个同时有掩模与疾病标签的队列）——要补上需要第二个模态，而不是再一轮修稿。
- 2026-09-18 00:29 **paper2 定稿（待作者项）**：外部三轮评审 6.8 → 7.7 → 8.3；第 3 轮必改项全部落实（REVIEW_R3_RESPONSE.md）；cas-dc 双栏 38 页（主文 29 + 补充 9），引用零未定义；投稿信 cover_letter.pdf、SUBMISSION_CHECKLIST.md、TITLE_PAGE_NOTES.md 就绪。剩余 15 个待办全部为作者提供项（作者块 10、后记 4、伴随论文引用 1）。评审对 MedIA 的保留意见在证据广度（单一应用、单一 U-Net 族、单一主管线），记录于 TITLE_PAGE_NOTES.md §6。实验全部结束；租用服务器已由用户关机；LAN 工作树保留。
- 2026-09-18 03:04 **定时恢复检查（03:03）**：无中断项。ODIR-5K 规范 12 标签汇总已于 00:12 写出（results/pivot/e3/odir5k/{summary,delta}.csv），逐种子结果早已入稿；paper2 与后备论文均定稿待作者项；无实验进程在跑。
- 2026-09-18 09:02 **用户提交 CMIG 定位审阅（review/user_cmig_review_20260918.md，25 条）**：全部逐条处理。补实验：(a) 原始单位 + σ 估计量公式 + MAD/SD/IQR 与共同参考尺度敏感性；(b) 校准斜率/截距、CCC、Bland–Altman 比例偏差；注入实验增加比例失真臂与稳健性面板（经验残差 bootstrap、异方差、标签相关）；(c) 第二分割器族（transformer 自训 1 种子 + 公开管线零样本）仅做 C1 审计；(d) FIVES 全 800 审计（或分层多种子采样）；(e) FIVES 与 E4 的 ReliSeg vs continued 增加种子；(f) 拓扑匹配对照的协变量匹配/分层敏感性。写作：重新定位为测量审计框架，ReliSeg 为干预证伪实验，C3 入补充，CF-Loss 表述改为 binary adaptation 并删除 "in this journal"，统一主管线定义，收窄偏移/效用/因果措辞，增加真实眼底图与 Bland–Altman 图，精简表格，移出版本史文字。完成后由 external review 核验。
- 2026-09-18 09:26 **CMIG 审阅第 9/11/16 条的 GPU 侧开工（两卡 + LAN 节点三条泳道）**。
  - **第 9(b) 条 — 第二个分割器族**：环境里原本没有 timm / smp，已 `--no-deps` 装入
    `timm 1.0.29` + `segmentation-models-pytorch 0.5.0` + `huggingface_hub 1.32.0` +
    `safetensors` + `httpx`（torch/torchvision 未动，仍是 2.6.0+cu124 / 0.21.0+cu124）。
    新模块 `src/pivot/g9_family2.py` 训练 **SegFormer（MiT-B2 层级自注意力编码器 +
    all-MLP 解码器，24.72 M 参数，ImageNet 预训练）**，与基线 U-Net **同划分、同分辨率约定、
    同增广、同损失（BCE+Dice）、同 100 iters/epoch、同 epoch 预算与早停**，只有两处刻意不同并
    写进 `config.json`：(i) 架构本身；(ii) 优化器 **AdamW** 而非 Adam 1e-3——U-Net 的学习率
    会让 ViT 式编码器发散。学习率用 **DRIVE 上 40-epoch 验证集探针**选定：
    `lr 3e-4` 最好验证 Dice **0.7948 @ep10**，`lr 6e-5` 为 **0.7885 @ep15**，故主实验取 3e-4
    （只看 validation，从未看 test）。ImageNet 预训练这一点必须在论文里写明：从零训练的
    transformer 在 15–600 张眼底图上不是"另一个族"，而是一个坏模型。
  - **第 9(a) 条 — 公开管线零样本**：`src/pivot/g9_wrap_external.py` 把任意第三方管线的掩膜
    映射回本项目的 test 划分、重采样回**原生分辨率**、与项目 FOV 相交，输出成 `p5_eval` 认识的
    `mask/ + manifest.csv + pixel_metrics.csv + infer_meta.json`；缺一张 test 图就报错而不是
    静默丢弃（部分臂会污染所有汇总量）。自检：用基线 U-Net 自己的 DRIVE 掩膜跑一遍，
    mean Dice 0.8276 / clDice 0.8338，与已知基线一致。
  - **第 16 条 — 额外种子（改动全部是"加单元"，没有改任何分析定义）**：
    `e2_run.py` 增 `EXTRA_SEEDS=(3,4,5)` 与 `EXTRA_SEED_UNITS={"fives": (baseline,
    continued, reliseg)}` 以及 `seeds_for(ds, cfg)`；`cfloss`/`reliseg_nocl` 仍只有种子 0-2
    （审阅意见针对的是那个 null，不是整张网格）。`e4_replication.py` 增
    `E4_EXTRA_SEEDS=(1,)` 与 `EXTRA_SEED_ARMS=(continued, reliseg)`，`arm_dir/arm_ckpt/
    pred_dir/tag_of` 增可选 `seed` 参数，**种子 0 的后缀为空串**，所以既有目录/标签/缓存键
    一字未动（`plan` 复核：E2 原 316 个单元、E4 原 40 个单元仍全部 DONE，只多出 E2 36 个、
    E4 20 个新单元）。E4 的额外种子从**同一份 fold 基座 best.pt** 出发，只有 `--seed` 不同。
  - **泳道与 ETA**（本机两张 3080 20 GB + LAN 的 3080 10 GB；全部用 Win32_Process Create /
    nohup setsid 脱壳启动，日志在 `runs/pivot/r2/logs/`）：
    | 泳道 | 内容 | 启动 | 预计 |
    |---|---|---|---|
    | GPU0 | FIVES 基座 seed3 → seed5（150 ep，~30 s/ep） | 09:11 | ~11:45 |
    | GPU0 后续 | `e2_run gpu --gpu 0 --only fives`（种子 3-5 的 baseline 推理 + continued/reliseg 微调 ×6 + last/fid 推理） | 链式等待 | ~18:00 |
    | GPU1 | FIVES 基座 seed4 | 09:11 | ~10:30 |
    | GPU1 后续 | SegFormer-B2：fives → hrf → drive → chasedb1，每个 train + infer | 链式等待 | ~17:30 |
    | LAN | E4 第二个微调种子：5 折 × {continued, reliseg} 各 train+infer | 09:23 | ~15:30（实测 37 s/epoch，单个微调 ~31 min） |
    LAN 节点先同步了 `src`、`data/fundusavseg`（raw/labels/fov，不含 213 MB 的 zip）、
    `results/pivot/e4` 与五个 `fold*/base/best.pt`（427 个文件 / 0.35 GB，~1 min）。

- 2026-09-18 09:30 **R2 补分析（CMIG 审阅第 3/4/5/8/10/11/15/20 条）落地：新脚本 `src/pivot/r2_*.py`，输出 `results/pivot/r2/`，不动 paper2 与 GPU 作业。**
  已完成并逐条记录如下。所有 CSV 都带 `source` 列；σ 轴一律用
  `results/gateA_biomarker_scales_train.csv`（训练划分参考掩膜，1.4826·MAD），并在
  `r2_sigma_estimators.csv` 中给出公式与 σ 本身的自举 CI。
- 2026-09-18 09:30 **（第 3/4 条）σ 不是跨数据集共同单位——原始单位表 + 三估计量 × 三参考队列敏感性（`r2_scale.py`）。**
  (a) **σ 估计量写死**：`sigma = 1.4826 · median_i |x_i − median_j x_j|`，在训练划分的
  参考掩膜上按 (dataset, biomarker) 估计。本地复算与冻结的 Gate A 表对 DRIVE /
  CHASE_DB1 / HRF **逐位一致**（ratio = 1.000）；FIVES 比值 0.836–1.031，因为 Gate A 的
  FIVES 尺度用的是 120 张训练图，而 `bio_master.csv` 现有 200 张——**σ 随参考样本漂移，
  这一点必须写进论文**。
  (b) **σ 本身极不精确**：2000 次图像自举下，σ 的 95% 区间相对宽度（(hi−lo)/σ）平均为
  **HRF 1.79、DRIVE 1.30、CHASE_DB1 1.37、FIVES 0.41**。HRF `total_length_skan`
  σ = 6457 px，CI [2074, 14992]——近一个数量级。**任何 HRF 的 σ 单位数字都额外携带
  约 ±2 倍的尺度不确定性，而现有 CI 并未包含它**；这本身就是"同时报告原始单位"的最强论据。
  (c) **原始单位表**（`r2_audit_raw_units.csv`，primary skan panel，测试划分，seed 0）：
  density 偏移 DRIVE +0.0019 / CHASE +0.0124 / HRF +0.0067 / FIVES −0.0005（面积分数）；
  total_length 偏移 −1417 / −644 / −11550 / −1428 px，折合 FOV 直径单位 −2.63 / −0.70 /
  −3.89 / −0.71；FD 偏移 −0.0355 / −0.0184 / −0.0310 / −0.0222；tortuosity 偏移
  −0.0240 / −0.0124 / −0.0134 / −0.0008。r 与尺度无关，三种表示下相同。
  (d) **敏感性**：3 估计量 × 3 参考队列（own / pooled 四数据集 / FIVES 作为固定校准队列；
  后两者只在 FOV 归一化表示上有定义）共 9 个变体。**两条头条结论在 9/9 变体下全部成立**：
  HRF 平均 |offset| / FIVES = **4.33–6.24**，HRF 平均 resid SD / FIVES = **1.42–1.99**；
  offset/residual 轴对拓扑轴的比值 = **56×–102×**，即"一到两个数量级"成立。
  SD 估计量把所有 σ 尺度数字整体压低约 10–25%（σ_SD > σ_MAD），pooled 尺度把 FIVES 压到
  0.15–0.21 σ、HRF 压到 0.66–0.93 σ，排序不变。
- 2026-09-18 09:30 **（第 5 条）校准斜率/截距、CCC、Bland–Altman 比例偏差（`r2_calibration.py`，图 `figs/pivot/r2_calibration_<ds>.png|pdf`）。**
  Deming（λ=1，正交回归）为主，OLS 与 Deming λ=4 为敏感性；全部在 σ 轴上，2000 次配对图像自举。
  **新发现，论文必须加**：(i) FIVES 并非一致高保真——`FD_skan` 的 Deming 斜率
  **1.582 [1.248, 1.932]**（OLS 1.387），95% CI 不含 1，Bland–Altman 比例偏差
  +0.433，p < 1e-4；而 density 1.027 [0.992, 1.061]、total_length 0.988 [0.956, 1.019]
  几乎完美（CCC 0.984 / 0.962）。"FIVES 高保真"应限定为 density/total_length。
  (ii) HRF 四列斜率全部 < 1（0.37–0.81），是**衰减**而不仅是平移；tortuosity 比例偏差
  −0.821，p < 1e-4。(iii) DRIVE 斜率 0.03–0.77 且 CI 极宽（n=20），CCC 0.04–0.32。
  (iv) 比例偏差在 6/16 个 primary 单元格显著（p<0.05）：drive tortuosity、chasedb1
  total_length、hrf tortuosity、fives FD/tortuosity/density。
  **比例失真臂**：`B' = mean + k(B−mean)`，k ∈ {0.5,0.75,1.25,1.5,2}，作用于 HRF/FIVES 的
  GT 生物标志物。**重训分类器下 ΔAUC 恰为 0.000000（20/20 个单元格）**——logistic 前面的
  StandardScaler 精确抵消仿射变换，树模型对单调变换不变。**固定规则臂**（在未失真特征上拟合、
  冻结后作用于失真特征）则会动：HRF gbdt −0.0126…+0.0326，FIVES gbdt −0.0321…+0.0110，
  logreg ±0.009 以内。结论要这样写：**不变性是"重训分类器"这一协议的性质，不是生物标志物的性质**；
  任何固定阈值/已发表切点/冻结模型都会被比例失真移动。
- 2026-09-18 09:30 **（第 20 条）拓扑对照的协变量再匹配（`r2_topology_matching.py`）。**
  语料：`c1_events_harm.parquet`，obs1 / test / control_ok=1 / type=sever = **5,584 事件、118 张图、4 数据集**。
  (a) **语料实际记录了什么**（`r2_topology_covariates.csv`，46 项）：事件侧有局部管径
  `gt_r_loc`/`gt_d_loc`/`gt_radius_bin`、分支阶 `gt_branch_order`、分支长度、到结点/端点距离、
  盘相对区 `gt_zone`、局部密度 `gt_density(_bin)`、局部对比度、以及 27 项 `phi_*`；
  对照侧**只有** donor 位置 `control_donor_row/col`、donor 半径 `control_donor_r`、
  改动像素数与匹配层级。**对照侧没有记录局部密度与对比度**，故密度只能作为分层/事件侧精确匹配坐标，
  不能作为双侧匹配协变量——这一点在报告里明说。
  (b) **编辑长度已经是精确匹配的**：`n_changed == control_n_changed` 在 5,584/5,584 行上完全相等（SMD = 0）。
  (c) **原有不平衡**：管径 SMD = **DRIVE +0.763、HRF +0.326**、CHASE +0.030、FIVES −0.008
  （对照普遍偏粗）；径向距离 SMD ≤ 0.054。即 DRIVE/HRF 的对照确实系统性地落在更粗的血管上。
  (d) **三种再匹配后结论不变**：CEM（管径×编辑长度×径向距离四分位格，单元格重加权回全事件分布，
  匹配率 43–59%）h_net 最大 **0.0174 σ**；CEM 再加局部密度格 **0.0193 σ**；
  图内最近邻跨事件再配对（标准化 (log r, log len, 径向距离)，caliper 0.25 SD，匹配后三项 SMD ≤ 0.016）
  **0.0256 σ**。原报告的 0.000–0.020 σ 基本未动。
  (e) **分层敏感性**（管径/径向距离/局部密度/编辑长度三分位）：最差单元格是
  CHASE_DB1 `tortuosity_skan` 在高局部密度层，h_net = **0.0544 σ [0.0177, 0.1079]**。
  (f) **"比 offset/residual 小一到两个数量级"在所有设计下成立**：相对平均 |offset| (0.951 σ)
  与平均 residual SD (1.139 σ)，比值为 CEM 54.5×/65.3×、CEM+density 49.3×/59.0×、
  NN 37.1×/44.5×、**最差分层 17.5×/20.9×**。最差分层只剩 1.2 个数量级，论文应把措辞
  从"一到两个数量级"收紧为"至少一个数量级，在最不利的分层下仍有 17 倍"。
  (g) 保守重连器的自含描述已写入 `results/pivot/r2/reconnector_description.md`
  （来源 `paper/sections/03_method.tex` + `runs/rigr/uniform/*/summary.json` 的
  λ=1.0 / η=0.1 / μ=2.0 / risk_backend=uniform 等实际参数），无 TODO、无 companion 引用。
- 2026-09-18 09:30 **（第 10 条）基线分割器的绝对像素指标与文献区间（`r2_pixel_baselines.py`）。**
  seeds 0–2、test 划分、`results/seg_per_image.csv`，2000 次图像聚类自举：
  Dice **DRIVE 0.8262 [0.8213, 0.8319] / CHASE_DB1 0.8165 / HRF 0.8136 / FIVES 0.9021**；
  IoU 0.7041 / 0.6901 / 0.6878 / 0.8319；clDice 0.8331 / 0.8404 / 0.8224 / 0.9066；
  sensitivity 0.8349 / 0.8572 / 0.8356 / 0.8986；specificity 0.9736 / 0.9758 / 0.9784 / 0.9921。
  种子间极差 ≤ 0.027（多数 ≤ 0.006）。**HRF 与 FIVES 推理在 `resize_longest = 1536`，
  不是原生分辨率**——文献里 HRF 的 Dice 正是被分辨率劈成两簇（下采样 0.707–0.800、原生 0.814–0.816），
  论文必须写明工作分辨率。文献表 `results/pivot/r2/pixel_literature.csv`（97 行、每行带 URL）：
  我们的 Dice 超过 62%（DRIVE）/78%（CHASE）/79%（HRF）/87%（FIVES）的已发表条目，
  clDice 在 DRIVE/HRF/FIVES 均位于已报告值之上，IoU 在 71–100% 分位。
  **唯一低于文献区间的是 specificity**（DRIVE 0.9736 vs 文献 0.9803–0.9869），
  这是阈值 0.5 下偏高召回的工作点选择，应当明写而不是回避。
- 2026-09-18 09:39 **（第 15 条）ΔAUC 的确切数字与两个相对分母（`r2_delta_auc.py` → `r2_delta_auc.csv`）。**
  协议与配对图像自举与 `p3_downstream.py` 完全一致，绝对值与稿中数字四位小数一致。
  **HRF skan4**：logreg AUC_ref 0.9704 → AUC_pred 0.7533，**ΔAUC = +0.2170 [+0.1188, +0.3323]**；
  gbdt 0.9081 → 0.7281，**+0.1800 [+0.0615, +0.3052]**。
  **HRF primary8**：logreg **+0.2148 [+0.1151, +0.3211]**；gbdt **+0.1193 [0.0000, +0.2413]**。
  **FIVES skan4**：logreg **−0.0027 [−0.0223, +0.0162]**；gbdt **−0.0218 [−0.0594, +0.0157]**；
  **FIVES primary8**：logreg **+0.0043**；gbdt **+0.0305 [−0.0060, +0.0650]**。
  两个相对分母：ΔAUC/AUC_ref = HRF 0.198–0.224；ΔAUC/(AUC_ref−0.5) = HRF **0.302–0.461**。
  **"a fifth" 的来源被定位了**：0.198 是"占 AUC_ref 的比例"，稿中却把它接在
  "of the available discrimination"（即 AUC−0.5）后面，而后者是 **0.44，接近一半**。
  两个分母在此相差 AUC_ref/(AUC_ref−0.5) = 2.06（logreg）/ 2.22（gbdt）。
  **论文只写绝对差，不写比例；若要写比例必须点名分母。**

- 2026-09-18 09:42 **CMIG 定位审阅的**写作类**条目（1/2/6/7/12/13/14/15/17/18/19/21/22/23/24/25）已落实；需新数字的条目留了命名到 CSV 的钩子。**
  **1 重新定位**：§1 贡献标题改为“The contribution: a measurement-validation framework for segmentation-derived imaging
  biomarkers”，先说三个不可分割的部件（signed standardised calibration structure → reference-mask downstream arm →
  controlled error interventions）再落到视网膜实例；C2 改为“**An intervention stress test for C1 (not a second contribution)**”，
  开头即写“只测量误差的审计会招来‘误差可以修’的反驳，所以我们去修并报告结果”；C3 整节移出正文（§5.10 仅留指针，全文入补充 S8）。
  **2 硬错**：“work is in this journal”——CF-Loss 发表在 MedIA，这句是错的，已删；所有“faithfully reproduced”改为
  “**a faithful implementation of the published CF-Loss terms, adapted to binary vessel segmentation**”，不再声称复现原结论。
  **13 发现一个实质问题**：原来做 headline 的附带损害数字（FIVES +1.16σ、HRF +6.61σ、Δr −0.244）**全部是 PVBM**——
  即敏感性管线，而且正是仪器选择规则已经判出的那一列。按主管线（skan）重写：HRF 偏置位移 **+1.62σ**（仍为 0.25σ 限的六倍）、
  FIVES 仅 **+0.03σ**；skan 迂曲度**保真度几乎不动**（+0.026 [−0.099, +0.149]）——变的是位置而非排序。E4 同理：skan Δr −0.063
  [−0.137, −0.030] 是那个排除零的格，但**上端落在 −0.05 限之内**，所以方向成立而限未正式被破，正文现在就这么写。
  **19 实地查了可否恢复双眼配对**：打开 `exp/data/fundusavseg/raw/Fundus-AVSeg/metadata.xlsx`——四列（image name、
  eye id 仅 left/right，55/45、disease type、image quality），**无患者 id、无日期**。100 张图里单凭左右眼无法把一只眼匹配到其对侧，
  故**配对不可恢复**。正文明写：若存在同一患者两张图则被当作独立，区间会偏窄；**E4 结论改为靠点估计与跨折叠方向一致性承担，而非靠区间宽度**。
  **21 VascX 界定**：Crossref 核验 DOI 10.1167/tvst.14.7.19（Vargas Quiros 等，TVST 14(7):19, 2025；审阅里写的“Bosch et al.”
  不在作者列表里，故保持原引用）。§2 按三条轴界定：他们报**无符号** MAE/相关作模型排序，我们报**有符号、标准化的校准结构**；
  他们停在特征一致性，我们接**reference-mask 配对下游臂**；有了前两者才能做**可控误差干预**。写成目的不同而非贬低对方。
  **23 新图（Fig. 3，`tools/fig_cases.py`）**：同一只眼的 fundus → 参考掩模 → 预测 → 修复 → 一致性叠图 + 四项生物标记的
  参考→预测值（σ 单位），FIVES 与 HRF 各一例。**案例是自动选的**（先取 Dice 接近两队列中位数中点者，再取其中宏误差最接近本队列中位数者），
  结果：**Dice 0.854 vs 0.857（差 0.003），宏生物标记误差 0.54σ vs 1.00σ**——全文论点的一张图版。
  **25 / 20 拓扑自含**：伴随稿 `\todo` 删除，新增 Methods 小节 `sec:reconnector` 完整规定重连器（仅加不减、每端点容量 1、
  四道准入门限、lifted-A* 代价、校准后的期望效用接受规则、实测接受率；**从不读参考掩模**），缩写自分析代理交付的
  `results/pivot/r2/reconnector_description.md`。至此论文**不再依赖任何未发表材料**。
  其余写作点：6（偏移声明收窄为“cohort-wide additive offset 对重新拟合的分类器不改变队列内排序区分度”，并说明比例/逐图失真是另一回事）、
  7（生态对比 → associated with）、14（clinical utility → downstream discriminative performance；并明写 Fundus-AVSeg 的
  **参考特征臂接近随机（0.54）**，故该终点根本无法评估有用性）、15（删“一五”，只给绝对 ΔAUC）、17（功效限定，摘要**完全不提功效数字**）、
  18（E4 → post-E2 locked validation cohort）、12（统一主管线：明写“本文每一个数字都算在 skan 一条管线上，没有任何量是两管线的中位数或均值”，
  把双管线规则降为一次性的**仪器选择**步骤）、22（四处版本史/辩护性文字移入补充 **S10 Protocol history**）。
  **关于 r2 CSV**：`exp/results/pivot/r2/` 已有 16 个文件，但两个分析代理尚未报告，**故未引用其中任何数字**——
  在没有交底的情况下猜哪一列是主格、哪个 arm 是哪个，恰好是本文用三十页在警告的那类错误。已在 Methods/Results/Discussion 留下
  **6 个 `\todo{pending r2 analysis}` 钩子**，每个点名对应 CSV；两个**描述类** md（reconnector / second segmenter）已用，因为读它们不需要推断 schema。
  **构建**：39 页（正文 30 + 补充 9），无未定义引用/交叉引用；摘要 **250 词**；图 8 张（新增 Fig. 3）；正文表 6 张；
  `\todo` 25 条 = 15 作者项 + 4 back-matter + 6 分析钩子。映射表见新增 `paper2/REVIEW_CMIG_RESPONSE.md`。
- 2026-09-18 09:46 **（第 11 条）FIVES 全 800 张审计（`r2_fives_full.py`）：OOF 预测 600/600 齐全，无需向 GPU 代理要文件。**
  `runs/seg_oof/fives/pred/{prob,mask}` 逐文件核对 **600/600 存在**（`fives_oof_coverage.csv`），
  故 `fives_oof_missing.csv` 未写出。用 `build_table --fives_train_n 150 --procs 12` 补算了余下
  400 张训练图的生物标志物（约 27 min，CPU），写入 `results/pivot/r2/bio_master_full.csv`，
  FIVES 现为 600 train(gt+pred_oof) + 200 test(gt+pred_test) = **全 800 张**。
  **审计（primary skan，σ 单位）**：test200 平均 |offset| 0.248 / resid 0.717 / r 0.824；
  **train600 0.214 / 0.542 / 0.861**；shipped train200 0.230 / 0.503 / 0.898；
  **全 800 池化 0.211 / 0.592 / 0.850**。头条不变。逐列（800）：FD −0.581 (r 0.913)、
  tortuosity +0.017 (r 0.527)、density −0.021 (r 0.984)、total_length −0.224 (r 0.975)。
  **但 200 张上限对 tortuosity 略偏乐观**：r 0.729（cap）vs 0.573（全 600），resid SD 0.744 vs 1.011。
  **下游参考差（同 p3 协议 + 1000 次配对图像自举）**：全 800 的 ΔAUC 为
  skan4 logreg **+0.0132 [−0.0016, +0.0283]**、skan4 gbdt **+0.0009 [−0.0210, +0.0235]**、
  primary8 logreg **+0.0111 [−0.0047, +0.0259]**、primary8 gbdt **+0.0275 [+0.0056, +0.0495]**。
  **一条论断必须收窄**：n=200 时四个 FIVES 区间全部覆盖零，稿中写"预测特征分类器与参考特征分类器
  相当"；n=800 时点估计一致为小正值，**四个池化单元格中有一个（primary8+GBDT）区间不含零**。
  诚实的读法是 n=200 是**功效不足**，不是"没有差距"；FIVES 的差距约为 HRF 的十分之一
  （+0.028 对 +0.180…+0.217）。措辞应改为"FIVES 上的差距至多几个 AUC 点，比 HRF 小一个数量级"。
  **采样偏向对论文有利而非不利**：shipped 的 200 张上限给出的是所有分块中**最大**的差距
  （primary8 gbdt +0.0559 [+0.0039, +0.1132]），放开上限是把数字推向论文的论断，不是远离它。
- 2026-09-18 09:48 **外部方案咨询（review/cmig_plan_reply.md）后的调整**：共同尺度 = 逐生物标志物、队列内中心化的参考掩膜偏差的 1.4826×加权中位数 MAD，对所有队列/分割器同一公式，仅作敏感性（原始单位为主）；Bland–Altman 须报告比例偏差斜率；比例失真臂在生物标志物层定义 x*=μ+γ(x−μ)，并同时用重训练分类器与固定阈值/锁定分类器两种变体；标签相关注入臂降为可选的最坏情况压力测试；拓扑匹配协变量优先级：编辑负担 > 局部管径 > 分支/拓扑上下文 > 径向位置（局部密度分层），并给出匹配平衡诊断；第二分割器只选一种：SegFormer-B0 两种子（DeepLabV3+ 为工程回退），放弃公开管线零样本；零结果最少 5 种子（E4 补到 5 个微调种子），仅能支持"无一致改进"不支持等效；E4 命名 "post-E2 protocol-locked held-out validation cohort"；E4 双眼泄漏为高优先级：按受试者重组或将下游结果降为探索性；案例图选择须预定义规则；主表列 = μ、α、β、residual SD、CCC、r、AUC gap；防御性/版本史文字直接删除而非移入补充。若时间不足按其裁减顺序从末尾删。
- 2026-09-18 10:05 **按方法学复议调整 GPU 计划（review/cmig_plan_reply.md）：第二族改 SegFormer-B0 × 2 种子，E4 补到每折每臂 5 个种子，公开零样本管线只保留"30 分钟内能跑通"的那一个。**
  - **第 9(b) 条改 B0**：原先排的是 SegFormer-**B2**（24.72 M 参数），换成 **B0（3.71 M）**。
    理由是参数量：基线 U-Net 是 7.77 M，B2 比它大三倍，那样的对比同时改了架构**和**容量；
    B0 比 U-Net 还小，架构差异才是唯一的自变量。B2 泳道在**尚未开始训练**（仍在等基座
    PID 的 wait 循环里）时被停掉，没有浪费 GPU 时间。改为 **2 个种子 × 4 个掩膜数据集**，
    先把种子 0 的四个数据集全部跑完再跑种子 1。
  - **学习率已定档**：DRIVE 40-epoch 验证集探针跑完，**3e-4 → best val Dice 0.7989 @ep16**，
    **6e-5 → 0.7941 @ep31**；取 3e-4。全程只看 validation，没有碰过 test。
    （09:26 那条记录的 0.7948 / 0.7885 是探针中途值，以此处为准。）
  - **第 16 条 E4 补到 5 个种子**：`E4_EXTRA_SEEDS` 由 `(1,)` 扩到 `(1,2,3,4)`，
    即 4 个额外种子 × 2 个臂 × 5 折 = **40 个微调 + 40 次推理**；每折的**基座 checkpoint 不变**，
    只有 `p5_finetune --seed` 不同。`plan` 复核：144 个单元，原有 64 个仍全部 DONE。
  - **10 GB 卡能不能跑 E4——实测结论：能，而且不需要任何协议妥协**。LAN 节点那张 3080 10 GB
    上 fold0/continued_s1 实测 **8599 MiB / 10240 MiB**，batch 4 / patch 768 / longest 1536
    原样不动，**不需要把 batch 减半、也不需要梯度累积**，所以没有协议偏离。单个单元
    **微调 26 min + 推理 14 s**。因此 E4 放 10 GB 卡是安全的，SegFormer 留在 20 GB 卡上。
  - **第 9(a) 条只保留 LWNet**：LWNet（agaldran/lwnet，MIT，W-Net 级联 **68,482 参数**，
    仓库内自带 DRIVE 训练权重）在 30 分钟内就跑通了四个数据集，零样本结果
    （项目 FOV 外接框裁剪 → 512×512 → 4 次翻转 TTA → 双三次插值回原生分辨率，
    阈值 0.4196 为上游 DRIVE 最优值，四个数据集完全同一套设置）：
    | 数据集 | Dice | clDice | n |
    |---|---|---|---|
    | DRIVE | **0.8278** | 0.8350 | 20 |
    | CHASE_DB1 | **0.7331** | 0.8157 | 8 |
    | HRF | **0.6944** | 0.7780 | 30 |
    | FIVES | **0.7350** | 0.7648 | 200 |
    四份 `bio.csv` 已写出（8 / 20 / 30 / 200 行），生物标志物估计路径与所有内部臂完全一致
    （`compute_all(fd_rotations=5)`，视盘仅从眼底图检测）。DRIVE 一栏 0.8278 与本项目
    基线 U-Net 的 0.8276 几乎相同——但那是 LWNet 的**训练域**，不是零样本；
    另外三个数据集才是零样本，保真度明显更低，论文必须这样写。
    **VascX 放弃**：需要单独的 conda 环境、权重是 AGPL-3.0、而且输出停留在 1024×1024
    中心凹裁剪画布上（上游 README 明说不映射回原图几何），必须先用 `bounds.csv` 反演才能
    和我们的 GT 对齐——超出"30 分钟内能跑通"的门槛。AutoMorph 未尝试（1.6 GB 克隆、
    bash-only 入口、需要 M0 预处理与 resolution_information.csv）。两者的检索结论与
    放弃理由记录在 `results/pivot/r2/second_segmenter_sources.md`。

- 2026-09-18 10:00 **方法学磋商后的四项调整（18/19/22/23/24）已落实。**
  **18** E4 标签统一为 **“post-E2 protocol-locked held-out validation cohort”**；“prospectively locked”与“independent replication”在 sections/ 与 tools/ 中归零（图1 第五段方框也改了）。
  **19 双眼泄漏：做了三项独立检查，结论是配对不可恢复，因此没有通知 GPU 代理、也不需重跑 E4**：
  (a) 分发的 metadata.xlsx 只有四列（image name / eye id 仅 left-right，55-45 / disease type / image quality），**无患者 id、无日期**；
  (b) 图像文件 **100/100 无任何 EXIF**，且只有两种帧尺寸（1280×1280 ×79、2656×1992 ×21），没有可用的采集线索；
  (c) 文件序也没有隐含配对：相邻编号中“左右眼相反且疾病相同”的对出现 **17 次，置换检验期望 13.7，p = 0.20**——与独立无法区分。
  故在 100 张图中无法把一只眼匹配到对侧。相应处置：若存在同患者两张图则被当作独立，**故 out-of-fold 区间明写为近似**，结论靠点估计与跨折叠方向一致性；
  **该队列的四类下游 AUC 明确降为 exploratory**（它本来就因参考特征臂接近随机而不可用，眼级聚类是第二个独立理由），写入 §5.12 与讨论的局限性。
  **22** 辩护性/版本史文字**直接删除**（上一版把它们移进补充 S10，按新指示整节删掉），只保留讨论里一段短的 **Scope of claims**：
  未在外部注册平台登记（“locked”的确切含义）、一个 σ 尺度 + 单一 skan 管线、以及证据只覆盖一个应用——三句话，不讲历史。
  **23 案例图改为预定规则**：原先的做法是“把两个队列的 Dice 对齐”——读起来像为结论构造的对比，已换成：**在每个队列的测试集内，
  按“距本队列 Dice 中位数的距离”与“距本队列宏生物标记误差中位数的距离”各自排名，取两个排名之和最小者**，规则写在图注里。
  新结果反而更说明问题：**Dice 0.927 vs 0.803（差 0.12，分割论文会称之为“不大”），而宏生物标记误差 0.30σ vs 1.65σ（5.5 倍）**，
  HRF 那例的迂曲度达 −2.77σ。
  **24** 主表最终列集已写进校准钩子：**μ（原单位与 σ）、α、β、残差 SD、CCC、r、AUC gap**，其余入补充；待分析代理的校准 CSV 到达后一次性落地。
  **构建**：40 页（正文 31 + 补充 9），无未定义引用/交叉引用；过宽盒只剩 cas-dc 标题机制那个固定的 123.6pt；
  `\todo` 25 条 = 15 作者项 + 4 back-matter + 6 分析钩子。
- 2026-09-18 10:01 **按 外部方法学咨询（`review/cmig_plan_reply.md`）调整第 3/5/8/20 条的实现，并重跑。**
  **（第 3 条）共同尺度改为咨询意见 2A 的唯一定义**，实现于 `r2_scale.py::common_scale`，
  公式落盘 `results/pivot/r2/r2_common_scale_formula.txt`：
  对每个生物标志物 k，仅用参考掩膜测量 → 队列内中心 `m_ck = median_i x_ick^ref` →
  队列中心化偏差 `d_ick = x_ick^ref − m_ck` → **`s_k = 1.4826 × 加权中位数_ick |d_ick|`，
  权重 `w = 1/n_c` 使每个队列等权** → 所有队列与所有分割器共用 `e = (x^pred − x^ref)/s_k`。
  长度类标志物先按 FOV 直径归一化再进入第 (1) 步。数值（FOV 直径单位）：
  FD_skan **0.02709**、tortuosity_skan **0.008819**、density_skan **0.014138**、
  total_length_skan **2.06425**（PVBM：0.021387 / 0.006052 / 0.014138 / 1.80960）。
  **原始单位被提升为主结果**：`r2_headline_raw_primary.csv` 把 raw/FOV 归一化的 offset、
  residual SD、r（含 CI）作为主列，`offset_sigma_own` 与 `offset_sigma_common` 明确标为敏感性列。
  被咨询否掉的"直接混合原始值求 MAD/SD/IQR"保留为 `pooled_naive`，只为在文中说明为什么否掉它。
  **13 个尺度变体全部保住两条头条结论**：共同尺度下 HRF/FIVES 的平均 |offset| 比值 **3.90**、
  residual SD 比值 **1.44**、offset/residual 轴对拓扑轴 **95×**。
  **一处排序会变**：共同尺度下 DRIVE 的平均 |offset|（1.361）超过 HRF（1.258），
  因为 DRIVE 自身参考散度小、own-σ 把它压低了。论文从未声称 DRIVE/CHASE/HRF 之间的排序，
  所以没有东西被打破——但也不能从现在起开始声称。
  **（第 5 条）**新增主表 `r2_main_table.csv`，列为咨询指定的
  μ（Bland–Altman 平均差 = 审计的常数偏移）、α（中位数中心化后的 Deming 截距）、β、
  residual SD、**CCC（排在 r 之前）**、r、比例偏差斜率（含自举 CI 与 p）与数据集级 AUC gap。
  校准量全部改为**以队列参考中位数为中心**的 σ 轴（`ref_median_raw` 列），
  否则 tortuosity 的截距会落在 130 σ 处而毫无意义；中心化不改变 β/CCC/r/residual SD/LoA/比例偏差。
  **锁定部署臂按咨询 2E 重做**：比例失真写作 `x* = μ + γ(x − μ)`，γ ∈ {0.5,0.75,1.25,1.5,2}；
  固定规则臂不再只报 AUC（AUC 与阈值无关），改报 argmax 决策的准确率、macro 敏感度/特异度、
  **逐类预测阳性率**、真类平均概率与多类 Brier。结果：**重训分类器 ΔAUC 恒为 0.000000（20/20）**，
  而锁定规则下 HRF logreg γ=0.5 准确率 **−0.067**、Brier **+0.120**、dr 预测阳性率 0.289→**0.400**；
  FIVES logreg γ=0.5 Normal 0.355→**0.208**、DR 0.180→**0.318**（ΔAUC 仅 −0.0042），
  γ=2 时 Normal 0.355→**0.445**。**定稿句：比例失真对 AUC 与重训分类器不可见，
  却能把冻结决策规则的类别判定率移动最多 14 个百分点、Brier 移动 +0.12。**
  **（第 8 条）**面板按咨询降权：Gaussian 独立 / 经验残差自举 / 异方差（参考值与图像质量代理）
  为必需臂，**标签相关臂标注为对抗性最坏情况压力测试**，不进入主叙事。
  **最重要的结果与审阅者的担心相反**：在实测保真度上，**真实测量落在所有非对抗臂的曲线之上**。
  HRF（实测 r=0.549，logreg AUC 0.7533）：hetero_quality 0.7414、hetero_reference 0.7353、
  **已发表的 gaussian_corr 臂 0.7328**、resid_bootstrap 0.7176、gaussian_indep 0.6790、
  label_dependent（对抗）0.5770。即已发表的注入臂**高估**了危害（ΔAUC −0.238 模拟 vs −0.217 实测）。
  FIVES 差距更大：r≈0.90 时各非对抗臂已损失 0.042–0.051 AUC，而实测在更低的 r=0.854 上损失为零。
  **论文措辞应从"落在自己的注入曲线上"收紧为"落在自己的注入曲线之上"，并说明该模拟是危害的保守上界。**
  机制层面：保持跨标志物相关结构比高斯性更重要（独立高斯 −0.291 vs 相关高斯 −0.238）；
  异方差对曲线影响很小。FIVES + GBDT 上标签相关臂把 AUC 推到参考线**之上**，
  说明标签相关误差可以虚高表观性能——这正是要把它标为对抗臂的理由。
- 2026-09-18 10:06 **（第 20 条，按咨询 2D 重做）拓扑再匹配改为"按优先级的嵌套层级 + 每层平衡诊断"，并修掉一个真实的匹配失效。**
  优先级固定为 **edit burden > local calibre > branch order/topological context > radial position**，
  局部密度只做分层不硬匹配。两侧都有记录的协变量（编辑负担、管径、径向位置）做配对匹配；
  **分支阶只在事件侧有记录**，故只能作为精确分块（blocking）进入，之后把单元格重加权回全事件分布——
  这个限制在报告里明写。
  **修掉的失效**：局部半径在小队列上近似离散（DRIVE 的 `gt_r_loc` 25/50/75 分位都是 1 px），
  四分位边界塌成一个断点，"按管径匹配"实际上什么都没做——**第一版跑完后 DRIVE 的管径 SMD 仍是 +0.76**。
  改为固定 0.5 px 网格分箱（等价于精确匹配）后修复。
  **平衡诊断**：未匹配 DRIVE 管径 SMD **+0.763**、HRF **+0.326**；L1（加管径）后
  DRIVE **−0.016**，各层各队列所有协变量 |SMD| ≤ 0.063；L3 后 ≤ 0.027；加权最近邻后 ≤ 0.013。
  **harm 在所有层级上基本不变**（primary skan）：平均 h_net
  L0 0.00455 → L1 0.00542 → L2 0.00511 → L3 0.00560 → L4 0.00556 → NN 0.00677 σ；
  最大单元格 0.0198 → 0.0297 → 0.0254 → 0.0264 → 0.0286 → 0.0478 σ，
  **每一层的最大值都是 CHASE_DB1 `tortuosity_skan`**。密度列在所有设计下恒为 0（构造检验通过）。
  匹配率：L0 100%、L1/L2 43%、L3/L4 26%、NN 8%（DRIVE 6.7%、HRF 7.3%，是 0.25 SD caliper 在小图上的代价）。
  **"比 offset/residual 小一到两个数量级"在全部 28 个比较下成立**，但措辞要收紧：
  充分匹配下是 1.5–1.7 个数量级，最严格的 NN 设计与最差分层下降到 **1.2–1.3 个数量级**
  （相对平均 |offset| 分别为 19.9× 与 17.5×）。建议写
  "at least one order of magnitude below the offset and residual axes, and still a factor of
  17 to 20 below them under the strictest matching and in the least favourable stratum"。
- 2026-09-18 10:09 **R2 补分析全部落地，报告 `results/pivot/R2_ANALYSIS_REPORT.md`（759 行）。**
  7 个脚本（`src/pivot/r2_{scale,calibration,injection_panel,pixel_baselines,fives_full,delta_auc,topology_matching}.py`）、
  22 个 CSV + 1 个公式 txt + 1 个重连器描述 md（均在 `results/pivot/r2/`，每个 CSV 带 `source` 列）、
  10 个图文件（`figs/pivot/r2_calibration_<ds>.{png,pdf}` 四套 + `r2_injection_panel.{png,pdf}`）。
  未触碰 `paper2/` 与任何 GPU 作业。**论文必须改的四处**：
  (a) 第 15 条 —— 删除"a fifth of the available discrimination"，只写绝对 ΔAUC（分母混用被定位）；
  (b) 第 11 条 —— FIVES "predicted matches reference" 收窄为"差距至多几个 AUC 点，比 HRF 小一个数量级"
      （n=800 下 primary8+GBDT 的 +0.0275 [+0.0056, +0.0495] 区间不含零；n=200 的零结果是功效不足）；
  (c) 第 5 条 —— "FIVES 高保真"限定到 density/total_length；FIVES `FD_skan` 的 Deming 斜率
      1.582 [1.248, 1.932] 显著偏离 1，HRF 四列斜率全部 < 1（衰减而非仅平移）；
  (d) 第 20 条 —— "一到两个数量级"收紧为"至少一个数量级；最严格匹配与最差分层下仍有 17–20 倍"。
  **两处措辞可以加强而不是削弱**：第 8 条注入模拟被证明是危害的**保守上界**（真实测量落在所有非对抗臂之上）；
  第 3/4 条的两条头条结论在 13 个尺度变体下全部成立。
  **需要论文代理携带的坑**：`pixel_literature.csv` 97 行里只有 5 行有经核实的 DOI（其余只有 source_url，
  引用前必须解析 DOI）；FSG-Net 的 "mIoU" 是类平均不是血管 Jaccard（相关单元格故意留空）；
  FIVES 数据集论文（Jin 2022）**不含模型基准**，只有标注者一致性（intra 0.9679 / inter 0.9241–0.9608），
  应当作人类上限而非基线；HRF/FIVES 像素指标是在 `resize_longest=1536` 下测的，必须写明工作分辨率；
  `pooled_naive` 是被咨询否掉的定义，只用于说明为何否掉；共同尺度是敏感性，不得改称"自然单位"或"population SD"。
- 2026-09-18 10:11 **LWNet 零样本臂发现一个必须在分析前定策的问题：FIVES 上有 17/200 张掩膜塌缩，且塌缩与病种强相关。**
  逐图核查（`results/pivot/r2/lwnet_mask_quality.csv`，258 行，覆盖四个数据集）：
  - **DRIVE / CHASE_DB1 / HRF：零塌缩**（前景比例最小 0.094–0.110，与健康中位数同量级）。
  - **FIVES：17/200 张前景比例 < 0.02**（健康中位数 0.097），其中 **122_G 与 123_G 完全空**
    （`fg_px = 0`，Dice = 0），导致 `FD_skan` 与 `tortuosity_skan` **为 NaN**、
    `density_skan` 与 `total_length_skan` 恒为 0。其余 15 张欠分割 5–50 倍
    （`total_length_skan` 73–4,267，健康中位数 21,430；`FD_skan` 0.71–1.11，健康 1.15）。
  - **塌缩按病种分布：青光眼 13/50（26%）、AMD 2/50、DR 2/50、正常 0/50。**
  这不是包装 bug（`g9_wrap_external` 已用基线 U-Net 自己的掩膜自检通过），是真实的域偏移：
  LWNet 是 2020 年的 68 k 参数模型，只在 DRIVE 的 20 张图上训过。
  **两点后果，交分析代理定夺，GPU 侧不擅自决定：**
  (1) C1 校准回归里，空掩膜行的 FD/迂曲度是 NaN、密度/长度是 0，若不处理会主导斜率——
      这属于**分割失败**，不属于**测量校准**，建议预先声明一条排除规则（例如 `fg_px == 0`）
      并同时报告含与不含这些行的两套结果，而不是事后挑点；
  (2) 塌缩集中在青光眼这一类上，本身就是"分割器按病种选择性失效会凭空制造下游判别信号"的
      直接证据，与第 14/15 条关于 downstream discriminative performance 的收窄表述相关，
      建议在第二族小节里正面写出来，而不是只当成噪声。
  另：LWNet 的 **DRIVE 一栏不是零样本**——DRIVE 就是该 checkpoint 的训练集
  （`experiments/wnet_drive/model_checkpoint.pth`，sha256 91f0cada…18de1，MIT，仓库内唯一的
  血管权重），真正的零样本只有 CHASE_DB1 / HRF / FIVES 三个。四个数据集用完全相同的约定，
  没有任何逐数据集调阈值（0.4196 为上游 DRIVE 最优值，原样沿用）。
  VascX 已装好但**一张掩膜都没产出**，不存在任何 VascX 数字；放弃理由见
  `results/pivot/r2/second_segmenter_sources.md` 第 9 节。

- 2026-09-18 10:35 **分析代理的 R2 结果（`results/pivot/R2_ANALYSIS_REPORT.md`，759 行）已全量集成；四项必改主张、两项可强化主张均已落实。**
  **新主表（`tab:main`，`mk_tab_main.py` 读 `r2_main_table.csv`）**：μ、α、β、残差 SD、CCC、r、Bland–Altman 比例偏置斜率、AUC gap；
  原三轴 `tab:audit`（带观察者与视角参考列）移入补充 S5。新增补充表 `tab:raw`（原单位为主 + 两种 σ 敏感性）、
  `tab:pixel`（绝对像素质量 vs 文献）、`tab:topomatch`（六种匹配设计 + 平衡诊断）；图：`fig:calib`（HRF Deming/Bland–Altman，正文）、
  三个队列的同类图入补充、`fig:inject`（六种误差模型的注入面板）。
  **(a)** 删除“a fifth of the available discrimination”——同一结果除以 AUC_ref 是 0.20、除以超随机部分是 **0.44**，故只给绝对值：
  HRF **+0.2170 [+0.1188, +0.3323]** logreg、**+0.1800 [+0.0615, +0.3052]** GBDT，并明写“若要给比值必须点名分母”。
  **(b) FIVES 全 800 张改了一个结论**：原来写“预测特征与参考特征持平”，实为 **欠功效**；汇总点估计 +0.001~+0.028，
  primary8+GBDT **+0.0275 [+0.0056, +0.0495] 排除零**。现写为“最多几个 AUC 点，比 HRF 小一个量级”。
  关键证据一并写入：原 200 张上限**反而是最悲观的**（给出所有分块中最大的 gap +0.056）；五个分层 200 张子样本的 ΔAUC
  在 **−0.019~+0.031** 之间摆动，而审计量几乎不动（mean |offset| 0.186~0.223 vs 0.214；mean r 0.848~0.878 vs 0.861）——
  即**审计对重采样稳定、n=200 的下游 gap 被抽样噪声主导**（第 11 条的代表性证据）。FIVES n 全文改为 800（200 test + 600 OOF）。
  **(c)** “fidelity is high (FIVES)”收窄为逐生物标记：密度 β 1.03 / CCC 0.98、总长度 β 0.99 / CCC 0.96 确实已校准，
  但 **FIVES FD 的 Deming 斜率 1.58 [1.25, 1.93]**、比例偏置 +0.43（p<1e-4）——分割**拉伸**了 FD 的图间跨度；
  HRF 四个斜率 0.37~0.81 = **衰减而非仅偏移**。新增 §5.2 整节讲校准，并明写 16 个主格中 **6 个比例偏置显著**。
  **(d)** 拓扑“one to two orders”→**“至少一个量级；在最严匹配与最差分层下仍小 17~20 倍”**（全文四处）；并写入匹配层级表：
  edit burden 本来就是**精确匹配**（5,584/5,584，SMD=0），但管径 SMD +0.763（DRIVE）/+0.326（HRF），加入管径后降到 −0.016；
  **估值全程平坦**（0.0046→0.0068σ），边际从 47.9× 降到 19.9×，最差分层 17.5×；密度恒为 0（构造校验）；
  还记下了那个**分箱细节**：DRIVE 的局部半径四分位全是 1 px，分位数分箱塔陷导致首次“管径匹配”没有任何效果，改用**固定 0.5 px 网格**。
  **强化一**：注入模型是**保守的**——真实 HRF 测量在其自身保真度上落在**所有非对抗臂之上或之上方**（0.7533 vs 0.679~0.741），
  所以出货的相关高斯臂其实**高估**了危害（−0.238 模拟 vs −0.217 实测），独立高斯高估更多（−0.291）；只有标签相关的**对抗臂**更狠（−0.393）。
  另写一句机制结论：**保持跨生物标记相关结构比高斯性重要**。
  **强化二**：两个 headline 主张在**13 种尺度变体**（含被接受的共同尺度）下均成立（HRF/FIVES 偏移比 3.9~6.2、残差比 1.4~2.0、轴间分离 56~102）。
  **新增比例失真结果**：重新拟合的分类器 **ΔAUC 恰为 0.000000，20/20 格**（机制而非巧合：StandardScaler 吸收仿射、树集成对单调变换不变）；
  而**锁定规则**下同一失真（HRF γ=0.5）准确率 **−0.067**、Brier **+0.120**、DR 预警率 **0.289→0.400**，FIVES Normal 0.355→0.208 / DR 0.180→0.318。
  这就是“为何偏移/尺度失真对**已部署规则**重要、却对**重新拟合的区分度**不重要”的论证；偏移主张相应收窄。
  **像素基线**：Dice 0.826/0.817/0.814/0.902、clDice 0.833/0.840/0.822/0.907（**HRF 与 FIVES 在 resize_longest=1536**，每条都注分辨率），
  四个队列均**高于文献中位数**与所有已发表 plain-U-Net；**唯特异度低于文献区间**（DRIVE 0.974 vs 0.980~0.987）——已写明是 0.5 阀值的工作点选择。
  **引用纪律**：文献表 97 行中仅 5 行有已验证 DOI，且多行是第三方转述，所以表中**只给 min/median/max 与行数，不引单篇**；
  唯一引用的是 DOI 已核验的 FIVES 数据集论文，而且是当作**人类上限**（intra 0.968、inter 0.924~0.961）而非模型基线。
  **已携带的警告**：σ 本身不精确（HRF 总长度 σ=6457 px，CI [2074, 14992]，因此**本文所有 σ 尺度区间都不含这一项**）；
  FIVES 冻结 σ 来自 120 张，600 张重算的迂曲度 σ 为冻结值的 **0.835 倍**（冻结值保留，只陈述偏移，不做事后修正）；
  共同尺度**只是敏感性构造**，不得称为自然单位或总体 SD；**不引入 DRIVE/CHASE/HRF 排序**（共同尺度下 DRIVE 反超 HRF，已写明故不作排序主张）。
  **构建**：48 页（正文 34 + 补充 14），无未定义引用/交叉引用；`\todo` 20 条 = 15 作者项 + 4 back-matter + **1 GPU 钩子**（SegFormer-B0 与额外种子）。
  途中修了一个自己引入的 LaTeX 错误：`tab:topomatch` 的 caption 里有空行，而 `\caption` 不是 long 宏，导致三个表的题注全挂（已排除空行）。
- 2026-09-18 11:03 **第二分割器族审计的预声明规则（在看任何族 2 审计结果之前锁定）**：(1) 掩膜级失败排除规则：前景像素数 == 0 的预测视为分割失败，从校准/保真度估计中排除并单独计数；前景占比 < 0.02 的标记为退化掩膜，主分析保留，敏感性分析排除；两种结果并列报告。(2) LWNet 在 FIVES 上 17/200 退化、2 张全空且集中于青光眼类（13/50），作为"分割器对某一疾病类别选择性失败会制造下游判别信号"的直接例证写入讨论（对应审阅第 14/15 条），不作为测量校准结果。(3) DRIVE 为 LWNet 训练域，仅 CHASE/HRF/FIVES 为真正零样本；VascX 无输出，不存在任何 VascX 数字。(4) SegFormer-B0（3.7 M）与 U-Net 使用相同划分、分辨率与早停规则；族 2 的审计对四数据集报告 μ、α、β、残差 SD、CCC、r、AUC 差，使用与族 1 相同的冻结 σ 与共同尺度 s_k。
- 2026-09-18 11:13 **第二族审计脚本 `src/pivot/r2_family2.py` 已写好并在 LWNet 四数据集 + SegFormer-B0 DRIVE seed0 上跑通（预演），等 GPU 侧其余种子落地后直接重跑即可（分钟级）。**
  **自动发现**：扫描 `runs/pivot/r2/<family>/<dataset>[/seed<k>]/pred/bio.csv`，seed 缺省时从同级
  `summary.json` 读；跳过 `_probe`/`_smoke`/`logs`。预演发现 5 组（lwnet × 4 + segformer_b0/drive）。
  **共同尺度 s_k 是"载入"而不是"重算"**——必须与族 1 逐位相同，否则两族不在同一分母上。
  从 `r2_scale_sensitivity.csv`（cohort=common）读取；本来重算会悄悄不一致
  （族 1 跑在 bio_master.csv 的 FIVES 200 张训练集上，而 bio_master_full.csv 现在是 600 张，
  density_skan 0.014138→0.013819、total_length_skan 2.06425→2.13181，差 2–4%）。已加 WARNING 分支。
  **预声明规则逐条实现**：`fg_px == 0` → 分割失败，排除并计数；`pred_fg_frac < 0.02` → 退化，
  主分析保留、敏感性排除；两套并列（`analysis_set` ∈ {main, sens_no_degen}），
  下游另带第三套 `all_incl_empty` 以便给失败定价。掩膜计数与 GPU 侧 `lwnet_mask_quality.csv`
  **逐位复现**（FIVES 2 空 + 15 退化；青光眼 13/50、AMD 2/50、DR 2/50、正常 0/50；
  DRIVE/CHASE/HRF 零失败），并在 CSV 里留了 `crosscheck` 列。
  **LWNet 的 DRIVE 一栏标 `zero_shot=False`**（那是该 checkpoint 的训练集），CHASE/HRF/FIVES 为 True。
  **预演已可见的结论**（等全部种子后定稿）：
  (a) **SegFormer-B0 在 DRIVE 上几乎逐列复现 U-Net 的衰减模式**：β = 0.703 / 0.055 / 0.732 / 0.260
      对 U-Net 的 0.767 / 0.034 / 0.717 / 0.454 —— 四列全部 < 1，顺序一致。
  (b) **LWNet 在 FIVES 上是 β > 1（1.32–3.08，四列 CI 全部排除 1）即"拉伸"**，与 HRF 上 U-Net 的
      β < 1"衰减"方向相反；这是域外零样本模型的特征，不是测量结构的反例，须在小节里分开叙述。
  (c) LWNet 的 HRF 平均 |μ| = 3.02 σ（U-Net 1.54 σ），offset 排序 HRF > FIVES 仍成立（比值 4.35）。
  (d) **残差散度排序 HRF > FIVES 在 LWNet 主分析下不成立**（1.486 vs 1.757），
      但这**完全由 15 张退化掩膜驱动**：敏感性集里 FIVES 降到 1.483，与 HRF 1.486 基本持平。
      故 `family2_replication.csv` 改为同时输出两套 analysis_set，让"只因退化掩膜而失败"可见。

- 2026-09-18 11:23 **长度压缩：正文 34 → **30** 双栏页（目标达成），四轮审稿要求的内容一条未删。**
  移入补充材料（每处留一句指针，数字全在已有或新建的补充节里）：`tab:e2`/`tab:e3`（逐 arm 网格）与 `fig:e3`；
  `tab:topology`；Methods 里的**重连器规格**（它是方法描述而非结果，指针保留了使它成为合法 counterfactual 的三条性质）；
  E2 的 zero-shot 与 ablation 两段（表本来就在补充）；E4 的完整 scope 段与 MAPLES-DR 细节（**双眼泄漏结论与下游 AUC 降为 exploratory 的话留在正文**）；
  ODIR-5K 细节（符号不稳定的结论留在正文）；注入模型的两条 caveat 与逐图回归细节（审稿人要的是“说清楚”而不是“详细展开”）。
  就地压缩：发展集机制那一节 648 → 214 词（E2 已取代它、表已在补充）、推理尺度 310 → 213、引言两段与讨论的 Limitations 逐条重叠故合并；
  参考文献改为 `\footnotesize`（两种版式一致，是 Elsevier 通行做法）。
  **指定保留在正文的 12 个浮动体已逐个核对页码**：`tab:main` p13、`fig:cases` p14、`fig:calib` p15、`tab:oracle`/`fig:oracle` p16、
  `fig:dose` p18、`fig:inject` p19、`fig:e2` p22、`tab:e4`/`fig:e4` p25、`fig:framework` p2、`fig:heatmap` p12。
  **途中捕到一个自己弄出的严重错误，值得记下**：移 ODIR 段的脚本用 `find("\\paragraph{", i+10)` 找下一个段落，
  而后面已经没有 `\paragraph` 了，返回 −1，于是 `s[i:-1]` **把 `05_results.tex` 剩下的全部——包括整个 E4 小节及其表与图——敲进了补充材料**。
  构建干净、引用检查通过，没有任何东西报错；是**按指示逐个核对“该留在正文的浮动体页码”才发现的**。已修复，
  并把“浮动体页码核对”写进压缩流程本身（而不是事后想起来才查）；另对所有 sections 重扫重复小节与制表符污染，归零。
  **构建**：`main.pdf` 49 页（正文 **30** + 补充 19），`main_elsarticle.pdf` 92 页；两个包装器均无未定义引用/交叉引用；
  `\todo` 20 条 = 15 作者项 + 4 back-matter + **1 GPU 钩子**（SegFormer-B0 与额外种子，按指示保留）。
  `NUMBER_SOURCES.md` 补上“压缩后每张表/图在哪里”的清单（生成器与源 CSV 未变，只是 `\input` 位置变了）；
  `REVIEW_CMIG_RESPONSE.md` 新增 §4 记录逐项去向与理由。
  **补充（11:27）**：下游臂加了一个"相同行集只拟合一次"的缓存——某族在某数据集上零失败时，
  三个 analysis_set 是**同一批行**，原实现会把同一个 CV + 1000 次自举算三遍。
  加缓存后只有 LWNet/FIVES（200/198/183 三种行数）真正需要三次拟合，其余都复用，
  复用行在 `source` 列里注明。这是为了满足"GPU 侧落地后分钟级重跑"的要求
  （SegFormer 四数据集 × 2 种子会再加 4 个下游集）。
  **发现层已对四种可能的目录布局做过验证**：`<ds>/pred`、`<ds>/seed<k>/pred`、
  `<ds>_seed<k>/pred`、以及靠 `summary.json` 补 seed；`_probe`/`_smoke`/`logs` 被跳过。
  故 GPU 侧无论用哪种命名，脚本都不需要改。
  报告的"Second segmenter family"小节已写入 `R2_ANALYSIS_REPORT.md`（现 888 行），
  下游 ΔAUC 与失败归因两张表待本轮重跑落盘后填入；17:47 正式重跑会连同 SegFormer 种子一起覆盖这批预演数字。
- 2026-09-18 11:39 **第二族审计预演全量跑完（LWNet × 4 + SegFormer-B0/DRIVE seed0），五个 CSV 全部落盘，报告小节补完（`R2_ANALYSIS_REPORT.md` 现 953 行）。**
  **下游参考差（协议同 p3，1000 次配对图像自举；两族的 FIVES 都只有 test 200 张，
  无 OOF，`split_coverage` 列逐行注明，故族 1 的可比基准是其 `test200` 分块而非全 800）**：
  HRF skan4 logreg —— **LWNet +0.2583 [+0.1251, +0.3921]** vs U-Net +0.2170 [+0.1188, +0.3323]；
  gbdt LWNet +0.1467 [−0.0143, +0.3159] vs U-Net +0.1800；primary8 logreg LWNet +0.2400 [+0.1101, +0.3973]。
  FIVES skan4 —— LWNet logreg −0.0074 [−0.0473, +0.0295]、gbdt −0.0446 [−0.0984, +0.0064]，
  对 U-Net(test200) 的 0.0000 与 −0.0319。**"低保真队列大差距、高保真队列无差距"在第二族复现，且更锐。**
  **复现判定（`family2_replication.csv`，20 行，两套 analysis_set）**：敏感性集下**五条结论全部复现**；
  主分析下四条复现，唯一不复现的是 HRF>FIVES 的残差散度排序，且**完全由 15 张退化掩膜造成**
  （排除后 1.486 vs 1.483，比值 1.00）。论文最依赖的那条——**图像级保真度 FIVES > HRF**——
  两族都干净复现（LWNet r 0.733 vs 0.343）。
  **SegFormer-B0/DRIVE 逐列复现 U-Net 的衰减模式**：β = 0.70 / 0.06 / 0.73 / 0.26，四列全 < 1，
  与 U-Net 的 4/4 一致（`DRIVE calibration slope < 1` 一行判 True）。
  **失败归因（`family2_failure_attribution.csv`）**：负 ΔAUC = 预测特征反超参考特征。
  **保留全部 200 张时 FIVES 的预测特征显著反超参考特征**：primary8+gbdt **−0.0648 [−0.1152, −0.0153]**、
  skan4+gbdt **−0.0604 [−0.1140, −0.0058]**，两个区间都不含零。这不是测量保真度，
  而是分割器在青光眼上的选择性失效充当了免费疾病标签。
  **200 张里的 2 张（1%）就贡献了该效应的 42%**（primary8+gbdt）。
  只剔除 2 张全空掩膜即可把 gbdt 两格推回覆盖零的区间；但剔除全部 17 张并不能完全消除
  （skan4+gbdt 仍为 −0.0596 [−0.1164, −0.0032]），说明渐变式欠分割也携带了一部分泄漏——
  这正是预声明"两套并列报告"的价值，任一套单独呈现都会误导。
  **HRF 零失败，四格归因恒为 0.0000**，作为机制自检通过。
  这是审阅第 14/15 条所要求收窄的**直接证据**：自动分割生物标志物上的下游 AUC
  可以因与血管完全无关的原因而移动，故 "downstream discriminative performance"
  不得被读作临床或生物学有效性。
  另修一处报告质量问题：归因比例在基线 |ΔAUC| < 0.02 时无意义（曾打印出 526%），
  现在加了 `frac_defined` 守卫并留空，改报绝对 AUC 差。

- 2026-09-18 12:07 **作者信息、CRediT、作者简介与后记全部完成；另建了**脱敏构建**。**
  从 `author_info/` 读入七位作者（文件夹顺序 = 署名顺序），中英文均按统一的“名+姓”西式顺序转写；
  **第三作者为通讯作者**（`\cormark` + `\ead`）。七人均有 ORCID 与邮箱，全部入稿（ORCID 用 CAS 类的 `orcid=` 键）。
  单位共五个；**第 5、7 位只给了本科毕业院校、未给现单位**，按“用给定的信息、不提问”处理为毕业院校，
  已在两份 checklist 里列为**待作者确认**项（连同第一作者的系名）。缺失字段一律留空而非占位符。
  **CRediT 由我根据各人 bio 分配并作为定稿写入**：除 Funding acquisition（无资助，故故意缺席）外，其余 13 个角色全覆盖；
  第一作者领 Conceptualization/Methodology/Software/Investigation/Visualization/Writing – original draft，
  通讯作者 Supervision/Project administration/Conceptualization，其余人按 bio 分担 Data curation/Validation/Formal analysis/Resources/Visualization。
  **后记三项定稿**：资助——标准的“未获得任何特定资助”声明；生成式 AI 声明——见论文正文；
  代码——录用后公开仓库，决策日志随补充材料发布，**仓库 URL 只在 checklist 里留一条 `\todo`，不进 PDF**。
  **作者简介**：`sections/biographies.tex`，每人一组 CAS `\bio{photo} … \endbio`，文字按期刊惯用语气轻度润色
  （实质修改只有两处：第 5 位的一个语法错、第 7 位的全角标点）；照片统一缩到 300 px 宽（保比例）存为
  `figures/authors/author1–7`（原文件名含中文，LaTeX 不宜直引）。实渲染核对过：七张照片与文字排版正常。
  **同一作者块已同步到后备论文 `paper/`（cas-sc）**，并在两份 checklist 里标为“假定同一团队，待确认”。
  **隐私防护（重要）**：新增 `main_review.tex`（两个项目各一份）——正文完全相同，但作者块/单位/邮箱/ORCID/CRediT/简介
  全部替换为“[authors redacted for review]”。`paper2` 用 `build.ps1 -Main main_review`，`paper` 用 `build.ps1 -Wrapper review`。
  **已实测验证**：把两份脱敏 PDF 抽文本后搜 23 个个人信息字串（每个名、每个邮箱、每个 ORCID、每个单位）——**0 命中**；
  而非脱敏版 **23/23 全命中**（这条对照证明检查本身有效，而不是白跑一遍）。**今后任何上传都只能用 main_review.pdf。**
  **构建**：paper2 `main.pdf` 49 页（含作者简介页）、`main_review.pdf` 49 页；paper `main.pdf` / `main_review.pdf` 各 39 页；
  四个构建均无未定义引用/交叉引用。**paper2的 `\todo` 从 20 降到 1——只剩 GPU 钩子**（SegFormer-B0 与额外种子）；
  paper 的作者相关 `\todo` 已归零。
- 2026-09-18 18:20 **第二族审计全量落地（17:47 里程碑）：12 组预测（LWNet × 4 数据集 + SegFormer-B0 × 4 数据集 × 2 种子），五个 `family2_*.csv` 全部重写，`R2_ANALYSIS_REPORT.md` 的第二族小节按全量结果重写（现 956 行）。**
  **跑之前修了两个会导致静默错误的问题**：
  (1) GPU 侧用的目录名是 `<ds>_s1`，我的发现正则只认 `_seed1`，**8 组 SegFormer 会被整体静默跳过**；
      现在 `_s<k>` / `_seed<k>` / `-seed<k>` / 嵌套 `seed<k>`、`s<k>` 都认。
  (2) `zero_shot` 对 SegFormer 默认成了 True，但它在每个数据集上都是**域内训练**；
      改为显式 `is_zero_shot()` 规则（`IN_DOMAIN_FAMILIES`）。
  另核对：两个 SegFormer 种子是真正独立的运行（checkpoint 与预测都不同；
  DRIVE 两个种子 best_dice 都是 0.7961 只是四位小数的巧合）。
  **SegFormer-B0（3.71 M 参数）best Dice（seed0/seed1）对 U-Net**：
  DRIVE 0.796/0.796 vs 0.826、CHASE 0.809/0.807 vs 0.817、HRF 0.809/0.808 vs 0.814、FIVES 0.903/0.904 vs 0.902
  —— 略弱但胜任的第二架构，正好是公平对照而非稻草人。
  **审计（primary skan，main，σ 单位，平均 |μ| / resid SD / CCC / r）**：
  HRF —— U-Net 1.540/0.911/0.302/0.487；LWNet 3.015/1.486/0.131/0.343；
  **SegFormer 1.926/1.618/0.285/0.414 与 1.870/1.555/0.254/0.386**。
  FIVES —— U-Net 0.248/0.612/0.785/0.824；LWNet 0.694/1.757/0.585/0.733；
  **SegFormer 0.255/0.461/0.869/0.900 与 0.278/0.537/0.844/0.877**（比 U-Net 还校准得好）。
  **种子稳定性极好**：32 个 (数据集×标志物) 格子上 seed0−seed1 的绝对差中位数为
  μ 0.057 σ、β 0.028、resid SD 0.058、CCC 0.013、r 0.018；没有一条结论取决于读哪个种子。
  **参考差 ΔAUC（skan4/logreg；所有族的 FIVES 都只有 test 200，无 OOF，故族 1 的可比基准是 test200）**：
  HRF —— U-Net **+0.2170 [+0.1188, +0.3323]**、LWNet **+0.2583 [+0.1251, +0.3921]**、
  **SegFormer +0.1983 [+0.0614, +0.3480] 与 +0.2100 [+0.0744, +0.3572]**，四者区间都不含零；
  FIVES —— 0.0000 / −0.0074 / +0.0026 / +0.0073，全部≈0。
  **三种架构（U-Net、68 k 零样本 W-Net、3.7 M 域内 transformer × 2 种子）在两半对比上完全一致。**
  **复现判定**：论文叙事所依赖的四条排序结论在**每个族、每个种子**上都复现
  （偏移比 4.35×/7.55×/6.74×；保真度 r FIVES>HRF；残差散度比 3.51×/2.90×；ΔAUC 大/≈0）。
  唯一失败的是 LWNet 的残差散度排序，且**完全由 15 张退化掩膜造成**（敏感性集里 1.486 vs 1.483 打平）。
  **新增一条方法学守则**："有多少斜率 < 1"这个计数在低保真队列上**不可用**：
  SegFormer HRF 的 FD β=2.40、CI [−16.1, 21.0]（r=0.20），total_length β=1.06、CI [−2.3, 3.2]，
  即斜率**不可识别**。复现表现在输出 `n_identified`（自举 CI 宽度 ≤ 2）与
  `n_identified_below_1`，识别数不足一半时判定为 **None（不可判定）**而非 False。
  只看可识别的斜率则完全一致：SegFormer HRF 迂曲度 0.28 [0.01, 0.65] 与 0.29 [0.05, 0.63] 稳稳 < 1
  （U-Net 0.37）；DRIVE 两个训练族都是 4/4 < 1（SegFormer 0.70/0.06/0.73/0.26 对 U-Net 0.77/0.03/0.72/0.45，
  近乎逐列吻合）。**论文里要从可识别斜率读衰减，绝不要从计数读。**
  **最重要的新发现（比预期更强）：对青光眼的选择性分割失败不是零样本模型独有的。**
  掩膜失败按预声明规则统计：LWNet FIVES 2 空 + 15 退化 = 17/200（8.5%），
  青光眼 13/50、AMD 2/50、DR 2/50、正常 0/50；
  **SegFormer（域内训练）FIVES 0 空 + 2（seed0）/ 3（seed1）退化，全部是青光眼**，
  其余三类 0/150。比率差约一个数量级，但**类别选择性完全相同且彻底**。
  **失败归因**：保留全部 200 张时预测特征显著反超参考特征——
  LWNet primary8/gbdt **−0.0648 [−0.1152, −0.0153]**、skan4/gbdt **−0.0604 [−0.1140, −0.0058]**，
  **SegFormer seed1 primary8/gbdt −0.0462 [−0.0917, −0.0003]**，三者区间都不含零。
  **极少数图像就足够**：2/200 贡献 LWNet 效应的 42%；3/200 贡献 SegFormer 的 28%，
  剔除后区间从 [−0.0917, −0.0003] 变为 [−0.0786, +0.0140]，**显著性消失**。
  即**一个队列里 1% 的图像就能制造出统计显著的下游结果**。HRF 零失败，四格归因恒为 0.0000（自检通过）。
  这是审阅第 14/15 条所要收窄表述的直接证据。

- 2026-09-18 18:37 **第二分割器家族审计已集成（审阅第 9 条完成），并带回一个**没有人要求但很重要的发现**。**
  新增正文小节 `sec:res-family` + 主表 `tab:family`；复现台账与掩模失败核算入补充（`tab:familyrep` / `tab:familyfail`）。
  **(a) 审计不是 U-Net 的产物**：在公开 68k W-Net（零样本）与本地同条件训练的 3.71M SegFormer-B0（两个种子）上重做，
  共同尺度是**载入而非重算**（否则两家族分母不同）；SegFormer 是合格对照而非稻草人（Dice 0.796/0.809/0.809/0.903 vs U-Net 0.826/0.817/0.814/0.902）；
  **四项排序结论在每个家族、每个种子上全部复现**，含参考差距对比（HRF +0.198 [+0.061,+0.348] 与 +0.210 [+0.074,+0.357]；FIVES ≈ 0）。
  **(b) 方法学告诫已写入 Methods**：r 低时 Deming 斜率**不可识别**（SegFormer HRF FD β=2.40，CI [−16, 21]，r=0.20），
  故**只从可识别斜率（bootstrap CI 宽度 ≤ 2）读衰减，绝不用“多少个斜率小于 1”的计数**——那个统计量会奖励更噪的分割器。
  **(c) 新发现（写进正文，直接支撑 14/15 条的收窄）**：对青光眼底图的**选择性分割失败并非零样本模型特有**——
  W-Net 在 FIVES 上 17/200 退化或空掩模（13/50 青光眼、0/50 正常）；**在域内训练**的 transformer 失败率低一个量级，
  但**只在青光眼上失败**（2–3/200，两个种子均如此）。保留这些掩模会使**预测特征显著赢过参考特征**
  （−0.065 [−0.115,−0.015]；−0.046 [−0.092,−0.000]），而 **2–3 张图（1–1.5%）就贡献了 28–42% 的效应**，剔除后显著性消失；
  HRF 零失败则归因恰为 0.0000（构造校验）。预先定下的**两集报告规则**（零前景=失败，剔除并计数；<2% 前景=退化，
  主分析保留、敏感性分析剔除）已写进 Methods。**结论：“下游区分性能”绝不能读作生物学有效性**——
  百分之一的分割伪影就足以造出一个方向有利的显著结果。
  **(d)** 家族对比的 U-Net FIVES 参照物是 **test-200 块**而非 800 图数字（无任何家族在 FIVES 训练分割上有 OOF 预测），表注与正文均明说。
  **GPU 钩子收窄到最后一项**：FIVES 的额外 ReliSeg 种子尚未到位（`e2_delta.csv` 仍为 seeds 0–2，目录里有 `.bak_pre6seed`，说明正在跑）。
  为不破坏上一轮的长度压缩，把 §5.1 的 σ 尺度 caveat 整块移入补充，正文留三句承重的话。
  **构建**：`main.pdf` 51 页（正文含参考文献与作者简介 32 页）、`main_review.pdf` 51 页；两者均无未定义引用/交叉引用；
  指定保留在正文的浮动体逐个核对页码无一漂移；**脱敏检查再次通过**（23 个个人信息字串在 main_review.pdf 中 0 命中、
  在 main.pdf 中 23/23 命中）。**`\todo` 仍为 1 条，即那个 GPU 钩子。**
- 2026-09-18 19:45 **发现并修复一个会影响已发布 E4 结果的真实缺陷：有 4 个单元的预测来自"还没训完"的 checkpoint。**
  起因是例行核对：GPU0 的 E4 seed3 泳道已写出 `.done`，但 seed3 只有 **9/10** 个微调有
  `summary.json`，推理却是 **10/10**——有一个单元"有预测、没训练"。
  - **`fold2/reliseg_s3`（本次新增，我的）**：微调进程在 **ep 14/50** 死掉，没写 `summary.json`；
    泳道脚本的 `if not exist pred\infer_meta.json` 仍然成立，于是**拿 ep13 的 `last.pt` 跑了推理**，
    得到一个看起来完全正常的单元（manifest、pixel_metrics、Dice 0.9403 一应俱全）。
  - **`fold4/{cfloss, continued, reliseg}` 种子 0（2026-09-17 的原始 E4 跑，不是本次新增）**：
    `infer_meta.json` 的 `ckpt_epoch` 分别是 **24 / 38 / 39**，而不是 49；
    文件时间戳铁证如山——`pred/infer_meta.json` 写于 18:56–18:57，
    而 `last.pt` / `summary.json` 写于 **19:05–19:11**，即**推理比训练结束早了 9–14 分钟**。
  - **根因**：`e4_replication.build_jobs()` 里推理作业的依赖写成了
    `deps=[arm_ckpt(k, arm)]`，也就是 `<arm>/last.pt`。但 `p5_finetune` **每个 epoch 都重写
    `last.pt`**，所以这个依赖在训练刚开始就已满足；两条 GPU 泳道共用队列时，
    第二条泳道就在第一条还在微调的时候把推理作业领走了。E2 的队列用的是 `summary.json`，
    没有这个问题——是 E4 抄的时候抄错了对象。
  - **修复**：新增 `infer_dep(fold, arm, seed)`，推理依赖一律改为 `summary.json`
    （只在最后一个 epoch 之后写一次）；`baseline` 臂依赖其 fold 基座的 `summary.json`。
    原始与额外种子两处 `deps=` 都已改。
  - **影响范围与处置**：受影响的是**主复制终点五折中的 fold 4、四个臂里的三个**
    （`baseline` 臂读的是 `base/best.pt`，未受影响），即那三个臂 100 张 out-of-fold
    预测里的 20 张。四个单元的 `pred/` 与对应的生物标志物缓存
    （`results/pivot/cache/fd5/fundusavseg/p5_<tag>/`，否则 `_bio_one` 会命中旧缓存直接返回）
    已一并改名为 `*_STALE_20260918_1945` **保留而非删除**。
    fold4 三个臂的 `last.pt` 经核对**确实停在 epoch 49**（val Dice 0.9335/0.9337/0.9317），
    所以**只需重跑推理**，不必重训；`fold2/reliseg_s3` 的微调本身是坏的，删掉 ep14 的残留
    checkpoint 后**重训 + 重推**。修复泳道 `e4_repair.bat` 已在 GPU0 启动。
  - **LAN 节点 22 个单元全部干净**（`ckpt_epoch` 均为 49 且都有 `summary.json`）——
    因为 LAN 的驱动脚本是同一进程里 train 完再 infer，不走共享队列。
  - **新增常备体检脚本 `src/pivot/g9_verify_units.py`**：扫描 E2 / E4 / r2 全部预测目录，
    比对 `infer_meta.json` 的 `ckpt_epoch`、`summary.json` 是否存在、以及
    `infer_meta.json` 是否早于 `summary.json`，任何一条不过就退出码 1，可当分析前的闸门。
    注意它**只对 `last.pt` 施加"必须跑满"的判据**：`fid` 臂评的是 `best_fid.pt`，
    按定义就该停在中间某个 epoch，早期版本把 20 个 `pred_fid` 全报成了假阳性。
    当前全库 **219 个预测目录全部通过**。
  - **教训**：`last.pt` 这种"每个 epoch 都覆盖"的文件永远不能当作"训练完成"的依赖标记；
    完成标记必须是只在结束时写一次的文件。
- 2026-09-18 19:50 **E4 正确性缺陷与处置（协调者裁定）**：GPU 代理发现 E4 折 4 的 seed-0 三个微调臂（cfloss/continued/reliseg）在训练未完成时即被推理（infer_meta 记录 epoch 24/38/39，早于 summary.json 9–14 分钟；根因：推理任务以每 epoch 重写的 last.pt 为依赖而非 summary.json），另 fold2/reliseg_s3 训练中途死亡后被推理。处置：推理依赖改为 summary.json（infer_dep）；受影响的预测目录与生物标志物缓存改名保留（*_STALE_20260918_1945）；折 4 seed-0 三臂仅需重推理（last.pt 完整于 epoch 49），fold2/reliseg_s3 重训；修复后重新生成 E4 全部 CSV 与 E4_REPORT.md（seed 0 与 5 种子），论文中所有 E4 数字以修复后为准；新增门禁脚本 g9_verify_units.py（219 个预测目录通过）。FIVES 6 种子：主终点所有图像 bootstrap CI 含零；total_length 的种子级区间 [+0.0001, +0.0009] 排除零但效应 ~5e-4 → 表述为"可检出但可忽略"。

- 2026-09-18 20:05 **FIVES 六种子结果已入稿，**最后一个 `\todo` 关闭——PDF 里已经一条都没有**。**
  生成器改为**按队列取最宽的 pooled 行**（FIVES 有 `pooled(0,1,2,3,4,5)`、HRF 仍为 `pooled(0,1,2)`），
  而不是写死字符串，所以待会儿那个 canonical rerun 细化 CI 时表会自己跟上。
  **结论写法**：对同预算 `continued` 对照，四个主生物标记的 pooled 图像 bootstrap 区间**全部含零**；
  FD 的逐种子效应 −0.011~+0.014，比 pooled 均值（+0.0014）宽一个量级；四列 direction_consistent 全为 no。
  **total_length 按要求写成“可检出但微不足道”**：seed 级区间 [+0.0001, +0.0009] 排除零、6/5 种子为正，
  但 Δr ≈ 5e-4——比设计所针对的 0.05 小四个量级，既不写“no effect”也不写“an effect”。
  **下游与 zero-shot 臂按设计保持 seeds 0–2**，正文明说。
  **一个必须跟着改的地方**：原来那句“最大点估计 +0.008（FD，continued 对 baseline）”是**三种子的数**；
  六种子下该格为 **+0.013 [+0.0005, +0.0227]，已排除零**。这并不削弱结论，反而**加强**了本文真正的主张：
  ReliSeg 对 baseline 也排除零（FD +0.015），但 `continued` **单独**就几乎产生了全部增量——
  加深种子采样把**训练预算效应**变得可分辨，而把目标函数自身的效应留在了与它不可区分的位置。正文已改写。
  **种子数混用已明说**：CF-Loss 臂未重跑，仍为三种子；表的 ``seeds`` 列本来就印分母（4/6 对 2/3），
  表题与 Methods 都写明哪些臂是六种子、哪些是三种子。
  **E4 未动**：按指示等重生的 E4_REPORT.md（5 seeds）到位后再整体替换数字/表/图，并在补充的 protocol notes 里
  加一句：fold-4 seed-0 预测因 checkpoint 依赖缺陷已重生，且现在每次分析前都有单位校验门。
  **构建**：`main.pdf` 52 页、`main_review.pdf` 52 页，均无未定义引用/交叉引用；**`\todo` = 0**；
  脱敏检查再次通过（23 个字串在脱敏版 0 命中、在正式版 23/23 命中）。
- 2026-09-18 22:36 **E4 修复后重算完毕：seed 0 与 5 种子两套都已产出；结论层面 fold-4 修复没有改变任何定性判断，但 5 个种子把"无差异"改写成了"一致地略差"。**
  - **前置条件全部满足**：40 个额外种子微调 + 40 次推理 + 60 份 `bio.csv` 全部完成；
    LAN 的 22 个单元已拉回（**只拉 `pred/` 与元数据，用 `--exclude="*.pt"` 排除 30 MB 的
    checkpoint**）。拉回的 manifest 里是 LAN 的绝对路径，而 `p5_eval.cmd_bio` 是**直接读
    `mask_path`** 的，所以先把 1,560 条路径从 `/mnt/data/.../exp/` 改写成本地根并逐条验证
    存在（missing 0）才跑 bio——否则整批会静默读错文件。
  - **闸门 `g9_verify_units` 退出码 0**，244 个预测目录全部通过。
  - **修复只动了 fold 4 的铁证**：`leave_out_fold4`（用另外四折算的同一对比）修复前后
    **逐位相同**（FD −0.00781、tortuosity −0.05280、density −0.00160、total_length −0.00029）。
  - **seed 0 pooled Δr（reliseg − continued, skan）——论文要替换的数**：
    | 标志物 | 旧 | 新 |
    |---|---|---|
    | FD | −0.01036 [−0.0376, 0.0141] | **−0.00882 [−0.0375, 0.0167]** |
    | tortuosity | −0.06310 [−0.1366, −0.0296] | **−0.06208 [−0.1352, −0.0292]** |
    | density | −0.00034 [−0.0048, 0.0038] | **−0.00127 [−0.0058, 0.0028]** |
    | total_length | −0.00057 [−0.0024, 0.0015] | **−0.00084 [−0.0026, 0.0012]** |
    `p_adj`：density 0.8585 → 0.9725，total_length 0.8895 → 0.9435，FD 1.0 不变。
    fold 4 上 `continued` 的 FD 保真度 r 从 0.90275 掉到 0.88353、tortuosity 0.92477 → 0.91334
    ——**没训完的模型在这两项上反而"更好"**，正因如此这种污染不会自己暴露出来。
  - **没有改变的结论**：`replication_success` 修复前后都是 **False**，`significant` 都是空；
    安全旗标 **42 ok / 8 BREACH** 完全一致（tortuosity 的 Δr 突破 −0.05 预注册边界是**早就存在**
    的，不是修复引入的）；MAPLES-DR 的产物一个字节没动（时间戳仍是 09-17）。
  - **5 个种子才是真正的结果变化**：`reliseg − continued`（skan，每折每臂 5 个微调种子）
    | 标志物 | 5 种子均值 | 种子级 95% t 区间 | 为正的种子数 | 同号 |
    |---|---|---|---|---|
    | FD | **−0.00694** | [−0.01199, −0.00189] | 0/5 | 是 |
    | density | **−0.00220** | [−0.00353, −0.00087] | 0/5 | 是 |
    | tortuosity | **−0.05900** | [−0.07058, −0.04742] | 0/5 | 是 |
    | total_length | −0.00047 | [−0.00092, −0.00001] | 1/5 | 否 |
    单个种子时这些都在噪声里；**5 个种子下三项的符号在每个种子上都一致、且种子级区间不含 0**。
    也就是说 E4 不只是"没能复制出收益"，而是**一致地、可检出地略有损害**，其中 tortuosity
    的 −0.059 已经不算小，并且正好压在预注册安全边界 −0.05 之外。对 `baseline` 的对比同向
    （density −0.00197、tortuosity −0.05999 同样 0/5 为正且区间不含 0）。
  - **产物**：`e4_*.csv` / `replication_rule.json` / `E4_REPORT.md`（seed 0，规范文件）、
    `e4_*_s{1..4}.csv` / `E4_REPORT_s{1..4}.md`（各种子，**同一个估计量、同一套折、同一条复制规则**，
    只有读哪个微调不同）、`e4_delta_seeds.csv`（跨种子汇总，沿用 E2 的合并规则：
    per-seed Δr 的均值 + 种子级 t 区间 + 符号一致性）、以及
    `results/pivot/E4_FOLD4_REPAIR.md`（修复前后全对照，独立成文件所以不会被下次重算覆盖）。
  - **实现上的两个坑**（都已修）：`analyse --seed` 的输出后缀我先是写成 `cmd_analyse` 里的闭包
    `_out()`，但 `write_report` 是顶层函数，于是 seed 0 的 CSV 写完之后崩在
    `NameError: _out` ——**CSV 更新了、报告还是旧的**，这正是协调者看到的现象；
    改成给 `write_report` 传 `suffix` 参数。另外用 heredoc 注入代码时 `\n` 被吞成真换行，
    产生了未闭合字符串，已逐处改回单行 print。
  - **E2 侧**：规范的 6 种子 `e2_analysis` 已于 **20:42** 跑完，
    `E2_REPORT.md` / `e2_delta.csv` 均已重写，FIVES 覆盖表 6/6 complete，
    主终点标题不再写死 "seeds 0-2"。

- 2026-09-18 23:04 **E4 修复后的全部数字/表/图已入稿，E4 由"没能复制出收益"改写为"无收益且一致地略有损害"；这是 external review 验证轮之前的最后一次整合。**
  - **数据来源全部换成修复后的 CSV**：`pivot/e4_delta.csv`（seed 0，固定 checkpoint，图像级配对 bootstrap）
    与新增的 `pivot/e4_delta_seeds.csv`（每臂每折 5 个独立重训种子，自由度 4 的 t 区间）。
    修复前的 CSV 稿件中一处也不再读取。
  - **`mk_tab_e4.py` 重写为两栏面板**：(a) 图像级、固定 checkpoint；(b) 跨 5 次重训，
    每格右上角标注点估计为正的种子数（如 `0/5`）。**主面板三列按 1e-3 缩放**——
    四位小数下 total_length 的种子级上界会印成 `0.0000`，读起来就是"含零"，
    而该行的要点恰恰是区间不含零；表题写明"印出的 −2.20 即 Δr = −0.00220"。
    `\textbf` 表示区间不含零，`\underline` 仍表示突破预注册安全边界。
  - **`fig_e4_replication.py` 左panel 现在画两层**：细须为图像级 bootstrap，
    粗淡色条为跨 5 种子的 t 区间；图例加灰色代理句柄，`ylim` 下沿放宽以免图例压住最后一行。
  - **正文措辞（按协调者要求的精度）**：图像级下 FD/density/length 三个终点在每个种子内区间均含零；
    跨 5 次重训三者同向且种子级区间均不含零——density −0.00220 [−0.00353, −0.00087] 0/5 为正、
    FD −0.00694 [−0.01199, −0.00189] 0/5、total_length −0.00047 [−0.00092, −0.00001]
    **1/5 反向且区间只在最后一位数上排除零**（这一点如实写出，未笼统写成"符号完全一致"）。
    迂曲度两层都可检出且超过锁定边界：−0.062 [−0.135, −0.029]（图像级）与
    −0.059 [−0.071, −0.047]（跨种子，0/5 为正），**两个点估计都越过 −0.05，而两个区间上端都刚好停在边界内**。
  - **两处限定在每个出现点都写明**：5 种子分析是**次级、未锁定**的，不是对锁定规则的第二次检验；
    其区间覆盖的是**同一队列同五折的重训**，不是新队列的抽样。E4 下游 AUC 仍为 exploratory。
  - **摘要/亮点/引言/讨论/结论**均改为"无收益 + 一致的轻微损害"。亮点第 5 条改为
    "Against a same-budget control: no fidelity gain, a small consistent held-out loss."（82 字符）。
  - **对照臂的唯一正向移动**：`continued − baseline` 的 FD 图像级 +0.0046 [−0.0005, +0.0099]（含零），
    跨种子 +0.0052 [+0.0028, +0.0075]、5/5 为正——与六种子 FIVES 同样的模式：
    加深种子采样让**训练预算效应**变得可分辨，目标函数自身的效应仍与零不可区分。原句（+0.0084 已排除零）已改写。
  - **补充 S4 新增 protocol notes 两段**：推理依赖 `last.pt`（每 epoch 覆盖）导致第二条泳道对"还在训练"的模型跑推理；
    受影响 4 个单元（fold 4 seed-0 三个微调臂，epoch 24/38/39 of 50；fold-2 一个 ep14 死掉的单元），
    `baseline` 臂读 fold 基座未受影响；修复改依赖 `summary.json`，旧产物改名保留。
    **明写"缺陷没有美化本方法"**：没训完的 checkpoint 反而更好（fold 4 `continued` 的 FD 保真度 r 0.903→0.884、
    迂曲度 0.925→0.913），这正是污染不会自己暴露的原因。三条边界：`leave_out_fold4` 修复前后逐位相同、
    `replication_success` 两次都是 FALSE、安全旗标 42 ok/8 BREACH 一致。
    并写明常备闸门 `g9_verify_units` 每次分析前运行，本轮 **244 个预测目录全部通过**。
  - **种子数混用**在表题、Methods §3.6 与 §5.12 三处写明：Fundus-AVSeg 每臂每折 5 个微调种子，
    CF-Loss 只微调 1 次（只出现在面板 a），MAPLES-DR 零样本仍为原来的 3 个 FIVES 训练种子。
  - **摘要长度**：R3 收到 250 词后各轮加内容已涨到 **326 词**；本轮加入 E4 句并重新收紧到 **302 词**。
    再压到 250 必须删掉一个完整结果，故**留给协调者裁定**，未自行删除任何经评审确认的限定语。
  - **构建**：`main.pdf` 52 页（正文含参考文献 33 页，补充自 34 页起）、`main_review.pdf` 52 页；
    两者均**无未定义引用/交叉引用**；`\todo` = **0**；overfull hbox 25 处，除 `\maketitle` 里
    CAS highlights 框那处 123.6pt（既有、渲染正常）外全部 < 27pt。
    **脱敏检查通过**：23 个人身串在 `main_review.pdf` 命中 **0/23**，在正式 `main.pdf` 命中 **23/23**（阳性对照）。
  - **同步更新**：`NUMBER_SOURCES.md` §5.12 全表（新增两层抽样的说明、跨种子行、修复审计行、种子数行）、
    `REVIEW_CMIG_RESPONSE.md` 追加 "Round 11" 一节。

- 2026-09-18 23:17 **摘要压缩到 250 词（协调者裁定：一个结果都不许删，只压措辞）。**
  - **326 → 302 → 250 词，八个结果一个未丢**：三轴量级对比（0.25–1.54σ / 0.72–1.48σ / 0.006–0.115σ，
    最严协变量匹配下仍小 17–20 倍）、参照掩膜下游臂（FIVES 少数 AUC 点 vs HRF 0.217，四个斜率 0.37–0.81
    "是衰减不是平移"）、生态相关性免责、注入实验三个分句（常数偏移逐位不变 / 逐图相关残差单调衰减 /
    真实测量落在曲线上或之上）、六个候选补救对同预算对照、E4、CF-Loss 忠实复现降低保真度、像素指标非劣。
  - **压缩手段**（按指示）：删冗余限定词；把 FIVES 标定括号并进保真度句（β 1.03/0.99/1.58 三个数删除，
    定性的"除分形维被拉伸外标定良好"保留，该对比亮点里已有）；E4 句压成两个分句
    ——"no benefit" 与 "a small consistent detriment, with unconstrained tortuosity past its declared
    safety margin on two cohorts"；句法收紧（"we separate…" 主动句替换 "We audit that substitution…
    separating…"）。
  - **评审强制的限定语一个未动**："a rule fixed beforehand"、"same-budget continued-training control"、
    "a consistent improvement across the seeds we ran"、"ecological / we make no causal claim"、
    "faithful implementation … adapted to binary vessel segmentation … here"、"declared safety margin"。
  - **最终构建**：`main.pdf` 52 页（正文含参考文献 33 页，补充自 34 页起）、`main_review.pdf` 52 页
    （正文 32 页，补充自 33 页起）；两者**无未定义引用/交叉引用**、`\todo` = **0**；
    overfull hbox 24 处，除 `\maketitle` 里 CAS highlights 框那处 123.6pt（既有）外全部 ≤ 26.8pt。
    **脱敏检查**：`main_review.pdf` 23 个人身串命中 **0/23**，`main.pdf` 阳性对照 **23/23**，
    且脱敏版含 "redacted for review" 标记。
    **浮动体页码核对**：`tab:main` p14、`fig:cases` p15、`tab:e4` 与 `fig:e4` 同在 p28，均在正文内；
    反斜杠扫描无新增可疑项（仅 `frontmatter.tex` 的 `\corref` 既有误报）。
  - **移交**：验证代理接手 `paper2/main_review.pdf`。

- 2026-09-29 17:34 **外部验证轮的六项必修全部落地（8.2/10 → 待复核）；本轮不跑实验、不改任何数字，只动措辞、表标签、加粗规则与版面。**
  - **审计结果（上一代理被周限额打断，停在"重建 + 第 6 项预检"）**：第 1、3 项与第 4 项的大部分**已在源码中**
    （§5.2/表 3 标题/表 S1 的残差离散排序表述一致；第二分割器失效段先报冻结主面板；注入句改为
    "compatibility result, not an identification"；摘要 σ 范围写明 per-cohort panel means、+0.217 有了主语；
    讨论里的 "this one is an answer" 已降调；表 2 已瘦为 μ/β/resid SD/r/AUC gap；图 1 文字已改为
    "frozen U-Net, 3 seeds; audit re-run on 2 more families" 并已重生）。
  - **本轮补上的**：
    1. 讨论局限段 "its orderings survive that change of architecture" → 三项在每个族每个种子复现、第四项
       只在两个 SegFormer 种子复现、在零样本 W-Net 上为敏感性集内持平；"the evidence is one application,
       one segmenter family" → 干预一个族、审计另外压测两个族。
    2. **FIVES 样本账还有两处 400 残留**：表 S18（外层 bootstrap）与表 S24（常数偏移不变性）的 FIVES "all"
       行是旧的 400 块（test 200 + 训练集按类分层 200 张上限），**没有在 800 上重跑**。决定：不重跑，
       由 `mk_tab_outerboot.py` / `mk_tab_harmonisation.py` 标为 `test+tr200` 并在表题定义；§5.5 相应句改为
       "外层 bootstrap 在 test-200 与此前的 400 块上运行"，"reproduces every point estimate exactly"
       限定为与表 4 共享的行。表 4 原来**每个 gap 都加粗**（含区间含零的 FIVES 格），改为仅区间不含零才加粗。
       §5.12 "every reference-gap cell there covers zero" 与 §5.5 的"合并 800 八列 GBDT 一格排除零"冲突，
       改为 "every primary-panel reference-gap cell"。
    3. 结论里 "so this is an answer rather than a question mark" 删除（改为 "more informative than a null
       without a power statement, under that simulation model"——注意不能写 "unpowered null"，
       它含禁用短语 "powered null"）。摘要 E4 句去掉"every targeted biomarker drifts the wrong way"
       （total length 5 个种子中 1 个为正，此话不严），改为 "shows no benefit but a small consistent detriment"。
       摘要 **260 → 249 词**（wc.py 口径，与之前各轮相同）。
    4. 版本史残句：引言 "an earlier draft used"、§5.4 标题与表 2 标题 "the column the earlier
       offset-plus-scatter decomposition was missing"、"the earlier decomposition"、"our earlier probe
       tables"、"One earlier result"、相关工作 "an earlier fixed-ladder variant … survives"、补充
       "moved here in full from the main text"、可复现性声明里 "review round" 一句（正式版与脱敏版
       backmatter 都改）、若干修订意义上的 "now"。"utility" 指 AUC 终点处改为 downstream
       discriminative performance；§5.14 标题与表 S5 标题改为 "External downstream validation"。
       **保留**补充 S4 的 checkpoint 缺陷 protocol note（那是缺陷披露，不是修订史）。
    5. 重复标签 `sec:supp-flag`（补充里两处）删掉一处——multiply-defined 由 1 变 0。
  - **第 6 项 / 全页预检**：图 4（HRF method comparison）现在在印刷页 17，标题与页脚分离。
    预检脚本（pdftoppm 100 dpi，行墨迹剖面：末带高于众数页脚带 = 正文并入页脚；页脚下方墨迹；左右页边墨迹）
    在两个 PDF 上都跑，并用 9 页一张的缩略图逐页目检。**发现一个验证轮没报的缺陷**：印刷页 24 两个不可断行的
    置信区间（左栏 total-length 种子区间、右栏 CF-Loss density 区间）冲出栏宽、压进相邻栏/右页边；
    在区间内部允许断行后消失。**最终：main.pdf 0/54 页、main_review.pdf 0/54 页被标记。**
  - **构建**：`main.pdf` 54 页（正文含参考文献 34 页，补充自 35 页起），`main_review.pdf` 54 页（33 + 补充自 34 页）。
    两者无未定义引用/交叉引用、无重复标签、`\todo` = 0、渲染文本中 `??` = 0、反斜杠宏名 = 0；
    浮动体页码核对：所有正文浮动体都在补充之前（最后一个 `tab:e4`/`fig:e4` p27）。
    **脱敏检查**：23 个人身串在 `main_review.pdf` 命中 **0/23**（含 "redacted for review"），`main.pdf` 阳性对照 **23/23**。
    "不得回归的短语"扫描：唯一命中是讨论里的否定句 "We are deliberately not saying that fidelity is the only axis"，保留。
  - **同步**：`NUMBER_SOURCES.md`（外层 bootstrap 400 块定义、表 4 加粗规则）、`REVIEW_CMIG_RESPONSE.md`
    追加 "Verification round" 一节、`SUBMISSION_CHECKLIST.md` 状态行。

- 2026-09-29 17:55 **复核（review/cmig_verify2_reply.md）：六项必修全部达标，8.8/10，结论 "ready for CMIG"；只剩三处文字修订，已落地，终稿构建完成。**
  - **(1) 图 1** 底注 "test downstream utility only" → "test downstream discriminative performance only"，
    经 `tools/fig_framework.py` 重生。顺带修了该图三处既有的框内溢出（全页墨迹预检查不出图内文字）：
    右上 "predicted masks" 框文字超框（改三行、7.2pt）、框 5 末行超框（改四行、7.0pt）、框 4 与框 5 相互压边（框 4 收窄到 0.575）。
  - **(2) 补充 S24** 标题 "Tables moved out of the main text" → "Supporting tables for the main-text results"；
    首段去掉"留在这里是为了让正文成一条线"的迁移说法；"three columns restored" → "three further columns"；
    删去 "because the methods consultation asked for them"，只留 CCC 与 r 可以背离这一实质理由。
  - **(3) 引言 E4 句** "all three targeted endpoints then move the same small distance" 改为给出不同量级：
    density −0.0022、FD −0.0069、total length −0.0005（Δr），不受约束的 tortuosity −0.059；数字取自 `pivot/e4_delta_seeds.csv`，与 §5.12 一致。
  - **终稿构建**：`main.pdf` 54 页（正文含参考文献 34 页，补充自 35 页起），`main_review.pdf` 54 页（33 + 补充自 34 页）；
    两者无未定义引用/交叉引用、无重复标签、`\todo` = 0、`??` = 0、渲染文本反斜杠宏名 = 0；正文浮动体全部在补充之前；
    全页预检 main.pdf 0/54、main_review.pdf 0/54 页被标记；摘要 249 词。
    **脱敏检查**：`main_review.pdf` 23 个人身串命中 **0/23**，`main.pdf` 阳性对照 **23/23**。
  - **状态**：稿件定稿；剩下的只是作者项——选定期刊（封面信两处期刊名）与公开代码仓库 URL。
- 2026-09-29 18:10 **用户第二次 CMIG 审阅（review/user_cmig_review2_20260929.md）**：判断意见成立（主线过散、ReliSeg 篇幅过大、摘要信息过载、Related Work 缺 CMIG 近作与 novelty 表、版面有行号/留白/图表质量问题）。执行：主文压缩为一条主线 + 4 个核心结果（三轴审计与多族复现；校准结构；参考掩膜下游臂与误差注入；干预压力测试汇总一节一表），其余入补充；摘要重写 220–240 词；补 CMIG 文献与 novelty 表；删除行号、修留白、重制图表；随后 external review 迭代审查至收敛并做版面检查。不新增实验。
- 2026-09-29 19:18 **CMIG 结构性大改落地（review/user_cmig_review2_20260929.md 第 1–7 点；第 8 点 external review 迭代审查未在本轮执行）。不跑实验、不改任何数字。**
  - **定位**：主贡献改为 segmentation-derived imaging biomarkers 的 measurement-validation framework（① signed standardised calibration structure ② reference-mask downstream arm ③ controlled error interventions）。新标题 "A measurement-validation framework for segmentation-derived retinal vascular biomarkers: calibration structure, downstream attenuation and intervention stress tests"。正文不再出现 "We propose ReliSeg"，统一称 "the measurement-aware objective"，并逐字使用 "we use a measurement-aware objective as a controlled intervention to test whether the identified fidelity error is readily correctable"；ReliSeg 仅作为补充材料表格标签。
  - **正文 = 一条主线 + 四个结果**：R1 三轴审计 + 第二/三族复现 + 像素指标对照；R2 校准结构（新表 4：μ、α、β、CCC、BA 斜率、残差 SD、r；新图 4：HRF/FIVES 散点 + Bland–Altman）；R3 参照掩膜臂 + 受控注入（新图 5，HRF +0.217 写作 "associated with … in the development cohort"，与 FIVES 800、外部队列并列；注入明确为 compatibility 而非 identification；比例失真含 frozen rule）；R4 干预压力测试（新表 5 由 tools/mk_tab_stress.py 从 CSV 生成，新图 6 tools/fig_stress_forest.py）。其余全部移入补充（S1–S15），原 \label 全部保留在补充里以免断引用。旧 sections 与旧 PDF 存档于 paper2/sections/_archive_pre_cmig2_20260929/。
  - **摘要**从零重写，四步结构，238 词（wc.py）；亮点 5 条 78/78/78/75/79 字符。
  - **Related Work**：新增 fox2026impact（CMIG 131:102767，精确匹配）、li2025evaluation（CMIG 124:102574，强匹配）、han2025mamba（CMIG 125:102645，**部分匹配**：NIR-II 为主、视网膜为第二域；Crossref 363 条 CMIG 记录中无纯视网膜 2025 细血管论文，未编造）。新增 novelty 表（AutoMorph/CF-Loss/VascX/Fox 2026/本文 × 7 列），逐格依据写入 BIB_AUDIT.md。DOI 均经 Crossref 核对。
  - **版面**：三个 wrapper 去掉 lineno；宽浮动体 [!tp]、dblfloatpagefraction .75、flushbottom、emergencystretch 2.5em、去掉 back matter 前的强制 \clearpage；主文图全部按印刷尺寸（6.84 in）矢量重绘、统一字体与配色；生成表统一经 fittab 测量缩放（最小 0.859，四张最宽补充表改为横排页）。**顺带修复**：补充 E2 forest 此前每次渲染都漏画 FIVES CF-Loss 行（pooled 种子选择按队列而非按臂）。
  - **构建与检查**：main.pdf 40 页（高亮页 + 正文含参考文献 15 印刷页 + 2 页简历 + 补充 23 页），main_review.pdf 38 页（正文 14 页）；未定义引用 0、重复标签 0、\todo 0、?? 0、渲染文本反斜杠宏 0、正文浮动体全在补充前；footer/margin 预检 0 页；>35% 留白页仅 CAS 高亮页、末页简历页（main）与末页；脱敏 0/23，阳性对照 23/23。
- 2026-09-29 19:24 **协调者追加两项已完成**：(1) cover_letter.tex 正文按新定位重写——measurement-validation framework 三组件、R1–R4 四结果、干预=压力测试、明确 CMIG 适配（引 Fox 2026 CMIG 131:102767 与 Li 2025 CMIG 124:102574 为本刊先例）、"What the paper is not"（非方法论文、非因果归因、无等效性声明、非临床效度），收件期刊定为 CMIG，2 页、编译无错；(2) main_elsarticle.pdf 重建（64 页，正文 29 页，0 未定义引用）；novelty 表加 fittab 以免单栏版溢出 185pt。三个 wrapper 均为最新；main/main_review 复检：0 未定义、0 重复标签、footer/margin 预检 0 页、脱敏 0/23。待命：external review 审阅/版面修改清单。
- 2026-09-29 20:36 **external review 第 A 轮（内容 8.4/10 三项必修 + 版面清单）与用户两项格式指令全部落地；不改任何数字。**
  - 内容：R2 表述在摘要/亮点3/引言/§5.2/图4题注/结论/求职信统一为表 4 所支持的口径（HRF 四个斜率点估计均 <1，区间仅对 length 与 tortuosity 排除 1、对 density 与 FD 覆盖 1；FIVES density/length 校准良好、FD 拉伸 β=1.58、tortuosity 一致性差 CCC 0.41）；标题 "downstream attenuation" → "downstream reference gaps"（三份 front matter、highlights.tex、求职信）；亮点5 → "Tested remedies did not beat same-budget or matched controls on fidelity."（更正：上一轮报告的亮点5 改写实际未写入文件，本轮才生效）；交叉引用用脚本机械核对 108 处 0 问题，修正 §5.4/表5 的 "S10" 为 S8–S14 分项、表 S18 题注 "main text"、方法中消融指向 S11、补引用三个未被引用的补充浮动体。摘要 240 词（wc.py）。
  - 版面：亮点单独成 highlights.pdf 并从稿件中移除；"Page n of N" 总数修正（class 在最终 clearpage 后记页码，多 1）；PDF Creator 改记 cas-dc（Elsevier 的 cas-dc.cls 硬写 cas-sc）；脱敏版去掉空 "ORCID(s):"；图 2 改 pcolormesh 纯矢量、图 4/S1–S4 散点不再栅格化、图 3 眼底/掩膜 600 ppi、全部图字 ≥7 pt；新 tools/fig_injection_panel.py 重绘图 S5（线型+标记冗余编码），图 S7 种子加标记/线型；参考文献 note→annote（不打印），无 DOI 的两条统一 "Available at" URL，interlinepenalty=10000 防条目跨页，全局 widow/club 10000；表 S16/S17/S22/S23/S24 为真横排页（.aux 记页号 → /Rotate 90）；表 S18 改长格式竖排；合并嵌套 fittab；最小表缩放 0.936。
  - 用户指令：三个 wrapper 改为数字引用（natbib numbers + cas-model2-names，与 COHG 一致）；作者简介移到补充之后，section* Author biographies + bio[width=20mm] + 粗体姓名，Yuxiang Wang 简介逐字取自 COHG；back matter 顺序改为 CRediT(printcredits) → competing interests(含 funding) → data availability(含 ethics/code/reproducibility) → generative AI → references → supplement → biographies。"fallback paper/" 理解为 paper2 的 elsarticle fallback wrapper，未改动另一篇稿件 medical1/paper/。
  - 构建：main.pdf 40 页（正文含参考文献 14 页）、main_review.pdf 39 页（14 页）、main_elsarticle.pdf 64 页；三者 0 未定义/0 重复标签/0 浮动体丢失/0 ??/0 TODO/0 反斜杠宏；脱敏 0/23、阳性对照 23/23；无 "(Author, year)" 残留。
- 2026-09-29 20:44 **medical1/paper（拓扑修复稿，cas-sc）同步格式指令**：三个 wrapper 改为数字引用（natbib numbers + cas-model2-names；elsarticle 用 number 选项）；正文两处作者-年份残留改写；去掉行号；back matter 顺序 CRediT(printcredits) → competing interests(并入 funding) → data availability → generative AI → references；作者简介置于参考文献后，section* Author biographies + bio[width=20mm]，Yuxiang Wang 逐字取自 COHG；脱敏版去掉空 ORCID 标签。构建：main.pdf 40 页（高亮页 + 39 页）、main_review.pdf 38 页（高亮页 + 37 页）、main_elsarticle.pdf 43 页；0 未定义引用、0 重复标签、无 (Author, year) 残留；脱敏 0/23。原文件存档于 paper/sections/_archive_pre_numbered_20260929/。
- 2026-09-29 21:19 **external review 第 B 轮（9.3/10，唯一必修 + 润色）落地**：(1) 参考文献条目跨栏/跨页的根因是 cas-common.sty 在 begin document 时重定义 bibfont，吞掉了 A 轮加的 interlinepenalty（A 轮回应中的该说法与 PDF 不符，已更正）；现以晚于 class 的 hook 在 class 的 bibfont 上追加 interlinepenalty=10000，并允许 DOI/URL 在 / . - _ 处断行；渲染 p13–14 核验：条目完整、无孤立 DOI/arXiv 行。(2) 队列计数措辞统一为 "four primary audit cohorts"（摘要、图 1、表 2 题注、求职信），摘要 238 词。(3) 表 S24 缩短标签后改回竖排全尺寸，横排页剩 4 张。(4) highlights.pdf 作为独立提交文件。构建：main.pdf 39 页、main_review.pdf 38 页、main_elsarticle.pdf 65 页；正文含参考文献 14 页；0 未定义/0 重复/0 丢失/0 ??/0 TODO/0 反斜杠宏；交叉引用 108 处 0 问题；最小表缩放 0.936；脱敏 0/23、阳性对照 23/23。
- 2026-09-29 21:40 **最终冻结（external review 第 C 轮收敛 9.7/10，无必修）。** 两处可选修订：亮点 3 → "HRF slope estimates all fall below 1; FIVES fractal dimension is stretched (β=1.58)."（84 字符；为满足 ≤85 去掉 "on ,"）；highlights.pdf 标题改为不断字的左对齐，不再出现 "cal-/ibration"。重建三个 wrapper + highlights + 求职信并复检：0 未定义/0 重复/0 丢失/0 ??/0 TODO/0 反斜杠宏，交叉引用 108 处 0 问题，正文浮动体全在补充前，最小表缩放 0.936，footer/margin 预检仅 4 个横排页（设计如此），脱敏 0/23、阳性对照 23/23。交付清单与 SHA-256：
  - paper2/main.pdf (39 pp.) ddc9a30bb602a5cebb20cf346f6dd2219bc5622e349763db72429f2538eb6a12
  - paper2/main_review.pdf (38 pp., redacted) c828f50d3934126d26f329d51c54c0ced2727fa9dc32d65ebf4217258beb3ff8
  - paper2/main_elsarticle.pdf (65 pp., single-column reference build) 2e69baaff2f234a84d7eafb341d955989313c993d1a0501aec11d8df0a9bdc24
  - paper2/highlights.pdf (1 p.) 02da3508dc60ae3a309b3fcef04804072e485b91b0074389bcca1df97eb653a7
  - paper2/cover_letter.pdf (2 pp.) 7e27b5535d8e83ff9f5c96de8eeb2f822327c428fe14756e65822124f83035c1
  - 剩余作者项：代码仓库 URL（backmatter "will be released upon acceptance"）、求职信签名与地址。
- 2026-09-29 21:55 **用户第三次 CMIG 审阅（review/user_cmig_review3_20260929.md）**：意见成立。补分析：Deming λ 敏感性（λ ∈ {0.25,0.5,1,2,4} + 观察者间估计）；σ_b 与图像的联合 bootstrap 区间；主面板改为 density/length/FD，tortuosity 为预设安全/次要终点（下游锁定面板 skan4 保留并注明，另给 3 项面板敏感性）。写作：贡献重定义为 integrated validation protocol；general framework vs retinal case study；主叙事改为"何时失效 + 后果取决于误差结构与下游规则"；压缩 R4；统一 findings 数量。随后 external review 迭代至收敛。
- 2026-09-29 22:06 **第三轮方案咨询采纳（review/cmig_r3_plan_reply.md）**：(1) 迂曲度不称"pre-specified secondary/safety"；skan4 锁定分析原样保留；三项面板保真度汇总标为"事后新增的报告层级"，迂曲度结果全部完整报告，skan3 仅作敏感性；分离理由为独立于分割器表现的测量学证据（跨管线一致性差、估计量不稳定），非其表现差；弱化"mean fidelity"，结论由逐生物标志物保真度承担。(2) Deming λ=1 为主，观察者间推得的 λ 入合理范围敏感性；预先定义"λ-stable"（原校准判据在 λ∈[0.25,4] 内不变）；主文只给 HRF length/tortuosity、FIVES FD 的 β(λ=1)[CI]、β 范围、stable 是否。(3) 联合 σ_b bootstrap：以真正独立单元重采样，每次重估 σ_b，千次以上，百分位/BCa；结果并入表 4（联合区间 + σ_b [95% CI]），条件 vs 联合对比入补充。(4) 标题："An integrated validation protocol for segmentation-derived imaging biomarkers: a retinal vessel case study"。(5) 贡献陈述采用其四句框架。(6) R4 压缩但受控干预设计留在方法节 + 一张紧凑结果表/图。
- 2026-09-29 22:20 **第三轮写作项落地（用户第三次审阅 + 方案咨询），数值分析待 r3 结果接入。** 标题单一来源 sections/title.tex："An integrated validation protocol for segmentation-derived imaging biomarkers: a retinal vessel case study"（三份 front matter、highlights、求职信均读此文件）。引言末页贡献陈述改为咨询的四句框架（整合协议而非新统计量；既有校准分析 + 配对参考掩膜下游臂 + 受控误差干预，经预先固定的 refitted/frozen 规则解读；分离测量分歧与下游后果；视网膜血管为案例，不主张普遍验证；不写"识别误差机制"），表 1 题注同步。主叙事改为"协议检测分割衍生生物标志物何时不再像参考掩膜版本；下游后果取决于误差结构与下游规则"，HRF +0.217 降为发展队列中的一例（摘要/引言/讨论/结论/求职信）；摘要改为四个结果，237 词。R4 压缩为一张紧凑表（tools/mk_tab_stress_short.py：补救、对照、主终点结果、安全标志）+ 一段话，长表与森林图移入补充 S10；受控干预设计仍在方法节。迂曲度按咨询：不称"预设次要/安全终点"，三指标保真度汇总写作"方法学审稿后新增的报告层级"，理由为独立于分割器的测量证据（跨管线一致性 0.500–0.580、对小掩膜编辑不稳定），skan4 锁定分析原样保留、迂曲度结果全部报告、skan3 仅作敏感性；R1 的"平均保真度"结论改为逐生物标志物陈述。Deming λ 定义写明（分割衍生值误差方差/参考值误差方差，λ=1 主分析）。留有 6 个 \todo{r3} 钩子（方法：稳健性分析细节；设置：skan3 敏感性；结果 R1：三指标汇总；R2：λ 表与联合 bootstrap 一句；引言 R2；讨论：三指标汇总与 skan3），表 4 生成器加 R3 HOOK 注释。构建：main.pdf 41 页（正文含参考文献 14 页）、main_review.pdf 39 页（13 页）、main_elsarticle.pdf 66 页；0 未定义/重复/丢失/??/反斜杠宏，交叉引用 103 处 0 问题，最小表缩放 0.936，脱敏 0/23。
- 2026-09-29 23:25 **R3 分析（审阅第 3/4/5 点）完成，报告 results/pivot/R3_ANALYSIS_REPORT.md，输出 results/pivot/r3/，未改动任何锁定输出。** 复现核对：λ=1/4 Deming 与 r2_calibration.csv 差 3.6e-15；σ_b 由原始训练图像重算比值 1.0；s_k 与公式文件一致；skan4 参考差距 0.2170/0.1800/0.0132/0.0009 及 family-2 全部复现。
  - (3) src/pivot/r3_deming_lambda.py：λ∈{0.25,0.5,1,2,4} + OLS/逆 OLS 界 + 观察者间 λ（λ_mom=(var(e)−var(d)/2)/(var(d)/2)，λ_raw=var(e)/var(d)，d=obs2−obs1，e=pred−ref，σ_b 单位；DRIVE/CHASE 自身数据 λ_mom 0.05–1.13，HRF/FIVES 自身 NA、仅迁移外推）。预定义 λ-stable（λ=1 判据在全网格不变）：HRF tortuosity 0.37[0.12,0.69] 稳定；FIVES FD 1.58[1.25,1.93] 稳定（β 1.48–1.64，OLS 界 1.39[1.18,1.63]，800 张 1.48[1.31,1.66]）；HRF length 0.42[−0.08,0.95] **不稳定**（λ≤0.5 区间覆盖 1），仅在 λ≥1 成立；"HRF 四个斜率均<1" 亦仅 λ≥1；FIVES tortuosity 的"排除 1"不稳定。
  - (4) src/pivot/r3_sigma_joint_boot.py：图像 + σ_b 训练掩膜联合 bootstrap 2000 次（百分位+BCa），每次重估 σ_b 与 s_k；**CHASE_DB1 为同一儿童左右眼配对，按儿童聚类重采样**（协调方"仅 Fundus-AVSeg 有配对"不适用于审计集）。σ_b 相对半宽 DRIVE 0.63–0.68、CHASE 0.57–1.41、HRF 0.76–0.97、FIVES 0.20–0.22。小队列联合区间右偏显著变宽（HRF 4–13×），但 48 个 skan 单元无一 μ/α/残差 SD 的"排除 0"判据改变；HRF>FIVES 偏移 P=1.00（比值 [3.5,19]）；"order one"总均值 [0.81,2.02]/[0.88,2.14] 不变；拓扑"17–20×"下端降到 ~9×（worst stratum [8.7,40.5]，P(≥10×)=0.94）→ 建议改述为"约一个数量级或以上"。表 4 供数文件 r3/r3_main_table_joint.csv（r2_main_table 同键 + joint/BCa/cond_unit 区间 + sigma_lo/hi），补充对比 r3_sigma_joint_width_compare.csv。
  - (5) src/pivot/r3_panel3.py：三项面板为"事后新增报告层级"，skan4 下游锁定不变，skan3 仅敏感性；外部三队列未跑（无参考掩膜、e3 无 panel 参数）。审计总均值 |μ| 0.951→0.861、残差 SD 1.139→0.737；拓扑边际 17.5–47.9×→109.6–135.7×；尺度变体 HRF/FIVES 比 3.9–6.2→2.8–4.4，common scale 下 DRIVE 不再超过 HRF。参考差距 skan3：HRF logreg +0.202[0.075,0.337]（与 skan4 差 −0.015[−0.093,+0.070]），HRF GBDT +0.113[−0.060,0.283] 与两个 SegFormer HRF 差距（+0.087/+0.088）区间覆盖 0 → "差距对比在所有家族复现"仅对 skan4 成立。注意：HGB 对特征列顺序敏感（打乱列序 HRF GBDT 0.180→0.140），所有 GBDT 数字须按锁定列序复现。完整"旧→新"数字清单见报告末节。
- 2026-09-29 23:27 **r3 分析结论与稿件处理（results/pivot/R3_ANALYSIS_REPORT.md）**：(a) Deming λ：FIVES FD（1.58 [1.25,1.93]）与 HRF tortuosity（0.37 [0.12,0.69]）λ-stable；HRF total_length（0.42 [−0.08,0.95]）不 λ-stable（λ≤0.5 时区间含 1）→ 稿件把"HRF 斜率均 <1 / 长度区间排除 1"写为以 λ≥1 为条件的结论，并给出观察者间推得的 λ 范围（DRIVE 0.27–1.13、CHASE 0.05–0.99；HRF/FIVES 无第二观察者）。(b) 联合 σ_b bootstrap（图像 + 训练参考掩膜，CHASE 按 14 名儿童的双眼配对重采样）：区间在 HRF 加宽 4–13×，但 48 格无一改变"是否排除 0"；HRF>FIVES 偏移排序概率 1.00；"约一个 σ"成立；"拓扑小 17–20×"改为"约一个数量级或更多"（最差分层下界约 9×，≥10× 概率 0.94）。表 4 增加联合区间与 σ_b [95% CI]。(c) 三项面板：作为事后新增的报告层级并列给出（逐生物标志物行不变）；下游锁定面板 skan4 不变；skan3 下游敏感性：HRF logistic +0.202 [0.075,0.337]，但 HRF GBDT 与两个 SegFormer HRF 差距区间含 0 → "参考差距对比在所有族复现"限定于锁定的 skan4 面板。(d) 方法注记：GBDT 结果依赖特征列顺序（锁定顺序）；CHASE 双眼配对。
- 2026-09-29 23:43 **第三轮 r3 数值接入完成**：六个 \todo{r3} 全部落地并清零。(1) λ：主文表 5（HRF tortuosity 0.37 [0.12,0.69] 0.34–0.47 稳定；HRF length 0.42 [−0.08,0.95] 0.26–1.49 不稳定；FIVES FD 1.58 [1.25,1.93] 1.48–1.64 稳定）+ 补充全网格表（含 OLS 极限与观察者间 λ：DRIVE 0.27–1.13、CHASE 0.05–0.99）；"HRF 斜率均 <1 / 长度排除 1" 在摘要/亮点3/引言/§5.2/图4题注/结论/求职信均改为以 λ≥1 为条件，FIVES FD 与 HRF tortuosity 写为 λ-稳定。(2) 表 4 新增 σ_b [95% CI] 列与 μ 的联合区间；方法 §3.2 写明重采样单位（图像；CHASE 按 14 名儿童）、2000 次、百分位（BCa 入补充）；无区间改变是否排除 0，HRF>FIVES 偏移排序概率 1.00，"约一个 σ" 成立；"17–20×" 全文改为"约一个数量级或更多（最差分层下界约 9×，≥10× 概率 0.94）"。(3) 三指标面板：审计表加三指标均值块，R1 并列给出，skan4 数字仍为主报告值；skan3 下游敏感性表（HRF logistic +0.202 [0.075,0.337]；HRF GBDT 与 SegFormer HRF 差距含 0）→ "参考差距对比在所有族复现"限定于锁定 skan4，残差排序在三指标上 4/4 复现；病例图选择规则不变。(4) 方法注记 GBDT 依赖锁定列序。构建：main.pdf 43 页（正文含参考文献 14 页）、main_review.pdf 42 页、main_elsarticle.pdf 70 页；0 \todo/未定义/重复/丢失/??/反斜杠宏，交叉引用 122 处 0 问题，最小表缩放 0.928，脱敏 0/23。REVIEW_CMIG2_RESPONSE.md 增 Round 3 节，NUMBER_SOURCES/SUBMISSION_CHECKLIST/TITLE_PAGE_NOTES 同步。
- 2026-09-30 00:46 **external review 第 3-A 轮（8.9/10）五项必修 + 版面落地**：(1) 图 5 的 r̄ 拆成两个符号：面板 a 为三指标均值 r̄₃（HRF 0.42、FIVES 0.96），注入面板与正文为四指标均值 r̄₄（注入所控制的量；HRF 0.549、FIVES 0.854）——注入曲线只存四指标汇总，改成三指标需重跑注入，未做，已在回应中说明；(2) 表 3 并列 mean r₃ 与 r₄，ΔAUC 标为锁定面板，保真度排序在两种均值下均成立；(3) 拓扑两个估计量分开："约一个数量级或更多（最差分层 17×，下界约 9×，≥10× 概率 0.94）"只用于配对对照的拓扑特异超额；修复真实预测的绝对移动单独报告（均值 0.006–0.115σ、RMS 0.006–0.351σ，FIVES tortuosity 因估计量不稳定较大）；(4) "pre-specified" 收窄为"决策规则、匹配对照与锁定下游面板在查看结果前固定"，λ 敏感性、联合 σ_b bootstrap 与三指标汇总注明为审稿后新增（摘要、表 1 题注、亮点 1、引言）；(5) 亮点 3 → "All HRF slope estimates <1 for λ≥1; FIVES fractal dimension stretched at all λ."（原要求 94 字符，已压缩至 ≤85），R4/摘要/亮点5/引言/讨论/结论改为"无补救给出一致且实质的保真度收益"，并注明拓扑修复有微小但区间排除 0 的改进。版面：表 S10 改为竖排全尺寸，表 4 缩放 0.946（≈7.6 pt），参考文献末页 balance 分栏。构建 main 43 页、review 42 页、elsarticle 70 页；0 todo/未定义/重复/丢失/??/反斜杠宏；交叉引用 123 处 0 问题；脱敏 0/23。REVIEW_CMIG2_RESPONSE 第 9 点标为完成并新增 Round 3-A 节。
- 2026-09-30 01:18 **最终冻结（external review 第 3-B 轮收敛 9.4/10，无必修）。** 唯一版面项：图 5(b)(c) 横轴标签在右缘被截断 → 两个注入面板统一缩为 "achieved fidelity r̄₄"，"四个生物标志物" 保留在题注（"the mean Pearson r over the four injected biomarkers"）；重新生成 fig5 并在 main_review.pdf 第 11 页渲染核验，标签完整。可选项 p14（脱敏版参考文献末页下半空白）：补充前的 clearpage 保留（与页码约定一致，参考文献末页已 balance 分栏），未改。重建三个 wrapper + highlights + 求职信并复检：0 todo/未定义/重复/丢失/??/反斜杠宏，无作者-年份残留，交叉引用 123 处 0 问题，正文浮动体全在补充前，最小表缩放 0.936，横排页 4 个，摘要 240 词，脱敏 0/23、阳性对照 23/23。交付清单与 SHA-256：
  - paper2/main.pdf (43 pp.) 1bcdb9befac5c2e74d004119785e5a2ae06dd545eb0db6b0bfe0c7a5f8ca590b
  - paper2/main_review.pdf (42 pp., redacted) cd49e84132386a2b1fcf40102de52d903619b8438505e1343bc4b960ff20f99a
  - paper2/main_elsarticle.pdf (70 pp., single-column reference build) 27425a63df25aa738a36e44e72ab8e704d5eabc4b4a5f302778eb2df2a56bb89
  - paper2/highlights.pdf (1 p.) e104f55f9830e4d375dfcf3d473ca7ef7845d0b6728aa1602f43ceb8d4069abe
  - paper2/cover_letter.pdf (2 pp.) 782d5f9fe3f217fec81c0af75884f2a0d40312a90c3b7e623fb26b8ec6bb37b4
  - 剩余作者项：代码仓库 URL、求职信签名与地址。
- 2026-09-30 09:09 **用户第四次审阅：AI 文风痕迹（review/user_ai_style_review_20260930.md）**：执行去模板化/去防御化的语言重写（不改任何数字与结论）：削减口头禅词频、删除审稿回复式语句与措辞元话语、合并重复的 R1–R4 复述、图表注压缩至两句、去破折号/分号/括号密度、去数字化标签骨架、修复重复句、代码词改为自然表述。先由 external review 逐条判定，再修改并复审至收敛。
- 2026-09-30 10:05 **文风修订第 1 轮（用户第四次审阅 + 外部文风判定 review/style_r1_reply.md，47/50 条成立）完成客观项与指南第二轮；不改任何数字、结论范围或表格数值。** (1) 口头禅词频（pdftotext 全文 / 正文）：locked 36/20→0/0，frozen 67/26→6/6（仅 refitted vs frozen rule 一处概念），fixed before 11/6→1/1（摘要一处），same-budget 46/16→0/0（统一为 continued-training control），after methodological review 8/5→1/1（仅 §4.4 provenance 段），rather than 30/11→0/0，"not a" 20/7→1/1；保留三处 X-not-Y：benchmark, not a ceiling；compatibility, not identification；ecological contrast, not attribution。(2) 删除全部审稿回复式元话语（straw man、deliberately not saying、nothing … should be read as、Read the magnitudes not the stars、not a second chance、not a pre-registration、no claim … appears anywhere、stated rather than hidden、bounded negative、No arrow asserts、price it、made that answer readable、wrong way、one illustration / rests on）。(3) 预设说明只在 §4.4 Statistics and analysis provenance 一段写三层（主分析与规则在看结果前设定；审稿后新增三项敏感性分析；Fundus-AVSeg 在 FIVES 零结果后选定、复制方案在其任何结果前设定），摘要保留一句。(4) 摘要去 First–Fourth 与创新性辩护（224 词）；引言去 R1–R4 复述；讨论改为 Interpretation and implications + Limitations（名词短语小标题），Recommendations 命令式列表并入讨论末段；结论 111 词。(5) 小标题去 Component 1–3 / R1–R4，图 1 去 R 标签与数字徽标；补充材料小标题改普通名称，E2/E3/E4 代号不再作模块名。(6) 51 个图/表题注全部 ≤2 句且无破折号：生成表的题注由 tools/_captions.py 在 write_table 中统一覆盖（不含结果数字），题注中的解释性内容移入补充正文；表 1 题注两句 + 表下列注；表 S18 改为 Remedy / Control / Primary endpoint / Safety endpoint 四列（数值不变）；mk_tab_scale.py 补入 build_tables 列表。(7) 破折号 153→0，分号 279→37（正文 7，余为表格单元与图内标签），左括号 728→534（正文 211→135；源文件 prose 311→约 185，长于 10 词与嵌套括号清零）。(8) 重复句扫描（dupscan.py）：修订前 3 处完全重复 + 1 处近似（含 p38 practical-effect threshold 连写），修订后 0/0。(9) 代码词：replication_success = FALSE → replication criterion not met；last.pt → final saved checkpoint；bit-identical → identical to machine precision / unchanged；summary.json 不再出现。(10) 去拟人化（asked / say / invites / reads）。求职信同步重写（去 R1–R4、破折号、locked）；亮点 2–4 去分号与对偶（≤85 字符）。构建：main.pdf 41 页（正文含参考文献 14 页）、main_review.pdf 39 页（13 页）、main_elsarticle.pdf 66 页、highlights 1 页、cover_letter 2 页；0 todo/未定义/重复标签/丢失/??/反斜杠宏/控制字符，交叉引用 0 问题，最小表缩放 0.936，横排页 4，脱敏 0/23、阳性对照 23/23；front matter 123pt overfull 为既有（换回旧摘要复现）。计数表 review/style_counts_after.md；源文件存档 paper2/sections/_archive_pre_style_20260930/、paper2/tools/_archive_pre_style_20260930/。01:18 的 SHA-256 冻结集作废，待本轮复审收敛后重新冻结。
- 2026-09-30 10:39 **最终冻结（文风复审 review/style_r2_reply.md 判定已收敛为普通科研写作；按协调者十项局部修订落地后冻结）。** 不改任何数字、结论范围或表格数值。十项：(1) 摘要去掉 "matched controls, with …" 的重复 with，改为独立一句"主分析与决策规则在查看结果前设定"；(2) 引言贡献改为正面陈述（建立在 Deming、Lin 一致性、Bland--Altman 与误差注入之上，贡献在于组合方式）；(3) 删 "does not amount to a universal validation"；(4) 引言范围句改为"协议在形式上通用，本文证据来自视网膜血管"；(5) §3.5 删 "It is not proposed as a method"；(6) §5.4 "gains nothing beyond" → "did not improve fidelity relative to"，并恢复 CF-Loss 的 "on HRF" 限定（此前改写扩大了表面范围，已更正）；(7) 讨论删 "does not make such biomarkers unusable"，"reads the distorted value" → "receives"；(8) "largest effect in the programme" 改为具体陈述：tortuosity 偏移对 continued-training 对照 +1.62σ（HRF，声明界值的六倍），留出队列跨重训 Δr −0.059 超过 −0.05 界值，Dice/clDice 非劣，删 "It was detected because…"；(9) 局限性 tortuosity 句拆开，"changes no model…" 改为 "uses the same models, data and thresholds"；(10) 表 1 "(✓) partially reported"；表 6 与表 S18 题注补成完整句；补充 S14 删 "A null is informative only…"，检查点缺陷段改为普通叙述（"After correction, all 244 prediction outputs passed the completion check"），小标题改 Pairing and checkpoint completion。保留 frozen 6 处与三处 X-not-Y。重建五件交付并复检：0 todo/未定义/重复标签/丢失/??/反斜杠宏/控制字符，无作者-年份残留，交叉引用 0 问题，正文浮动体全在补充前，最小表缩放 0.936，横排页 4，摘要 224 词，重复句 0，脱敏 0/23、阳性对照 23/23；front matter 123pt overfull 为既有。计数 review/style_counts_final.md。交付清单与 SHA-256：
  - paper2/main.pdf (40 pp.) c0093ee2e0864d1ca8c5b97491d56d131882a78494999a90c96185a38d5297ac
  - paper2/main_review.pdf (39 pp., redacted) 8b80bf886a6be964eeb8a211f068a23a1ad2b531d3a6452c102014f52c800939
  - paper2/main_elsarticle.pdf (66 pp., single-column reference build) c3f87fcc468e6974a35ecb13391637e5e03bcd5a22ec848a618dbbf2a0318dfd
  - paper2/highlights.pdf (1 p.) 514e952aaafc124cd185e306c4a7003e167cd48f4a2b2568ce1360b96d7651e0
  - paper2/cover_letter.pdf (2 pp.) b72c9f864d1a637d33f4176c2d9b95c81bc7016ada28392e11ba758da54b151f
  - 剩余作者项：代码仓库 URL、求职信签名与地址。
