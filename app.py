import json
from dataclasses import asdict

import numpy as np
import os
import pandas as pd
import streamlit as st

from nlp_pyramidal_flow.cleaning import clean_text, detect_issues
from nlp_pyramidal_flow.config import PIGConfig
from nlp_pyramidal_flow.pyramidal_flow import PyramidalFlowPerturber
from nlp_pyramidal_flow.rectified_flow.model import RFModelConfig
from nlp_pyramidal_flow.rectified_flow.sampler import (
    ODESolveConfig,
    solve_to_clean_embedding,
    solve_trajectory,
)
from nlp_pyramidal_flow.rectified_flow.trainer import RFTrainConfig, train_rectified_flow
from nlp_pyramidal_flow.tasks.embeddings import TfidfEmbedder
from nlp_pyramidal_flow.directed import DirectedConfig, compute_directed_spans
from nlp_pyramidal_flow.quality import GPT2FluencyScorer, score_quality

try:
    from nlp_pyramidal_flow.embeddings.transformer_embedder import TransformerEmbedder
except Exception:
    TransformerEmbedder = None  # type: ignore
from nlp_pyramidal_flow.tasks.classification import (
    ClassificationTrainer,
    evaluate_classifier,
)
from nlp_pyramidal_flow.tasks.matching import cosine_match
from nlp_pyramidal_flow.tasks.ner import simple_ner
from nlp_pyramidal_flow.tasks.qa import retrieval_qa
from nlp_pyramidal_flow.tasks.retrieval import SimpleRetriever
from nlp_pyramidal_flow.tasks.generation import template_generate
from nlp_pyramidal_flow.style import apply_style


st.set_page_config(page_title="NLP伪干扰生成系统", layout="wide")


from typing import Optional


def _section_title(title: str, desc: Optional[str] = None) -> None:
    st.markdown(f"### {title}")
    if desc:
        st.caption(desc)


def _sidebar_config() -> PIGConfig:
    st.sidebar.markdown("### 伪干扰配置")
    st.sidebar.caption("从粗到细：句级→词/短语→子词/格式→字符噪声。")

    seed = st.sidebar.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1)
    lang = st.sidebar.selectbox("语言/文本类型", ["自动", "中文", "英文"], index=0)

    st.sidebar.markdown("#### 强度（0~1）")
    sent = st.sidebar.slider("句级（结构/连接词/赘词）", 0.0, 1.0, 0.25, 0.05)
    word = st.sidebar.slider("词/短语（轻量替换/插入）", 0.0, 1.0, 0.25, 0.05)
    subw = st.sidebar.slider("子词/格式（空白/全角半角/符号）", 0.0, 1.0, 0.20, 0.05)
    char = st.sidebar.slider("字符（错别字/形近/同音/键盘）", 0.0, 1.0, 0.20, 0.05)

    st.sidebar.markdown("#### 预算")
    max_ops = st.sidebar.slider("最大操作数（总预算）", 1, 64, 16, 1)
    protect_numbers = st.sidebar.checkbox("保护数字/URL/邮箱（尽量不改）", value=True)

    return PIGConfig(
        seed=int(seed),
        lang_hint=lang,
        intensity_sentence=float(sent),
        intensity_word=float(word),
        intensity_subword=float(subw),
        intensity_char=float(char),
        max_ops=int(max_ops),
        protect_numbers=bool(protect_numbers),
    )


def main() -> None:
    st.title("NLP伪干扰生成系统")
    st.caption("面向真实 NLP 脏数据与表达变化：多粒度、可控、可追踪的伪干扰生成 + 多任务鲁棒性实验台。")

    cfg = _sidebar_config()
    perturber = PyramidalFlowPerturber(cfg)

    page = st.sidebar.radio(
        "功能页",
        [
            "伪干扰生成器",
            "多任务工作台",
            "Rectified Flow（训练/清洗演示）",
            "攻防小游戏（你来修复）",
            "批量增强与导出",
            "增强训练（分类）",
            "鲁棒评测（分类）",
            "对抗攻防（分类）",
            "数据清洗测试",
        ],
        index=0,
    )

    if page == "伪干扰生成器":
        _section_title("伪干扰生成器", "输入一段文本，系统按“句→词→子词/格式→字符”分层注入扰动，并提供 trace 便于写报告。")
        left, right = st.columns(2, gap="large")
        with left:
            x = st.text_area(
                "原始文本 x",
                height=220,
                value="因为天气不好，所以我们取消了户外活动。请把会议时间改到明天上午10点，谢谢！",
            )
            t = st.selectbox("干扰风格 t（可选）", ["通用", "OCR风格", "ASR风格", "口语化", "社交媒体"], index=0)
            fun_style = st.selectbox("趣味风格偏向（可选）", ["默认", "口语化", "轻微消极", "轻微积极", "实体模糊"], index=0)
            directed_mode = st.selectbox(
                "定向干扰模式（可选）",
                ["通用", "NER-保护实体", "匹配-保留关键词", "QA-改写问题"],
                index=0,
            )
            keyword_topk = st.slider("关键词保护数量（匹配模式）", 2, 20, 6, 1)

            st.markdown("#### 质量评分（自动）")
            use_gpt2 = st.checkbox("流畅度用 GPT2 打分（需下载模型，较慢）", value=False)
            gpt2_model = st.selectbox("GPT2 模型（英文更准）", ["distilgpt2", "gpt2"], index=0, disabled=not use_gpt2)
            sem_backend = st.selectbox("语义相似度后端", ["TF-IDF（快速）", "BERT/ERNIE（更强，需下载模型）"], index=0)
            sem_model = st.selectbox(
                "BERT/ERNIE 模型",
                [
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "bert-base-chinese",
                    "hfl/chinese-bert-wwm-ext",
                    "nghuyong/ernie-3.0-base-zh",
                    "bert-base-uncased",
                ],
                index=0,
                disabled=(sem_backend == "TF-IDF（快速）" or TransformerEmbedder is None),
            )
            if st.button("生成伪干扰文本", type="primary"):
                dcfg = DirectedConfig(mode=directed_mode, keyword_topk=int(keyword_topk))
                spans = compute_directed_spans(x, dcfg)
                y = perturber.perturb(x, style=t, extra_protected_spans=spans)
                st.session_state["last_x"] = x
                st.session_state["last_y"] = y
                st.session_state["last_t"] = t
                st.session_state["last_fun_style"] = fun_style
                st.session_state["last_directed"] = {"mode": directed_mode, "keyword_topk": int(keyword_topk), "spans": spans}
                st.session_state["last_quality_cfg"] = {
                    "use_gpt2": bool(use_gpt2),
                    "gpt2_model": str(gpt2_model),
                    "sem_backend": str(sem_backend),
                    "sem_model": str(sem_model),
                }

        with right:
            y = st.session_state.get("last_y")
            if y is None:
                st.info("点击左侧按钮生成。")
            else:
                fun_style = st.session_state.get("last_fun_style", "默认")
                x0 = st.session_state.get("last_x", "")
                x1 = apply_style(y.text, fun_style)
                st.text_area("伪干扰文本  x̃", value=x1, height=220)
                st.markdown("#### Trace（可追踪操作）")
                st.json(y.trace.to_dict())
                st.markdown("#### 定向模式/掩码")
                st.json(st.session_state.get("last_directed", {}))

                st.markdown("#### 质量评分（语义/流畅度/有效性）")
                qcfg = st.session_state.get("last_quality_cfg", {"use_gpt2": False, "sem_backend": "TF-IDF（快速）"})
                trainer = st.session_state.get("clf")  # type: Optional[ClassificationTrainer]

                flu_scorer = None
                if qcfg.get("use_gpt2"):
                    try:
                        @st.cache_resource
                        def _get_gpt2(model_name: str):
                            return GPT2FluencyScorer(model_name=model_name, device="cpu")

                        flu_scorer = _get_gpt2(str(qcfg.get("gpt2_model", "distilgpt2")))
                    except Exception:
                        flu_scorer = None

                sem_embedder = None
                if qcfg.get("sem_backend") != "TF-IDF（快速）" and TransformerEmbedder is not None:
                    try:
                        @st.cache_resource
                        def _get_sem(model_name: str):
                            return TransformerEmbedder(model_name=model_name, device="cpu")

                        sem_embedder = _get_sem(str(qcfg.get("sem_model")))
                    except Exception:
                        sem_embedder = None

                qs = score_quality(x0, x1, classifier=trainer, fluency_scorer=flu_scorer, semantic_embedder=sem_embedder)
                st.dataframe(pd.DataFrame([qs.to_dict()]))
                st.markdown("#### 配置快照")
                st.code(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    elif page == "多任务工作台":
        _section_title("多任务工作台", "同一段文本在不同任务下测试：抽取/匹配/检索/问答/生成，并支持一键“加伪干扰”。")
        x = st.text_area(
            "输入文本",
            height=160,
            value="华中科技大学的张三在2024年12月23日给李四发了邮件：zhangsan@example.com，讨论了大模型在中文NLP鲁棒性上的问题。",
        )
        t = st.selectbox("扰动风格", ["通用", "OCR风格", "ASR风格", "口语化", "社交媒体"], index=0)
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            use_noise = st.checkbox("对输入先施加伪干扰", value=False)
            if use_noise:
                out = perturber.perturb(x, style=t)
                x2 = out.text
                st.text_area("扰动后输入", value=x2, height=140)
            else:
                x2 = x

            st.markdown("#### 1) 实体抽取（启发式）")
            ents = simple_ner(x2)
            st.dataframe(pd.DataFrame(ents))

            st.markdown("#### 2) 文本匹配（余弦相似）")
            b = st.text_input("对比文本 B", value="张三在华中科技大学讨论中文NLP鲁棒性，并通过邮件联系对方。")
            sim = cosine_match(x2, b)
            st.metric("相似度", f"{sim:.4f}")

        with col2:
            st.markdown("#### 3) 检索 & 4) 问答（检索式）")
            kb = st.text_area(
                "知识库（每行一段文本/一条文档）",
                height=160,
                value="\n".join(
                    [
                        "华中科技大学位于武汉，是一所综合性研究型大学。",
                        "鲁棒性评测常用对比：clean 输入与带噪输入在指标上的变化。",
                        "伪干扰可用于增强训练，也可用于对抗攻防与数据清洗测试。",
                    ]
                ),
            )
            docs = [d.strip() for d in kb.splitlines() if d.strip()]
            retriever = SimpleRetriever(docs)
            q = st.text_input("问题（QA）", value="为什么要做鲁棒性评测？")
            if st.button("检索并回答", type="primary"):
                ans = retrieval_qa(question=q, retriever=retriever)
                st.markdown("**Top 文档**")
                st.write(ans.top_doc)
                st.markdown("**回答（抽取式/检索式）**")
                st.write(ans.answer)

            st.markdown("#### 5) 生成（模板式）")
            prompt = st.text_input("提示词/需求", value="把输入文本改写得更口语化，并保留关键信息")
            if st.button("生成文本", type="secondary"):
                st.write(template_generate(x2, prompt=prompt))

    elif page == "Rectified Flow（训练/清洗演示）":
        _section_title(
            "Rectified Flow（Flow Matching）训练/清洗演示",
            "在连续表征空间（TF‑IDF 向量）上训练速度场 vθ，使其回归 u_t = x1 - x0，并用 ODE 积分从噪声表征推回干净表征。",
        )

        st.markdown("#### 1) 训练数据（clean 文本集合）")
        st.caption("默认用 `data/sample_cls.csv` 的 text 作为 clean 语料，你也可以在下面粘贴更多（每行一条）。")
        extra = st.text_area("额外语料（可选，每行一条 clean 文本）", height=120, value="")

        try:
            df0 = pd.read_csv("data/sample_cls.csv")
            base_texts = df0["text"].astype(str).tolist()
        except Exception:
            base_texts = []

        extra_texts = [t.strip() for t in extra.splitlines() if t.strip()]
        clean_texts = list(dict.fromkeys(base_texts + extra_texts))
        st.write(f"当前 clean 语料条数：**{len(clean_texts)}**")

        st.markdown("#### 2) 训练配置")
        colA, colB, colC = st.columns(3)
        with colA:
            epochs = st.slider("epochs", 1, 50, 10, 1)
            bs = st.selectbox("batch_size", [16, 32, 64, 128], index=2)
        with colB:
            lr = st.selectbox("lr", [1e-4, 3e-4, 1e-3, 3e-3], index=2)
            noisy_per_text = st.slider("每条 clean 生成多少条 noisy（构造 x0）", 1, 10, 2, 1)
        with colC:
            hidden = st.selectbox("hidden", [128, 256, 384, 512], index=1)
            depth = st.selectbox("depth", [2, 3, 4, 5], index=1)

        style = st.selectbox("构造 noisy 的风格（用于 x0）", ["通用", "OCR风格", "ASR风格", "口语化", "社交媒体"], index=0)
        device = st.selectbox("device（无 GPU 就选 cpu）", ["cpu", "cuda"], index=0)

        if st.button("开始训练 Rectified Flow", type="primary"):
            if len(clean_texts) < 4:
                st.error("clean 语料太少：请至少提供 4 条以上文本。")
            else:
                with st.spinner("训练中...（小数据集通常几十秒内完成）"):
                    rf_art = train_rectified_flow(
                        clean_texts=clean_texts,
                        perturber=perturber,
                        style=style,
                        rf_cfg=RFModelConfig(dim=8, hidden=int(hidden), depth=int(depth), dropout=0.0),
                        train_cfg=RFTrainConfig(
                            epochs=int(epochs),
                            batch_size=int(bs),
                            lr=float(lr),
                            device=str(device),
                            seed=int(cfg.seed),
                        ),
                        noisy_per_text=int(noisy_per_text),
                    )
                st.session_state["rf_art"] = rf_art
                st.success("训练完成：模型已保存到当前会话（session）。")
                st.json(rf_art.meta)
                st.line_chart(pd.DataFrame({"train_loss": rf_art.train_loss}))

        st.divider()
        st.markdown("#### 3) 清洗演示：noisy →（RF ODE）→ clean embedding → 检索映射回文本")
        rf_art = st.session_state.get("rf_art")
        if rf_art is None:
            st.info("请先在上面训练一个 Rectified Flow 模型。")
        else:
            noisy_in = st.text_area("输入（可带噪/可先用上方伪干扰生成器生成）", height=150, value="这。。。家店服 务 真 的 很差！！！")
            # embed noisy
            emb: TfidfEmbedder = rf_art.embedder
            x0 = emb.transform([noisy_in])

            ode_steps = st.slider("ODE steps", 5, 200, 30, 5)
            direction = st.selectbox("轨迹方向（用于演示）", ["伪干扰→清洗（forward）", "原始→伪干扰（reverse）"], index=0)
            fun_style2 = st.selectbox("输出风格偏向（可选）", ["默认", "口语化", "轻微消极", "轻微积极", "实体模糊"], index=0, key="rf_fun_style")

            if st.button("执行 RF 并生成轨迹", type="secondary"):
                dir_key = "forward" if direction.startswith("伪干扰") else "reverse"
                traj = solve_trajectory(
                    x_start=x0,
                    model=rf_art.model,
                    device=str(device),
                    cfg=ODESolveConfig(steps=int(ode_steps)),
                    direction=dir_key,
                )  # [T,B,D]
                st.session_state["rf_traj"] = traj
                x1_hat = traj[-1, :, :]

                # map embedding back to text by nearest neighbor retrieval over clean corpus
                clean_mat = emb.transform(clean_texts)
                # cosine similarity
                num = (clean_mat @ x1_hat.T).reshape(-1)
                denom = (np.linalg.norm(clean_mat, axis=1) * np.linalg.norm(x1_hat.reshape(-1)) + 1e-12)
                sim = num / denom
                topk = np.argsort(sim)[::-1][:5]
                rows = [{"rank": i + 1, "text": apply_style(clean_texts[int(j)], fun_style2), "sim": float(sim[int(j)])} for i, j in enumerate(topk)]
                st.markdown("**Top-5 候选 clean 文本（按 embedding 相似度）**")
                st.dataframe(pd.DataFrame(rows))
                st.caption("说明：这里用“检索映射”把连续表征空间的 x̂1 映射回可读文本，便于系统演示与报告呈现。")

            st.markdown("#### 4) 动态流轨迹可视化（动画）")
            st.caption("将高维嵌入轨迹做 PCA 到 2D，用动画直观展示点在 Rectified Flow 中随 t 演化。")
            if st.button("播放轨迹动画", type="primary"):
                traj = st.session_state.get("rf_traj")
                if traj is None:
                    st.warning("请先点击“执行 RF 并生成轨迹”。")
                else:
                    try:
                        import importlib
                        px = importlib.import_module("plotly.express")
                        from sklearn.decomposition import PCA
                    except Exception as e:
                        st.error(f"缺少依赖：请安装 plotly。错误：{e}")
                        return

                    T, B, D = traj.shape
                    xs = traj[:, 0, :]  # [T,D]
                    pca = PCA(n_components=2, random_state=0)
                    pts2 = pca.fit_transform(xs)
                    dfp = pd.DataFrame({"step": list(range(T)), "x": pts2[:, 0], "y": pts2[:, 1]})

                    speed = st.slider("播放速度（ms/帧）", 50, 2000, 300, 50)
                    fig = px.scatter(
                        dfp,
                        x="x",
                        y="y",
                        animation_frame="step",
                        range_x=[dfp["x"].min() - 0.2, dfp["x"].max() + 0.2],
                        range_y=[dfp["y"].min() - 0.2, dfp["y"].max() + 0.2],
                        title="Rectified Flow 轨迹动画（PCA 2D）",
                    )
                    fig.update_traces(marker=dict(size=12))
                    fig.update_layout(
                        updatemenus=[
                            {
                                "type": "buttons",
                                "buttons": [
                                    {
                                        "label": "Play",
                                        "method": "animate",
                                        "args": [None, {"frame": {"duration": speed, "redraw": True}, "fromcurrent": True}],
                                    },
                                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}}]},
                                ],
                            }
                        ]
                    )
                    st.plotly_chart(fig, use_container_width=True)

    elif page == "攻防小游戏（你来修复）":
        _section_title(
            "攻防小游戏（你来修复）",
            "系统先生成伪干扰，你来“修复”它；系统根据语义相似度 + 下游任务稳定性给出攻防得分。",
        )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            orig = st.text_area("原文（你要保护的语义）", height=160, value="华中科技大学的张三在2024年12月23日给李四发了邮件，讨论中文NLP鲁棒性评测。")
            style = st.selectbox("生成伪干扰风格", ["通用", "OCR风格", "ASR风格", "口语化", "社交媒体"], index=0, key="game_style")
            fun_style = st.selectbox("趣味风格偏向", ["默认", "口语化", "轻微消极", "轻微积极", "实体模糊"], index=0, key="game_fun")
            if st.button("生成伪干扰（作为攻击）", type="primary"):
                pr = perturber.perturb(orig, style=style)
                adv = apply_style(pr.text, fun_style)
                st.session_state["game_orig"] = orig
                st.session_state["game_adv"] = adv
                st.session_state["game_trace"] = pr.trace.to_dict()

            adv = st.session_state.get("game_adv", "")
            st.text_area("伪干扰文本（攻击结果）", value=adv, height=160)
            if st.session_state.get("game_trace") is not None:
                st.markdown("#### 攻击 Trace")
                st.json(st.session_state["game_trace"])

        with col2:
            adv = st.session_state.get("game_adv", "")
            repair = st.text_area("你的修复版本（手动编辑）", height=160, value=adv)

            trainer = st.session_state.get("clf")  # type: Optional[ClassificationTrainer]
            if trainer is None:
                st.caption("提示：当前没有训练好的分类器（增强训练页训练后，这里会加上“任务稳定分”）。")
            else:
                st.caption("提示：会同时评估“修复后预测是否回到原文预测”。")

            if st.button("评分（攻防得分）", type="secondary"):
                sim = cosine_match(orig, repair)  # 0~1
                score_sem = int(round(sim * 100))

                score_task = 0
                detail = {"semantic_similarity": float(sim)}
                if trainer is not None:
                    pred_o = trainer.predict([orig])[0]
                    pred_r = trainer.predict([repair])[0]
                    score_task = 100 if pred_o == pred_r else 0
                    detail.update({"pred_orig": pred_o, "pred_repair": pred_r})

                total = int(round(0.7 * score_sem + 0.3 * score_task))
                st.metric("语义修复分（0~100）", score_sem)
                if trainer is not None:
                    st.metric("任务稳定分（0/100）", score_task)
                st.metric("总攻防得分（0~100）", total)
                st.json(detail)

    elif page == "批量增强与导出":
        _section_title(
            "批量增强与导出",
            "上传数据后批量生成伪干扰文本，自动评分，并导出 CSV/JSONL（可选 HuggingFace Dataset 结构）。10万行以上建议用 batch_process.py。",
        )

        up = st.file_uploader("上传 CSV（至少包含 text 列，可选 label 列）", type=["csv"], key="batch_csv")
        if up is None:
            st.info("可先用示例数据 `data/sample_cls.csv`。")
            df = pd.read_csv("data/sample_cls.csv")
        else:
            df = pd.read_csv(up)

        if "text" not in df.columns:
            st.error("CSV 缺少 text 列。")
        else:
            text_col = st.selectbox("文本列", list(df.columns), index=list(df.columns).index("text"))
            label_col = st.selectbox("标签列（可选）", ["（无）"] + list(df.columns), index=(1 if "label" in df.columns else 0))

            style = st.selectbox("干扰风格", ["通用", "OCR风格", "ASR风格", "口语化", "社交媒体"], index=0, key="batch_style")
            directed_mode = st.selectbox(
                "定向干扰模式",
                ["通用", "NER-保护实体", "匹配-保留关键词", "QA-改写问题"],
                index=0,
                key="batch_directed",
            )
            keyword_topk = st.slider("关键词保护数量（匹配模式）", 2, 20, 6, 1, key="batch_kw")
            variants = st.slider("每条样本生成几条伪干扰", 1, 5, 1, 1)
            limit = st.slider("处理前 N 行（避免一次太大）", 10, min(20000, max(10, len(df))), min(200, len(df)), 10)

            st.markdown("#### 评分配置")
            use_gpt2 = st.checkbox("流畅度用 GPT2 打分（需下载模型）", value=False, key="batch_gpt2")
            gpt2_model = st.selectbox("GPT2 模型", ["distilgpt2", "gpt2"], index=0, disabled=not use_gpt2, key="batch_gpt2_model")

            if st.button("开始批量生成 + 评分", type="primary"):
                dcfg = DirectedConfig(mode=directed_mode, keyword_topk=int(keyword_topk))
                trainer = st.session_state.get("clf")  # type: Optional[ClassificationTrainer]

                flu_scorer = None
                if use_gpt2:
                    try:
                        flu_scorer = GPT2FluencyScorer(model_name=str(gpt2_model), device="cpu")
                    except Exception:
                        flu_scorer = None

                out_rows = []
                prog = st.progress(0)
                base = df.head(int(limit)).reset_index(drop=True)
                total = int(len(base) * int(variants))
                done = 0
                for i, r in base.iterrows():
                    x = str(r[text_col])
                    y = "" if label_col == "（无）" else str(r[label_col])
                    spans = compute_directed_spans(x, dcfg)
                    for k in range(int(variants)):
                        pr = perturber.perturb(x, style=style, extra_protected_spans=spans)
                        qs = score_quality(x, pr.text, classifier=trainer, fluency_scorer=flu_scorer)
                        out_rows.append(
                            {
                                "original": x,
                                "perturbed": pr.text,
                                "label": y,
                                "style": style,
                                "directed_mode": directed_mode,
                                **qs.to_dict(),
                            }
                        )
                        done += 1
                        prog.progress(min(1.0, done / max(1, total)))

                out_df = pd.DataFrame(out_rows)
                st.session_state["batch_out_df"] = out_df
                st.success(f"完成：生成 {len(out_df)} 条记录。")
                st.dataframe(out_df.head(50))

            out_df = st.session_state.get("batch_out_df")
            if out_df is not None:
                st.markdown("#### 导出")
                csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下载 CSV", data=csv_bytes, file_name="augmented.csv", mime="text/csv")

                # Build JSONL manually to preserve unicode
                import json as _json

                jsonl = "\n".join(_json.dumps(r, ensure_ascii=False) for r in out_df.to_dict(orient="records"))
                st.download_button("下载 JSONL", data=jsonl.encode("utf-8"), file_name="augmented.jsonl", mime="application/json")

                # Optional HF Dataset export
                try:
                    import tempfile
                    import zipfile
                    from datasets import Dataset

                    if st.button("导出 HuggingFace Dataset（zip）", type="secondary"):
                        with tempfile.TemporaryDirectory() as td:
                            ds = Dataset.from_pandas(out_df, preserve_index=False)
                            ds.save_to_disk(td)
                            zip_path = td + ".zip"
                            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                                for root, _, files in os.walk(td):
                                    for fn in files:
                                        full = os.path.join(root, fn)
                                        rel = os.path.relpath(full, td)
                                        zf.write(full, arcname=rel)
                            with open(zip_path, "rb") as f:
                                st.download_button(
                                    "下载 HF Dataset ZIP",
                                    data=f.read(),
                                    file_name="hf_dataset.zip",
                                    mime="application/zip",
                                )
                except Exception:
                    st.caption("如需 HuggingFace Dataset 导出，请安装 `datasets`（requirements.txt 已包含）。")

    elif page == "增强训练（分类）":
        _section_title("增强训练（分类）", "上传 `text,label` 数据，系统可在训练时混入伪干扰版本，得到更鲁棒的轻量分类器。")
        file = st.file_uploader("上传 CSV（text,label）", type=["csv"])
        if file is None:
            st.info("你也可以先用示例数据 `data/sample_cls.csv`。")
            try:
                df = pd.read_csv("data/sample_cls.csv")
                st.dataframe(df.head(10))
            except Exception:
                df = None
        else:
            df = pd.read_csv(file)

        if df is not None:
            aug_ratio = st.slider("增广比例（训练集里多少样本替换为伪干扰）", 0.0, 1.0, 0.5, 0.05)
            if st.button("训练分类器（含增广）", type="primary"):
                trainer = ClassificationTrainer()
                trainer.fit_with_augmentation(df, perturber=perturber, style="通用", aug_ratio=aug_ratio)
                st.session_state["clf"] = trainer
                st.success("训练完成，已保存到当前会话。你可以去“鲁棒评测/对抗攻防”页面继续。")

    elif page == "鲁棒评测（分类）":
        _section_title("鲁棒评测（分类）", "对比 clean vs perturbed 的准确率/F1，并输出“最脆弱”的扰动统计，便于写系统分析。")
        trainer = st.session_state.get("clf")  # type: Optional[ClassificationTrainer]
        if trainer is None:
            st.warning("当前会话还没有训练好的分类器。请先到“增强训练（分类）”训练。")
        else:
            file = st.file_uploader("上传测试 CSV（text,label）", type=["csv"], key="eval_csv")
            if file is None:
                df = pd.read_csv("data/sample_cls.csv")
            else:
                df = pd.read_csv(file)

            style = st.selectbox("评测干扰风格", ["通用", "OCR风格", "ASR风格", "口语化", "社交媒体"], index=0)
            if st.button("开始评测", type="primary"):
                report = evaluate_classifier(df, trainer=trainer, perturber=perturber, style=style)
                st.markdown("#### 指标对比")
                st.dataframe(pd.DataFrame(report["metrics"]))
                st.markdown("#### 脆弱样本（前 10）")
                st.dataframe(pd.DataFrame(report["worst_examples"]))

    elif page == "对抗攻防（分类）":
        _section_title("对抗攻防（分类）", "针对当前分类器：优先破坏对预测最关键的词/子词特征，在预算内寻找让预测翻转的扰动。")
        trainer = st.session_state.get("clf")  # type: Optional[ClassificationTrainer]
        if trainer is None:
            st.warning("当前会话还没有训练好的分类器。请先到“增强训练（分类）”训练。")
        else:
            x = st.text_area("输入要攻击的文本", height=180, value="这家店服务很差，太失望了。")
            style = st.selectbox("攻击时的风格（用于扰动算子选择）", ["通用", "社交媒体", "OCR风格"], index=0)
            if st.button("执行对抗攻击（贪心）", type="primary"):
                res = trainer.greedy_attack(x, perturber=perturber, style=style)
                st.markdown("#### 结果")
                st.write(res["summary"])
                st.text_area("对抗样本", value=res["adv_text"], height=160)
                st.markdown("#### Trace")
                st.json(res["trace"])

    elif page == "数据清洗测试":
        _section_title("数据清洗测试", "检测常见脏文本问题，并给出修复后的版本；可用于“伪干扰 → 清洗”闭环验证。")
        x = st.text_area(
            "输入脏文本（可先手动加点噪声）",
            height=200,
            value="这。。。家  店  服 务  真  的  很  差！！！  2024／12／23  联系：zhangsan @example.com",
        )
        col1, col2 = st.columns(2, gap="large")
        with col1:
            if st.button("检测问题", type="secondary"):
                issues = detect_issues(x)
                st.dataframe(pd.DataFrame(issues))
        with col2:
            if st.button("清洗并修复", type="primary"):
                y = clean_text(x)
                st.text_area("修复后文本", value=y, height=200)


if __name__ == "__main__":
    main()


