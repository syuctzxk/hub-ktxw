"""
GPT4Rec简化实现：基于生成式语言模型的个性化推荐框架
步骤1: 根据用户历史观看的电影标题，使用GPT-2生成代表用户未来兴趣的"搜索查询"
步骤2: 使用BM25Okapi算法从电影库中检索出最相关的电影作为推荐候选
"""

import re
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 确保已下载必要的nltk数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# 1. 加载数据
ratings = pd.read_csv("./M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv("./M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding="latin")
movies.columns = ["movie_id", "movie_title", "movie_tag"]


# 2. 数据预处理：为用户创建历史观看序列
def create_user_history(ratings_df, movies_df, min_ratings=5):
    """为每个用户创建历史观看电影序列"""
    # 过滤出评分较高的电影（假设评分>3表示喜欢）
    positive_ratings = ratings_df[ratings_df["rating"] > 3].copy()

    # 合并电影信息
    positive_ratings = pd.merge(positive_ratings, movies_df[["movie_id", "movie_title"]], on="movie_id", how="left")

    # 按时间戳排序
    positive_ratings = positive_ratings.sort_values(["user_id", "timestamp"])

    # 为每个用户创建历史序列
    user_histories = {}
    for user_id, group in positive_ratings.groupby("user_id"):
        if len(group) >= min_ratings:
            history = group.iloc[:]["movie_title"].tolist()
            user_histories[user_id] = {
                "history": history
            }

    return user_histories


# 创建用户历史数据
user_histories = create_user_history(ratings, movies, min_ratings=5)
print(f"处理了 {len(user_histories)} 个用户的历史数据")


# 3. GPT4Rec组件1: 查询生成器
class QueryGenerator:
    """基于GPT-2的查询生成器"""

    def __init__(self, model_name="../../models/openai-community/gpt2"):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        except Exception as e:
            print(f"无法加载GPT-2模型: {e}")

        self.prompt_template = """The user has watched the following movies:\n{history}\nBased on these, the user might want to watch:"""

        # 设置填充标记
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # 将模型设置为评估模式
        self.model.eval()

    def format_history(self, history_titles, max_titles=5):
        """格式化历史记录"""
        # 限制历史电影数量
        recent_titles = history_titles[-max_titles:] if len(history_titles) > max_titles else history_titles
        return "\n".join([f"- {title}" for title in recent_titles])

    def generate_queries(self, history_titles, num_queries=3, max_length=30):
        """使用GPT-2生成多个查询"""
        formatted_history = self.format_history(history_titles)
        prompt = self.prompt_template.format(history=formatted_history)

        # 编码输入，同时获取attention_mask
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        # 手动创建attention_mask
        attention_mask = inputs["attention_mask"]

        # 获取input_ids
        input_ids = inputs["input_ids"]

        # 生成多个查询（使用num_return_sequences）
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,  # 传入attention_mask
                    max_length=input_ids.shape[1] + max_length,
                    num_return_sequences=num_queries,
                    temperature=0.9,  # 控制多样性
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,  # 避免重复
                    top_p=0.95,  # 核采样
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            except Exception as e:
                print(f"GPT-2生成失败: {e}")

        # 解码生成的查询
        generated_queries = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # 提取提示之后的部分作为查询
            query = generated_text[len(prompt):].strip()

            # 清理查询文本
            query = re.sub(r'[^\w\s]', '', query)  # 去除非字母数字字符
            query = query.strip()

            if query and len(query.split()) > 1:  # 过滤掉太短的查询
                generated_queries.append(query)

        return generated_queries[:num_queries]


# 4. BM25检索器实现（使用rank_bm25库）
class BM25Retriever:
    """基于BM25Okapi算法的物品检索器"""

    def __init__(self, movies_df):
        """
        初始化BM25检索器

        参数:
        movies_df: 电影DataFrame，包含movie_id, movie_title, movie_tag
        """
        self.movies_df = movies_df.copy()

        # 预处理电影文本并分词
        self.movies_df["processed_text"] = self.movies_df.apply(
            lambda row: self._preprocess_and_tokenize(f"{row['movie_title']} {row['movie_tag']}"),
            axis=1
        )

        # 构建BM25索引
        self._build_bm25_index()

    def _preprocess_and_tokenize(self, text):
        """文本预处理和分词"""
        if not isinstance(text, str):
            return []

        # 分词
        tokens = word_tokenize(text)

        # 移除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        return tokens

    def _build_bm25_index(self):
        """构建BM25索引"""
        # 获取所有文档的分词列表
        corpus = self.movies_df["processed_text"].tolist()

        # 初始化BM25Okapi
        self.bm25 = BM25Okapi(corpus)

        # 创建索引到电影ID的映射
        self.idx_to_movie_id = {idx: movie_id for idx, movie_id in enumerate(self.movies_df["movie_id"])}

    def search(self, query, top_k=10, exclude_movie_ids=None):
        """
        搜索与查询最相关的电影

        参数:
        query: 查询字符串
        top_k: 返回前K个结果
        exclude_movie_ids: 排除的电影ID集合

        返回:
        包含movie_id, movie_title, score的字典列表
        """
        if exclude_movie_ids is None:
            exclude_movie_ids = set()

        # 预处理查询
        query_tokens = self._preprocess_and_tokenize(query)
        if not query_tokens:
            return []

        # 使用BM25获取文档分数
        doc_scores = self.bm25.get_scores(query_tokens)

        # 获取分数最高的文档索引
        top_indices = np.argsort(doc_scores)[::-1]

        # 收集结果
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break

            movie_id = self.idx_to_movie_id[idx]

            # 排除指定的电影
            if movie_id in exclude_movie_ids:
                continue

            movie_title = self.movies_df.iloc[idx]["movie_title"]
            score = doc_scores[idx]

            if score > 0:  # 只返回有正分数的结果
                results.append({
                    "movie_id": movie_id,
                    "movie_title": movie_title,
                    "score": score
                })

        return results

    def get_movie_by_id(self, movie_id):
        """根据电影ID获取电影信息"""
        return self.movies_df[self.movies_df["movie_id"] == movie_id].iloc[0]


# 5. GPT4Rec主框架
class GPT4Rec:
    """GPT4Rec推荐框架"""

    def __init__(self, movies_df, query_generator=None, bm25_retriever=None):
        self.movies_df = movies_df

        # 初始化组件
        self.query_generator = query_generator or QueryGenerator()
        self.bm25_retriever = bm25_retriever or BM25Retriever(movies_df)

        # 记录电影标题到ID的映射
        self.title_to_id = {}
        for _, row in movies_df.iterrows():
            self.title_to_id[row["movie_title"]] = row["movie_id"]

    def get_movie_ids_from_titles(self, titles):
        """从电影标题列表获取电影ID列表"""
        movie_ids = []
        for title in titles:
            if title in self.title_to_id:
                movie_ids.append(self.title_to_id[title])
        return movie_ids

    def recommend_for_user(self, user_history, num_queries=3, top_k_per_query=5, final_top_k=10):
        """为用户生成推荐"""
        if not user_history:
            # 如果没有历史记录，返回热门电影
            return self._get_popular_movies(final_top_k)

        # 步骤1: 生成查询
        queries = self.query_generator.generate_queries(user_history, num_queries=num_queries)
        print(f"生成的查询: {queries}")

        # 获取用户历史中的电影ID
        watched_movie_ids = self.get_movie_ids_from_titles(user_history)

        # 步骤2: 为每个查询检索物品
        all_candidates = []
        for query in queries:
            candidates = self.bm25_retriever.search(
                query,
                top_k=top_k_per_query,
                exclude_movie_ids=set(watched_movie_ids)
            )
            all_candidates.extend(candidates)

        # 去重并合并分数（使用最大分数）
        candidate_scores = defaultdict(float)
        candidate_titles = {}

        for candidate in all_candidates:
            movie_id = candidate["movie_id"]
            movie_title = candidate["movie_title"]
            score = candidate["score"]

            # 使用最大分数（也可以使用平均分数）
            if movie_id not in candidate_scores or score > candidate_scores[movie_id]:
                candidate_scores[movie_id] = score
                candidate_titles[movie_id] = movie_title

        # 转换为列表并按分数排序
        sorted_candidates = sorted(
            [(movie_id, candidate_scores[movie_id], candidate_titles[movie_id])
             for movie_id in candidate_scores],
            key=lambda x: x[1],
            reverse=True
        )

        # 返回前K个推荐
        recommendations = []
        for movie_id, score, movie_title in sorted_candidates[:final_top_k]:
            recommendations.append({
                "movie_id": movie_id,
                "movie_title": movie_title,
                "score": score
            })

        return recommendations

    def _get_popular_movies(self, top_k=10):
        """获取热门电影（基于评分数量）"""
        # 计算每部电影的评分数量
        movie_rating_counts = ratings.groupby("movie_id")["rating"].count()

        # 获取评分最多的电影
        popular_movie_ids = movie_rating_counts.sort_values(ascending=False).index[:top_k]

        recommendations = []
        for movie_id in popular_movie_ids:
            movie_info = self.bm25_retriever.get_movie_by_id(movie_id)
            recommendations.append({
                "movie_id": movie_id,
                "movie_title": movie_info["movie_title"],
                "score": 0.0  # 没有BM25分数
            })

        return recommendations


# 6. 运行GPT4Rec
def main():
    """主函数"""
    print("初始化GPT4Rec推荐系统 (GPT-2 + BM25Okapi)...")

    # 创建查询生成器
    query_generator = QueryGenerator()

    # 创建BM25检索器
    bm25_retriever = BM25Retriever(movies)

    # 创建GPT4Rec实例
    gpt4rec = GPT4Rec(
        movies,
        query_generator=query_generator,
        bm25_retriever=bm25_retriever
    )

    # 为一个特定用户生成推荐
    if user_histories:
        sample_user_id = list(user_histories.keys())[0]
        sample_history = user_histories[sample_user_id]["history"]

        print(f"\n{'=' * 60}")
        print(f"为用户 {sample_user_id} 生成推荐示例:")
        print(f"历史电影: {sample_history[-5:]}")

        recommendations = gpt4rec.recommend_for_user(sample_history, num_queries=3, final_top_k=5)

        print("\n推荐结果:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['movie_title']} (BM25分数: {rec['score']:.3f})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行失败: {e}")
