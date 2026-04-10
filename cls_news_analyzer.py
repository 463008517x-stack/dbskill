"""
财经 RSS 新闻 → DeepSeek 分析 A 股影响 → PushPlus 推送
依赖: pip install requests openai python-dotenv feedparser
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import requests
import feedparser
from openai import OpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 配置项（通过 GitHub Secrets 注入，绝对安全）
# ─────────────────────────────────────────
# ⚠️ 【代码已脱敏】为了安全上云，Token 已经改回从环境变量读取。绝对不要把真实 Token 写在这里！
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN")

# 群组编码 01 不是敏感密码，保留写死，保证最稳定的运行
PUSHPLUS_TOPIC = "01"   

# ── 大模型配置（默认 DeepSeek，切换其他模型只需改这三行）──────────
LLM_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL    = "deepseek-chat"

# 每次拉取的新闻条数
NEWS_LIMIT = 20
# 一次性发送的新闻条数上限（避免单条消息过长）
MAX_NEWS_PER_BATCH = 10


# ─────────────────────────────────────────
# 1. 通过 RSS 拉取财经新闻（多源聚合）
# ─────────────────────────────────────────

# 备用 RSS 源列表：任意一条可用即可，按优先级排列
RSS_SOURCES = [
    {
        "name": "新浪财经·滚动新闻",
        "url": "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2516&k=&num=50&page=1&r=0.3&callback=",
        "fmt": "sina_json",   # 新浪用 JSON 滚动接口，单独处理
    },
    {
        "name": "财联社·要闻 RSS",
        "url": "https://www.cls.cn/api/sw?app=rss&sv=7.7.5&os=web",
        "fmt": "rss",
    },
    {
        "name": "东方财富·财经要闻 RSS",
        "url": "https://so.eastmoney.com/news/s?keyword=A%E8%82%A1&pageindex=1&pagesize=20&type=rss",
        "fmt": "rss",
    },
    {
        "name": "金融界·财经新闻 RSS",
        "url": "https://www.jrj.com.cn/rss/news.xml",
        "fmt": "rss",
    },
]

_REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _parse_rss_source(source: dict, limit: int) -> list[dict]:
    """解析单个 RSS 源，返回标准化新闻列表。"""
    url, fmt, name = source["url"], source["fmt"], source["name"]

    try:
        resp = requests.get(url, headers=_REQ_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("[%s] 请求失败: %s", name, e)
        return []

    news_list = []

    # ── 新浪滚动 JSON 格式 ──────────────────────────
    if fmt == "sina_json":
        try:
            raw = resp.json()
            items = raw.get("result", {}).get("data", [])
            for item in items[:limit]:
                pub = item.get("ctime") or item.get("create_time", "")
                try:
                    t = datetime.fromtimestamp(int(pub)).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    t = pub[:16] if pub else ""
                news_list.append({
                    "source":  name,
                    "title":   item.get("title", "").strip(),
                    "content": item.get("intro", item.get("content", "")).strip(),
                    "time":    t,
                    "link":    item.get("url", ""),
                })
        except Exception as e:
            logger.warning("[%s] JSON 解析失败: %s", name, e)
        return news_list

    # ── 标准 RSS/Atom 格式（feedparser） ──────────────
    feed = feedparser.parse(resp.text)
    if feed.bozo and not feed.entries:
        logger.warning("[%s] feedparser 解析异常: %s", name, feed.bozo_exception)
        return []

    for entry in feed.entries[:limit]:
        tp = entry.get("published_parsed") or entry.get("updated_parsed")
        if tp:
            t = datetime(*tp[:5]).strftime("%Y-%m-%d %H:%M")
        else:
            t = ""

        content = entry.get("summary", "")
        if not content and entry.get("content"):
            content = entry["content"][0].get("value", "")

        import re
        content = re.sub(r"<[^>]+>", "", content).strip()

        news_list.append({
            "source":  name,
            "title":   entry.get("title", "").strip(),
            "content": content[:300],   
            "time":    t,
            "link":    entry.get("link", ""),
        })

    return news_list


def fetch_cls_news(limit: int = NEWS_LIMIT) -> list[dict]:
    """依次尝试各 RSS 源，返回第一个成功拉取到数据的结果。"""
    for source in RSS_SOURCES:
        logger.info("尝试 RSS 源：%s", source["name"])
        news_list = _parse_rss_source(source, limit)
        if news_list:
            logger.info("成功从 [%s] 拉取 %d 条新闻", source["name"], len(news_list))
            return news_list
        logger.warning("[%s] 未获取到数据，尝试下一个源", source["name"])

    logger.error("所有 RSS 源均不可用")
    return []


# ─────────────────────────────────────────
# 2. 用大模型分析对 A 股板块的影响
# ─────────────────────────────────────────
def analyze_news_with_llm(news_list: list[dict]) -> str:
    if not news_list:
        return "⚠️ 暂无可分析的新闻。"

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    news_text = "\n\n".join(
        f"[{n['time']}][{n.get('source', '')}] {n['title']}\n{n['content']}"
        for n in news_list[:MAX_NEWS_PER_BATCH]
    )

    system_prompt = """你是一位专业的 A 股市场分析师，擅长从财经新闻中快速提炼出对各板块的影响。
请用简洁、专业的语言输出分析，格式要求如下：

## 📰 今日要点（3句话内）
…

## 📈 利好板块
| 板块 | 相关新闻 | 理由 |
|------|---------|------|
…

## 📉 利空板块
| 板块 | 相关新闻 | 理由 |
|------|---------|------|
…

## ⚠️ 需关注的风险
…

## 💡 操作建议
…

注意：
- 只分析与 A 股市场明确相关的新闻
- 利好/利空判断须基于新闻事实，不要过度推断
- 如果新闻信息不足以判断影响，请注明"信息待观察"
"""

    user_prompt = f"请分析以下财经新闻对 A 股各板块的影响：\n\n{news_text}"

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        analysis = resp.choices[0].message.content
        logger.info("大模型分析完成（%s），共 %d 字", LLM_MODEL, len(analysis))
        return analysis

    except Exception as e:
        logger.error("大模型 API 调用失败: %s", e)
        return f"❌ AI 分析失败: {e}"


# ─────────────────────────────────────────
# 3. 发送到 PushPlus
# ─────────────────────────────────────────
def send_to_pushplus(content: str, title: str = "A 股板块影响分析") -> bool:
    if not PUSHPLUS_TOKEN or PUSHPLUS_TOKEN == "your_pushplus_token":
        logger.error("PUSHPLUS_TOKEN 未配置，请在 GitHub Secrets 中填写")
        return False

    payload = {
        "token":    PUSHPLUS_TOKEN,
        "title":    title,
        "content":  content,
        "template": "markdown",
        "channel":  "wechat",  
        "topic":    PUSHPLUS_TOPIC,   
        }

    try:
        resp = requests.post(
            "http://www.pushplus.plus/send",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") == 200:
            logger.info("PushPlus 推送成功：%s", result.get("msg", ""))
            return True
        else:
            logger.error("PushPlus 推送失败，code=%s，msg=%s", result.get("code"), result.get("msg"))
            return False

    except requests.RequestException as e:
        logger.error("PushPlus 请求异常: %s", e)
        return False


# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────
def main():
    logger.info("=== 财经 RSS 新闻 A 股分析任务开始（模型：%s）===", LLM_MODEL)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    news_list = fetch_cls_news(limit=NEWS_LIMIT)
    if not news_list:
        send_to_pushplus("⚠️ 财经新闻拉取失败，请检查网络或 RSS 源状态。", title="A 股新闻分析")
        return

    logger.info("开始大模型分析...")
    analysis = analyze_news_with_llm(news_list)

    title  = f"📊 A 股板块影响分析 · {run_time}"
    footer = f"\n\n---\n> 数据来源：财经 RSS 多源聚合 · 分析时间：{run_time} · Powered by {LLM_MODEL}"
    send_to_pushplus(analysis + footer, title=title)

    logger.info("=== 任务完成 ===")

if __name__ == "__main__":
    main()