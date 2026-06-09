import threading
import os

import anthropic


def generate_taste_comment(
    animal: dict,
    area_fruits: list[dict],
    fruit_proportions: list[float],
    match_score: float,
    callback,  # callable(str)
) -> None:
    """Call Claude Haiku in a background thread; invoke callback(text) when done."""

    def _run():
        try:
            client = anthropic.Anthropic()

            fruit_desc = "、".join(
                f"{f['name']}({int(p * 100)}%)"
                for f, p in zip(area_fruits, fruit_proportions)
            )
            score_pct = int(match_score * 100)

            prompt = (
                f"あなたはかわいいジュース屋さんのキャラクターです。"
                f"どうぶつの「{animal['name']}」は「{animal['pref']}」が好きです。"
                f"今回作ったジュースは {fruit_desc} でできています。"
                f"マッチスコアは{score_pct}点です。"
                f"このジュースの味の感想を、子供向けに「{animal['name']}」のセリフとして"
                f"かわいく2〜3文の日本語で書いてください。"
                f"文末は「！」や「♪」などで明るく締めてください。"
            )

            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            text = next(
                (b.text for b in response.content if b.type == "text"),
                "おいしいジュースができたよ！",
            )
            callback(text)
        except Exception as e:
            print(f"[ai_comment] error: {e}")
            callback("おいしいジュースができたよ！")

    threading.Thread(target=_run, daemon=True).start()
