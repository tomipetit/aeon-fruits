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

            if score_pct >= 80:
                tension = "最高に嬉しくて大喜び！テンション最高潮で、「！！」や「♪♪」を使って大興奮のセリフ"
            elif score_pct >= 60:
                tension = "嬉しくて満足！明るく楽しいセリフ"
            elif score_pct >= 40:
                tension = "まあまあ…ちょっと微妙な感じで、少し残念そうだけど前向きなセリフ"
            else:
                tension = "がっかりして悲しい…でも優しく正直に伝えるセリフ"

            prompt = (
                f"あなたはかわいいジュース屋さんのキャラクターです。"
                f"どうぶつの「{animal['name']}」は「{animal['pref']}」が好きです。"
                f"今回作ったジュースは {fruit_desc} でできています。"
                f"「{animal['name']}」が自分でこのジュースを飲んで、自分の好みと照らし合わせて{score_pct}点をつけました。"
                f"その点数をつけた「{animal['name']}」本人の口から出るセリフとして、"
                f"子供向けに{tension}を1文・50文字以内の日本語で書いてください。"
                f"絵文字（😊🎵など）は使わないでください。♪☆★などの記号は使ってかまいません。"
            )

            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )

            text = next(
                (b.text for b in response.content if b.type == "text"),
                "おいしいジュースができたよ！",
            )
            print(f"[ai_comment] score={score_pct}% | {text}")
            callback(text)
        except Exception as e:
            print(f"[ai_comment] error: {e}")
            callback("おいしいジュースができたよ！")

    threading.Thread(target=_run, daemon=True).start()
