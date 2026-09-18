#!/usr/bin/env bash
# start_tts_cosyvoice2.sh —— 用 CosyVoice2 + 男女兩個 zero-shot 語者起 tts_service。
#
# 放成腳本而不是一行指令，是因為男聲的逐字稿含中文且必須跟音檔內容逐字一致，
# 塞進 shell 的引號層層轉義很容易在不知不覺中被改掉（zero-shot 靠「音檔 +
# 對應文字」抽音色，文字對不上抽出來就會偏）。
#
# 男聲參考音檔怎麼來的見 scripts/make_male_prompt.py。拿到真人錄音之後，
# 換掉 MALE_WAV 指到的檔案與下面的逐字稿即可。
#
# 用法（在 wsl -d Ubuntu-24.04 -u root 裡）：
#   bash /mnt/c/imood_project/imood-voice/scripts/start_tts_cosyvoice2.sh

set -euo pipefail

REPO=/mnt/c/imood_project/imood-voice
PY=/root/miniconda3/envs/cosyvoice_vllm/bin/python
MALE_WAV="$REPO/assets/male_prompt.wav"

export COSYVOICE_REPO=/root/CosyVoice
export COSYVOICE_MODEL_DIR=/root/CosyVoice/pretrained_models/CosyVoice2-0.5B
export MODELSCOPE_OFFLINE=1

if [ -f "$MALE_WAV" ]; then
    export TTS_PROMPT_WAV_MALE="$MALE_WAV"
    export TTS_PROMPT_TEXT_MALE="大家好，我是 imood 的语音助理。今天天气不错，希望你有个愉快的一天，有什么需要帮忙的地方都可以告诉我。"
else
    echo "[start] 找不到 $MALE_WAV，只會註冊女聲（voice=male 會退回女聲）" >&2
fi

cd "$REPO"
exec "$PY" -m uvicorn tts_service:app --host 0.0.0.0 --port 8001
