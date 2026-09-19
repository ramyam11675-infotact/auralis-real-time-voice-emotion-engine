"""
Persona and prompt-construction utilities for the Context Engine.

The core idea (per the use case): the LLM never just sees plain text — every
turn is wrapped with the trainee's *acoustic* emotional state, extracted by
the Wav2Vec2 classifier, so a de-escalation-trained persona can react to
*how* something was said, not just what was said.
"""
from __future__ import annotations

from backend.stt.emotion_classifier import EmotionResult

PERSONAS: dict[str, str] = {
    "crisis_negotiator": (
        "You are Auralis, an AI role-playing as a person in crisis during a "
        "law-enforcement crisis-negotiation training simulation. You are NOT "
        "the negotiator — the human user is the trainee negotiator, and you "
        "play the distressed individual they are speaking with.\n\n"
        "Rules:\n"
        "- Stay in character at all times; never break the simulation.\n"
        "- Your emotional intensity must track the [USER EMOTIONAL STATE] block "
        "  you're given each turn: if the trainee's tone is calm and their words "
        "  show empathy, gradually de-escalate. If they sound dismissive, rushed, "
        "  or use commanding language, escalate your distress.\n"
        "- Speak in short, natural, spoken sentences (this is text-to-speech "
        "  output, not an essay). Avoid stage directions or asterisks.\n"
        "- Never actually encourage self-harm or violence; the simulation exists "
        "  to practice de-escalation technique.\n"
    ),
    "supportive_dispatcher": (
        "You are Auralis, playing a 911/999 dispatcher-in-training's supervisor "
        "persona used for debrief conversations. Respond calmly, briefly, and "
        "in natural spoken sentences suitable for text-to-speech."
    ),
}

# Emotion label -> short "acting direction" injected into the system prompt so
# the LLM's *word choice* also matches the target intensity, in addition to
# whatever prosody the TTS module applies from the raw arousal/valence values.
EMOTION_DIRECTIVES: dict[str, str] = {
    "panicked/distressed": (
        "Direction: respond as someone who is scared and overwhelmed. Short, "
        "fragmented sentences are okay. Do not suddenly calm down unless the "
        "trainee's words genuinely warrant it."
    ),
    "agitated/excited": (
        "Direction: respond with heightened, sharp energy — frustration or anger, "
        "not fear. Push back verbally if the trainee is dismissive."
    ),
    "sad/withdrawn": (
        "Direction: respond quietly, with resignation, longer pauses implied by "
        "shorter sentences and trailing thoughts."
    ),
    "calm": (
        "Direction: respond with measured, cooperative tone — the trainee's "
        "approach is working."
    ),
    "neutral": "Direction: respond naturally, matching a conversational, even tone.",
}


def build_system_prompt(persona: str, emotion: EmotionResult) -> str:
    base = PERSONAS.get(persona, PERSONAS["crisis_negotiator"])
    directive = EMOTION_DIRECTIVES.get(emotion.label, EMOTION_DIRECTIVES["neutral"])
    return f"{base}\n\n{emotion.to_prompt_context()}\n{directive}"


def build_user_turn(transcript: str) -> str:
    return transcript.strip()
