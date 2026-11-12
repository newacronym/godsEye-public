"""Claude as query planner and result verifier.

The planner splits a question into conditions: every visual attribute the answer must have, each tagged
with the scale it is visible at (the whole person or object, or a close-up of the head, torso, hands or
feet), plus the activity if there is one. The verifier answers each condition with yes, maybe, no or not
visible, and the verdict follows from those: all yes is a match, any no is a rejection, and the rest is
possible when something points to it. Asking per condition is what stops "a red top and a grey backpack" passing for "a red
backpack", or a red shirt passing for a red Nike shirt.

Haiku 5.5 plans and checks activities. Sonnet 5.5 checks appearance: binding several attributes to the
right thing, and details at the limit of the pixels, are where it is consistent and Haiku is not
.

Calls run with thinking off and low effort: they are short, structured judgments on the latency path.
A refusal or a transient API error degrades gracefully (no plan, or "unverified"); configuration errors
such as a bad key or a malformed request raise, so they can't pass for "nothing matched".
"""
import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-haiku-5-5"
APPEARANCE_MODEL = "claude-sonnet-5-5"
MODEL_NAMES = {MODEL: "Haiku 5.5", APPEARANCE_MODEL: "Sonnet 5.5"}
FAST = {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
# Newer models turn thinking off with "between_tools" instead of "disabled".
NO_THINKING = {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}
TRANSIENT = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)

Scale = Literal["whole", "head", "torso", "hand", "legs", "feet"]


class Attribute(BaseModel):
    phrase: str = Field(description="One visual condition as it looks at its scale, e.g. 'a man', 'a red t-shirt', "
                                    "'a Nike logo on a shirt', 'a hand with a tattoo', 'white sneakers'.")
    scale: Scale = Field(
        description="Where it is visible. whole = on the whole body or object: clothing type and color, bags "
                    "carried, build, sex or age if stated, vehicle type and color. head = the face and head up "
                    "close: glasses, hats, masks, beards, earrings, hairstyle, headphones. torso = the front or "
                    "back of the upper body up close: logos, brand marks, text or prints on clothing, lanyards, "
                    "badges. hand = hands, wrists and forearms: tattoos, rings, watches, bracelets, gloves, small "
                    "things in the fingers. legs = pants, shorts, skirts, leggings up close. feet = shoes. Use whole for "
                    "anything a whole-body view shows.")
    side: Literal["left", "right", "either"] = Field(
        description="For hand: which of the person's own hands the question names; either otherwise.")


class Plan(BaseModel):
    subject: Literal["person", "vehicle", "bag", "other"] = Field(
        description="Who or what the question is about (for an activity, whoever does it): person; vehicle (car, "
                    "truck, bus, motorcycle, bicycle); bag (a backpack, handbag or suitcase by itself; 'a person "
                    "with a red backpack' is a person); other (anything else, or a place in the scene).")
    object_phrase: str = Field(description="The whole description as one short visual phrase, e.g. 'a man in a red "
                                           "Nike t-shirt', 'a person carrying a large box'.")
    attributes: list[Attribute] = Field(
        description="Every visual condition on how the answer looks, one thing each, at most 5. Split compound "
                    "descriptions: 'a man in a red Nike t-shirt' is 'a man' (whole), 'a red t-shirt' (whole) and "
                    "'a Nike logo on a shirt' (torso). Don't repeat a bare subject ('a person', 'someone', 'a "
                    "vehicle'), but keep words that narrow it ('a man', 'a kid', 'a pickup truck'). A thing asked "
                    "about by itself is one condition ('an umbrella'). Attributes describe one person or object: "
                    "for several people ('two people talking'), the others belong in the activity ('talking with "
                    "another person'). Empty if only an activity is asked about.")
    activity: str = Field(description="The ongoing activity or brief action asked about, as a short phrase "
                                      "('texting on a phone', 'opening a car door'); empty if none.")
    intent: Literal["appearance", "state", "action", "count"] = Field(
        description="appearance = what someone or something looks like; state = an ongoing activity visible at any "
                    "moment (texting, talking on a phone, sitting, reading, two people talking, waiting); action = "
                    "a brief change that unfolds over a few seconds (picking up, putting down, opening a door, "
                    "handing over, getting into a car, a car stopping or turning); count = the user wants a number.")
    cameras: list[str] = Field(description="Camera ids the user restricted the search to; empty if none.")
    time_from: str | None = Field(description="HH:MM start of the requested window, or null.")
    time_to: str | None = Field(description="HH:MM end of the requested window, or null.")

    def useful_attributes(self) -> list[Attribute]:
        """Attributes minus a bare restatement of the subject ('a person'), which narrows nothing."""
        return [a for a in self.attributes if a.phrase.strip().lower().rstrip(".") not in BARE_SUBJECTS]

    def conditions(self) -> list[str]:
        return [a.phrase for a in self.useful_attributes()] + ([self.activity] if self.activity.strip() else [])

    def key_conditions(self) -> list[bool]:
        """Per condition: is it the specific thing asked (a close-up detail or the activity)?"""
        return [a.scale != "whole" for a in self.useful_attributes()] + ([True] if self.activity.strip() else [])


BARE_SUBJECTS = {"person", "a person", "someone", "somebody", "anyone", "a human", "people", "a vehicle", "vehicle",
                 "an object", "a thing", "an item", "a bag", "two people", "a group of people", "several people",
                 "a couple", "a pair of people"}


PLANNER_SYSTEM = """You turn a security operator's question about CCTV footage into a search plan.
Footage: {footage}. Cameras: {cameras}.
Split the description into attributes, one visual condition each, tagged with the scale it is visible at.
Put the activity, if any, in activity, not in attributes. Only restrict cameras or times when the question
explicitly asks for it. Keep phrases visual and concrete; leave out left and right (side holds that), and
words about cameras, times and counting."""


class Check(BaseModel):
    condition: str
    answer: Literal["yes", "maybe", "no", "not_visible"]


class Judgment(BaseModel):
    checks: list[Check] = Field(description="Every condition, in the order given.")
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(description="At most 20 words, about what is visible.")


@dataclass
class Verdict:
    match: bool
    unsure: bool          # possible: nothing contradicts the question, but something couldn't be seen
    confidence: float
    reason: str
    checks: list[dict] = field(default_factory=list)


def decide(conditions: list[str], j: Judgment, key: list[bool] | None = None) -> Verdict:
    """All yes: a match. Any no: rejected. Otherwise possible when something points to it: every key
    condition (the specific thing asked: a close-up detail, the activity, the person in the photo) is
    yes or maybe, or, with no key conditions, any condition is. A hoodie whose PUMA lettering can't be
    seen is not a possible PUMA hoodie."""
    answers = [c.answer for c in j.checks][:len(conditions)]
    answers += ["not_visible"] * (len(conditions) - len(answers))     # an unanswered condition is unseen
    checks = [{"condition": c, "answer": a} for c, a in zip(conditions, answers)]
    if "no" in answers:
        return Verdict(False, False, j.confidence, j.reason, checks)
    if all(a == "yes" for a in answers):
        return Verdict(True, False, j.confidence, j.reason, checks)
    positive = [a in ("yes", "maybe") for a in answers]
    keyed = [pos for pos, k in zip(positive, key or []) if k]
    return Verdict(False, all(keyed) if keyed else any(positive), j.confidence, j.reason, checks)


VERIFIER_SYSTEM = """You check one candidate from a CCTV search against an operator's question. You get the
conditions to check and images of the candidate: {evidence}. The candidate is outlined in red where it
was tracked.

Answer every condition, in order:
- yes: the most likely reading of the images is that it holds. Low resolution is normal; judge what is most
  likely, not what is certain: if it looks like the condition and nothing else explains it better, that is
  yes, even when blurry.
- maybe: something suggests it, but a specific different reading is about as likely; name that reading
  in the reason.
- no: the images show the relevant thing clearly enough, and it does not hold.
- not_visible: the relevant thing can't be seen at all: out of view, facing away, too small or too dark.
Bind each condition to the right thing: "a red backpack" needs the backpack itself to be red (a red top with a
grey backpack is no). A brand or text needs the logo or lettering itself to be recognizable; a color alone
is not a brand. Conditions are about the outlined candidate, not anyone else in the frame.
{extra}"""

EVIDENCE = {
    "appearance": "close-up crops of the candidate, a zoomed full-resolution frame, close-ups of any body part a "
                  "condition is about (cut at full resolution; hand close-ups run from the elbow to the fingertips "
                  "and carry the person's own LEFT or RIGHT side from pose estimation, which can be wrong when the "
                  "person faces away), and the whole scene",
    "state": "zoomed frames of the candidate about a second apart, close-ups of any body part a condition is "
             "about, and the whole scene",
    "action": "zoomed frames 0.5 s apart in time order, close-ups of any body part a condition is about, and the "
              "whole scene",
}
EXTRA = {
    "appearance": """For a hand, the wrist and forearm count. A tattoo shows as dark ink that stays on the skin
across views; a watch or bracelet as a band at the wrist. Facing the camera, a person's right hand is on the
image's left; if a condition names a side and the detail is only on the other hand, answer no.""",
    "state": "An ongoing activity needs no motion: it holds if the candidate does it in any frame.",
    "action": """An action holds if the candidate performs it at any point in these frames, including starting or
finishing it. Judge from the change between frames: an object leaves the floor and ends up in the hand, a
door swings open, a car slows to a halt. Being near an object, or only being in the end state, is not the
action.""",
}


REFERENCE = """The first image is a reference photo of the person to find, taken by someone else, possibly with a
phone. "The same person as in the reference photo" is yes when everything visible agrees: garments and their
colors, patterns, prints and logos, shoes, bags and accessories, hair, and build. Lighting and camera shift
colors somewhat, and a jacket can be open or closed; a different garment, shoe or hair color is no. Judge
appearance only: don't try to recognize who anyone is."""


# A key file on the Studio takes precedence over Lightning's injected env var, so the key can be
# rotated without restarting the Studio (Lightning only injects secrets at start).
KEY_FILE = Path.home() / "data/secrets/anthropic_api_key"


def api_key() -> str | None:
    return KEY_FILE.read_text().strip() if KEY_FILE.exists() else os.environ.get("ANTHROPIC_API_KEY")


def _images(images: list[tuple[str, bytes]]) -> list[dict]:
    content = []
    for caption, jpeg in images:
        content.append({"type": "text", "text": caption})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(jpeg).decode()}})
    return content


class Claude:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=api_key(), max_retries=5, timeout=45.0)

    def _parse(self, model: str, **kw):
        return self.client.messages.parse(model=model, **(FAST if model == MODEL else NO_THINKING), **kw)

    def plan(self, question: str, footage: str, cameras: list[str]) -> Plan | None:
        try:
            r = self._parse(MODEL, max_tokens=768, messages=[{"role": "user", "content": question}],
                            system=PLANNER_SYSTEM.format(footage=footage, cameras=", ".join(cameras)),
                            output_format=Plan)
        except TRANSIENT:
            return None
        return None if r.stop_reason == "refusal" else r.parsed_output

    def verify(self, question: str, conditions: list[str], images: list[tuple[str, bytes]],
               intent: str = "appearance", model: str = MODEL, reference: bool = False,
               key: list[bool] | None = None) -> Verdict | None:
        """images: (caption, jpeg bytes) in display order; intent picks the evidence description;
        reference: the first image is a photo of the person to find; key: which conditions a
        possible match needs some sign of (decide())."""
        kind = intent if intent in EVIDENCE else "appearance"
        extra = EXTRA[kind] + ("\n" + REFERENCE if reference else "")
        listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(conditions))
        content = _images(images) + [{"type": "text", "text": f"Question: {question}\nConditions, in order:\n{listed}"}]
        try:
            r = self._parse(model, max_tokens=512, messages=[{"role": "user", "content": content}],
                            system=VERIFIER_SYSTEM.format(evidence=EVIDENCE[kind], extra=extra),
                            output_format=Judgment)
        except TRANSIENT:
            return None
        if r.stop_reason == "refusal" or r.parsed_output is None:
            return None
        return decide(conditions, r.parsed_output, key)
