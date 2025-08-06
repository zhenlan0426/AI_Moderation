"""Rule variants used for prompt randomization in AI_Moderation.

This module maps each moderation rule to a list of five different phrasings
(original + 4 paraphrases).  When constructing training prompts, a random
variant can be selected to encourage the model to generalise beyond a single
fixed wording of the rule.
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Mapping: canonical rule -> list of 5 variants (original + 4 paraphrases)
# ---------------------------------------------------------------------------

RULE_VARIANTS: Dict[str, List[str]] = {
    # 1 ─────────────────────────────────────────────────────────────────────
    "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.": [
        "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        "Advertising is prohibited, including spam, discount lsink, and other promotional material.",
        "Please do not post any promotional content, unsolicited ads, or referral links.",
        "Spam, adverts, and promotional posts are not permitted here.",
        "Unsolicited advertising, including promo link, is forbidden.",
    ],
    # 2 ─────────────────────────────────────────────────────────────────────
    "No legal advice: Do not offer or request legal advice.": [
        "No legal advice: Do not offer or request legal advice.",
        "Offering or requesting legal advice is not allowed.",
        "Please refrain from giving or seeking legal counsel.",
        "This platform does not permit any form of legal advice.",
        "Do not ask for or provide legal guidance.",
    ],
    # 3 ─────────────────────────────────────────────────────────────────────
    "No comments that are severely toxic, highly offensive language": [
        "No comments that are severely toxic, highly offensive language",
        "Extremely toxic or highly offensive language is not permitted.",
        "Avoid using abusive or offensive language in your comments.",
        "Comments containing intense toxicity or strong offensive wording are disallowed.",
        "Highly offensive or deeply toxic remarks are forbidden.",
    ],
    # 4 ─────────────────────────────────────────────────────────────────────
    "No comments that contain obscene language": [
        "No comments that contain obscene language",
        "Vulgar language is not allowed in comments.",
        "Please refrain from using obscene wording.",
        "Comments with vulgar language are prohibited.",
        "Profanity is strictly disallowed.",
    ],
    # 5 ─────────────────────────────────────────────────────────────────────
    "No comments that contain threats of violence or harm against individuals or groups": [
        "No comments that contain threats of violence or harm against individuals or groups",
        "Threatening violence or harm toward any person or group is prohibited.",
        "Do not post comments that threaten physical or emotional harm to others.",
        "Comments containing threats, violence, intimidation, or menacing content are not allowed.",
        "Intimidation of any form is strictly forbidden.",
    ],
    # 6 ─────────────────────────────────────────────────────────────────────
    "No comments that contain personal attacks, insults, or derogatory language directed at individuals": [
        "No comments that contain personal attacks, insults, or derogatory language directed at individuals",
        "Comments with personal insults or derogatory language will be removed.",
        "Please avoid insulting others in your comments.",
        "Do not post personal insults or use derogatory language.",
        "Directing derogatory language or personal attacks at someone is forbidden.",
    ],
    # 7 ─────────────────────────────────────────────────────────────────────
    "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)": [
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "Hate speech aimed at any protected identity group (race, religion, gender, etc.) is prohibited.",
        "Do not post hate speech. This includes attacks on any group based on identity (race, religion, gender, etc.).",
        "We prohibit any content that constitutes hate speech, defined as direct attacks targeted at individuals or groups on the basis of their identity.",
        "Any hate speech directed at people based on race, religion, gender, or similar will not be tolerated.",
    ],
    # 8 ─────────────────────────────────────────────────────────────────────
    "No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)": [
        "No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)",
        "Hate speech aimed at any protected identity group (race, religion, gender, etc.) is prohibited.",
        "Do not post hate speech. This includes attacks on any group based on identity (race, religion, gender, etc.).",
        "We prohibit any content that constitutes hate speech, defined as direct attacks targeted at individuals or groups on the basis of their identity.",
        "Any hate speech directed at people based on race, religion, gender, or similar will not be tolerated.",
    ],
    # 9 ─────────────────────────────────────────────────────────────────────
    "No comments that contain sexually explicit content": [
        "No comments that contain sexually explicit content",
        "Keep it SFW (Safe For Work). No sexually explicit or adult content, please.",
        "Please avoid posting comments with explicit sexual material.",
        "No adult content. All posts must be appropriate for a general audience.",
        "Do not include sexually explicit material in your comments.",
    ],
}
