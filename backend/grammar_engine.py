"""
LectureCast ISL Grammar Engine (Teacher -> Student)
====================================================
Converts spoken Subject-Verb-Object (S-V-O) syntax (and multilingual English/Telugu/Hindi
classroom speech) into Indian Sign Language (ISL) Subject-Object-Verb (S-O-V) glosses
with spatial indexing markers (IX-loc), Non-Manual Markers (NMM), and 3D Avatar Bone
Rotation Keyframes.

Incorporates linguistic rules from NIOS Secondary Course 230 (Indian Sign Language):
- Lesson 8: Manual (Handshape, Place of Articulation, Movement) & Non-Manual Components,
  Minimal Pairs, and Intensifier Fusion (VERY/EXTREMELY -> stronger movement + facial NMM).
- Lesson 9: Word-Level Structures (Kinship compounds, Composite WH-signs, Numeral/Time
  incorporation, Reduplication ++, Directional verbs, and Fingerspelled compounds).
- Lesson 10: Four Sentence Types (Simple, Order/Command with clause-final MUST,
  Interrogatives with clause-final WH/HAVE/FINISH, and Negation with clause-final NOT).
- Lesson 11: Body-Location Sign Families (Head/Thinking, Chest/Feeling, Mouth/Communication,
  Money) and Neologism Compound Processes (WATER+SUN+RISE = EVAPORATION, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    spacy = None
    Doc = None


# Multilingual STEM & Classroom Lexicon Normalization (Telugu & Hindi -> English Canonical Glosses)
MULTILINGUAL_LEXICON: Dict[str, Dict[str, str]] = {
    "te": {
        "ఉపాధ్యాయుడు": "teacher",
        "విద్యార్థి": "student",
        "విద్యార్థులు": "students",
        "అల్గారిథమ్": "algorithm",
        "డెరివేటివ్": "derivative",
        "అవకలనం": "derivative",
        "కిరణజన్య సంయోగక్రియ": "photosynthesis",
        "వేగం": "velocity",
        "ద్రవ్యవేగం": "momentum",
        "సమీకరణం": "equation",
        "వివరిస్తారు": "explain",
        "నేర్చుకుంటారు": "learn",
        "లెక్కించండి": "calculate",
        "ఈరోజు": "today",
        "ఇప్పుడు": "now",
        "ఎందుకు": "why",
        "ఎలా": "how",
        "ఏమిటి": "what",
        "ఎక్కడ": "where",
        "ఎవరు": "who",
        "ఎప్పుడు": "when",
        "బాష్పీభవనం": "evaporation",
        "వరద": "flood",
    },
    "hi": {
        "शिक्षक": "teacher",
        "अध्यापक": "teacher",
        "छात्र": "student",
        "विद्यार्थी": "student",
        "एल्गोरिदम": "algorithm",
        "अवकलन": "derivative",
        "प्रकाश संश्लेषण": "photosynthesis",
        "वेग": "velocity",
        "संवेग": "momentum",
        "समीकरण": "equation",
        "समझाते": "explain",
        "सीखते": "learn",
        "गणना": "calculate",
        "आज": "today",
        "अभी": "now",
        "क्यों": "why",
        "कैसे": "how",
        "क्या": "what",
        "कहाँ": "where",
        "कौन": "who",
        "कब": "when",
        "वाष्पीकरण": "evaporation",
        "बाढ़": "flood",
    },
}

# NIOS Course 230 (Lessons 9 & 11) Word-Level Morphological Breakdowns
NIOS_230_MORPHOLOGY: Dict[str, Dict[str, str]] = {
    # Lesson 9.3: Sign Families (Kinship Compounds)
    "WIFE": {"compound": "WOMAN + MARRY", "category": "Sign Family (Kinship - L9.3)"},
    "HUSBAND": {"compound": "MALE + MARRY", "category": "Sign Family (Kinship - L9.3)"},
    "SISTER": {"compound": "FEMALE + SIBLING", "category": "Sign Family (Kinship - L9.3)"},
    "BROTHER": {"compound": "MALE + SIBLING", "category": "Sign Family (Kinship - L9.3)"},
    "DAUGHTER": {"compound": "FEMALE + BORN", "category": "Sign Family (Kinship - L9.3)"},
    "SON": {"compound": "MALE + BORN", "category": "Sign Family (Kinship - L9.3)"},
    # Lesson 9.4: Composite Question Signs
    "WHO": {"compound": "FACE + QUESTION", "category": "Question Compound (Clause-Final - L9.4)"},
    "WHERE": {"compound": "PLACE + QUESTION", "category": "Question Compound (Clause-Final - L9.4)"},
    "WHEN": {"compound": "TIME/DAY + QUESTION", "category": "Question Compound (Clause-Final - L9.4)"},
    # Lesson 9.6: Reduplication / Repetition (++)
    "CHILDREN": {"compound": "CHILD++", "category": "Reduplication Plural (L9.6)"},
    "STUDENTS": {"compound": "CLASS + CHILD++", "category": "Reduplication Plural (L9.6)"},
    "FRIDAY": {"compound": "F++ (Repeated)", "category": "Letter Reduplication (L9.1/L9.6)"},
    # Lesson 9.7: Fingerspelling + Sign Compounds
    "APRIL": {"compound": "A + HOT", "category": "Fingerspelled Time Compound (L9.7)"},
    "DECEMBER": {"compound": "D + COLD", "category": "Fingerspelled Time Compound (L9.7)"},
    "SUNDAY": {"compound": "S + CLOSED", "category": "Fingerspelled Time Compound (L9.7)"},
    "PRINCIPAL": {"compound": "P + TOP", "category": "Fingerspelled Title Compound (L9.7)"},
    "TRAINING": {"compound": "T + TEACH", "category": "Fingerspelled Action Compound (L9.7)"},
    "LIBRARY": {"compound": "L + BOOK", "category": "Fingerspelled Place Compound (L9.7)"},
    "MINISTER": {"compound": "M + BOSS", "category": "Fingerspelled Title Compound (L9.7)"},
    "DIRECTOR": {"compound": "D + BOSS", "category": "Fingerspelled Title Compound (L9.7)"},
    # Lesson 11.3: Creating New Signs (Meaning Explanation & General+Specific)
    "FLOOD": {"compound": "WATER + RISE-UP", "category": "Meaning Explanation Compound (L11.3)"},
    "EVAPORATION": {"compound": "WATER + SUN + RISE", "category": "STEM Meaning Compound (L11.3)"},
    "COUNTRIES": {"compound": "WORLD + PLACES", "category": "General + Specific Compound (L11.3)"},
    "STATES": {"compound": "INDIA + PLACES", "category": "General + Specific Compound (L11.3)"},
    "COURT": {"compound": "LAW + PLACE", "category": "General + Specific Compound (L11.3)"},
}

# NIOS Course 230 (Lesson 11.2) Body-Location Sign Families
NIOS_230_SIGN_FAMILIES: Dict[str, str] = {
    "FORGET": "Thinking Family (Head Level - L11.2)",
    "THINK": "Thinking Family (Head Level - L11.2)",
    "REMEMBER": "Thinking Family (Head Level - L11.2)",
    "WORRY": "Thinking Family (Head Level - L11.2)",
    "LEARN": "Thinking Family (Head Level - L11.2)",
    "UNDERSTAND": "Thinking Family (Head Level - L11.2)",
    "ALGORITHM": "STEM Cognitive Family (Head/Upper Torso)",
    "DERIVATIVE": "STEM Calculus Family (Upper Signing Space)",
    "PHOTOSYNTHESIS": "STEM Biology Family (WATER+SUN+LEAF)",
    "HAPPY": "Feeling Family (Chest Level - L11.2)",
    "SAD": "Feeling Family (Chest Level - L11.2)",
    "ENJOY": "Feeling Family (Chest Level - L11.2)",
    "SHOCK": "Feeling Family (Chest Level - L11.2)",
    "INFORM": "Communication Family (Mouth Level - L11.2)",
    "TELL": "Communication Family (Mouth Level - L11.2)",
    "COMPLAIN": "Communication Family (Mouth Level - L11.2)",
    "SING": "Communication Family (Mouth Level - L11.2)",
    "EXPLAIN": "Communication Family (Mouth/Chest Level - L8.2/L11.2)",
    "TEACH": "Directional Verb (Towards Learner - L9.6)",
    "HELP": "Directional Verb (Towards Recipient - L9.6)",
    "RICH": "Money Family (Hand/Palm Level - L11.2)",
    "PAY": "Money Family (Hand/Palm Level - L11.2)",
    "EXPENSIVE": "Money Family (Hand/Palm Level - L11.2)",
    "CHEAP": "Money Family (Hand/Palm Level - L11.2)",
}

# Stopwords omitted in canonical Indian Sign Language (articles, auxiliaries, copulas)
# Note: "have"/"has" as main possessive verb is preserved at the end of the sentence per NIOS L10.1
ISL_OMITTED_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "shall", "to", "of", "by",
}

TEMPORAL_MARKERS = {
    "today", "tomorrow", "yesterday", "now", "later", "morning", "evening", "tonight",
    "always", "never", "first", "then", "next", "finally", "friday", "sunday", "monday",
    "april", "december",
}

WH_QUESTIONS = {"what", "why", "how", "where", "when", "who", "which"}
NEGATION_MARKERS = {"not", "no", "never", "neither", "none", "nothing", "n't", "dont", "don't"}
COMMAND_MODALS = {"must", "should"}
INTENSIFIERS = {"very", "extremely"}

# Pre-authored 3D Rigged Avatar Bone Keyframes (Euler XYZ in radians) for ISL Signs
ISL_SIGN_KEYFRAMES: Dict[str, List[Dict]] = {
    "ALGORITHM": [
        {
            "duration": 0.45,
            "bones": {
                "RightArm": {"x": -0.65, "y": 0.30, "z": -0.40},
                "RightForeArm": {"x": 0.0, "y": -1.10, "z": 0.35},
                "RightHand": {"x": 0.25, "y": 0.40, "z": 0.0},
                "LeftArm": {"x": -0.55, "y": -0.25, "z": 0.40},
                "LeftForeArm": {"x": 0.0, "y": 1.05, "z": -0.30},
                "Head": {"x": 0.05, "y": 0.10, "z": 0.0},
            },
            "nmm": {"eyebrowRaise": 0.2, "mouthOpen": 0.1},
        },
        {
            "duration": 0.45,
            "bones": {
                "RightArm": {"x": -0.85, "y": 0.15, "z": -0.25},
                "RightForeArm": {"x": 0.20, "y": -0.75, "z": 0.65},
                "RightHand": {"x": -0.35, "y": -0.30, "z": 0.25},
                "LeftArm": {"x": -0.75, "y": -0.15, "z": 0.25},
                "LeftForeArm": {"x": 0.15, "y": 0.80, "z": -0.60},
                "Head": {"x": 0.0, "y": -0.08, "z": 0.0},
            },
            "nmm": {"eyebrowRaise": 0.35, "mouthOpen": 0.15},
        },
    ],
    "DERIVATIVE": [
        {
            "duration": 0.45,
            "bones": {
                "RightArm": {"x": -0.95, "y": 0.45, "z": -0.30},
                "RightForeArm": {"x": 0.10, "y": -1.20, "z": 0.50},
                "RightHand": {"x": 0.0, "y": 0.35, "z": -0.45},
                "LeftArm": {"x": -0.60, "y": -0.40, "z": 0.50},
                "LeftForeArm": {"x": 0.0, "y": 1.25, "z": -0.10},
                "LeftHand": {"x": 0.0, "y": 0.0, "z": 0.0},
                "Head": {"x": 0.12, "y": 0.15, "z": -0.05},
            },
            "nmm": {"eyebrowRaise": 0.4, "mouthOpen": 0.05},
        },
        {
            "duration": 0.45,
            "bones": {
                "RightArm": {"x": -0.50, "y": -0.15, "z": -0.55},
                "RightForeArm": {"x": 0.0, "y": -0.60, "z": 0.15},
                "RightHand": {"x": 0.30, "y": 0.55, "z": 0.20},
                "LeftArm": {"x": -0.60, "y": -0.40, "z": 0.50},
                "LeftForeArm": {"x": 0.0, "y": 1.25, "z": -0.10},
                "Head": {"x": 0.05, "y": -0.10, "z": 0.0},
            },
            "nmm": {"eyebrowRaise": 0.25, "mouthOpen": 0.1},
        },
    ],
    "PHOTOSYNTHESIS": [
        {
            "duration": 0.50,
            "bones": {
                "RightArm": {"x": -1.35, "y": 0.35, "z": -0.20},
                "RightForeArm": {"x": 0.0, "y": -0.55, "z": 0.30},
                "RightHand": {"x": 0.45, "y": 0.20, "z": -0.35},
                "LeftArm": {"x": -0.55, "y": -0.35, "z": 0.45},
                "LeftForeArm": {"x": 0.0, "y": 1.15, "z": -0.20},
                "Head": {"x": -0.18, "y": 0.12, "z": 0.0},
            },
            "nmm": {"eyebrowRaise": 0.5, "mouthOpen": 0.2},
        },
        {
            "duration": 0.50,
            "bones": {
                "RightArm": {"x": -0.75, "y": 0.10, "z": -0.45},
                "RightForeArm": {"x": 0.20, "y": -1.05, "z": 0.40},
                "RightHand": {"x": -0.20, "y": 0.0, "z": 0.15},
                "LeftArm": {"x": -0.85, "y": -0.15, "z": 0.30},
                "LeftForeArm": {"x": 0.25, "y": 0.95, "z": -0.45},
                "Head": {"x": 0.05, "y": 0.0, "z": 0.0},
            },
            "nmm": {"eyebrowRaise": 0.3, "mouthOpen": 0.1},
        },
    ],
    "IX-LEFT": [
        {
            "duration": 0.35,
            "bones": {
                "RightArm": {"x": -0.55, "y": 0.65, "z": -0.25},
                "RightForeArm": {"x": 0.0, "y": -0.45, "z": 0.15},
                "RightHand": {"x": 0.0, "y": 0.30, "z": 0.0},
                "Head": {"x": 0.0, "y": 0.30, "z": 0.0},
            },
            "nmm": {"eyebrowRaise": 0.2, "mouthOpen": 0.0},
        }
    ],
    "IX-RIGHT": [
        {
            "duration": 0.35,
            "bones": {
                "RightArm": {"x": -0.55, "y": -0.55, "z": -0.45},
                "RightForeArm": {"x": 0.0, "y": -0.40, "z": 0.15},
                "RightHand": {"x": 0.0, "y": -0.30, "z": 0.0},
                "Head": {"x": 0.0, "y": -0.30, "z": 0.0},
            },
            "nmm": {"eyebrowRaise": 0.2, "mouthOpen": 0.0},
        }
    ],
}


@dataclass
class SyntaxTokenInfo:
    text: str
    lemma: str
    pos: str
    dep: str
    isl_role: str


@dataclass
class ISLParsedOutput:
    original_text: str
    language: str
    normalized_english: str
    svo_tokens: List[Dict]
    isl_sov_gloss: List[str]
    isl_gloss_string: str
    spatial_markers: List[Dict]
    non_manual_markers: List[str]
    avatar_keyframes: List[Dict]
    parser_backend: str
    sentence_type: str = "Simple Sentence (NIOS 230 L10.2)"
    word_level_structures: List[Dict] = field(default_factory=list)
    sign_families: List[Dict] = field(default_factory=list)


class ISLGrammarEngine:
    """
    Converts spoken S-V-O syntax into Indian Sign Language (ISL) S-O-V structure
    with spatial indexing, NIOS 230 word-level morphology, and 3D avatar bone keyframes.
    """

    def __init__(self) -> None:
        self.nlp = None
        self.parser_name = "nios230-isl-grammar-engine"
        if spacy is not None:
            for model_name in ("en_core_web_trf", "en_core_web_sm"):
                try:
                    self.nlp = spacy.load(model_name)
                    self.parser_name = f"spacy:{model_name}+nios230"
                    break
                except Exception:
                    continue

    def normalize_multilingual(self, text: str, language: str = "en") -> str:
        """Translates Telugu/Hindi classroom & STEM tokens into canonical English prior to dependency parsing."""
        lang_prefix = (language or "en").split("-")[0].lower()
        cleaned = text.strip()
        if lang_prefix in MULTILINGUAL_LEXICON:
            lex = MULTILINGUAL_LEXICON[lang_prefix]
            for native_phrase, eng_equiv in sorted(lex.items(), key=lambda x: len(x[0]), reverse=True):
                cleaned = cleaned.replace(native_phrase, f" {eng_equiv} ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _synthesize_procedural_keyframes(
        self, gloss: str, index: int, is_question: bool, is_negated: bool, is_intensified: bool = False
    ) -> List[Dict]:
        """Generates deterministic, fluid 3D skeletal keyframes for any gloss token."""
        upper = gloss.upper()
        if upper in ISL_SIGN_KEYFRAMES:
            return [
                {**frame, "gloss": upper}
                for frame in ISL_SIGN_KEYFRAMES[upper]
            ]

        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(upper))
        amp = 1.25 if is_intensified else 1.0
        phase_a = (((seed % 60) - 30) / 100.0) * amp
        phase_b = ((((seed // 7) % 60) - 30) / 100.0) * amp
        eyebrow = 0.65 if is_question else (0.45 if is_intensified else (0.15 if not is_negated else -0.25))
        head_z = 0.18 if is_question else 0.0
        head_y = 0.22 if (index % 2 == 0) else -0.22

        return [
            {
                "gloss": upper,
                "duration": 0.42,
                "bones": {
                    "RightArm": {"x": -0.75 + phase_a, "y": 0.25 + phase_b, "z": -0.35},
                    "RightForeArm": {"x": 0.15, "y": -0.95 + phase_a, "z": 0.40},
                    "RightHand": {"x": phase_b, "y": 0.25, "z": -0.15},
                    "LeftArm": {"x": -0.60 - phase_b * 0.5, "y": -0.20, "z": 0.35},
                    "LeftForeArm": {"x": 0.10, "y": 0.85 - phase_a, "z": -0.30},
                    "LeftHand": {"x": -phase_a, "y": -0.20, "z": 0.10},
                    "Head": {"x": -0.12 if is_question else 0.0, "y": head_y * 0.4, "z": head_z},
                    "Spine": {"x": 0.04, "y": phase_a * 0.2, "z": 0.0},
                },
                "nmm": {
                    "eyebrowRaise": eyebrow,
                    "mouthOpen": 0.28 if is_intensified else 0.18,
                    "headShake": 1.0 if is_negated else 0.0,
                },
            },
            {
                "gloss": upper,
                "duration": 0.38,
                "bones": {
                    "RightArm": {"x": -0.55 - phase_b, "y": 0.10, "z": -0.25},
                    "RightForeArm": {"x": 0.05, "y": -0.65 - phase_b, "z": 0.20},
                    "RightHand": {"x": -phase_a, "y": -0.15, "z": 0.10},
                    "LeftArm": {"x": -0.50, "y": -0.15, "z": 0.30},
                    "LeftForeArm": {"x": 0.0, "y": 0.65, "z": -0.20},
                    "LeftHand": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "Head": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "Spine": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "nmm": {
                    "eyebrowRaise": eyebrow * 0.5,
                    "mouthOpen": 0.05,
                    "headShake": 0.0,
                },
            },
        ]

    def parse_to_isl(self, text: str, language: str = "en") -> Dict:
        """
        Transforms spoken input into ISL S-O-V glosses, spatial markers, non-manual markers,
        NIOS 230 word-level structures, sentence types, and 3D avatar bone rotation keyframes.
        """
        normalized_text = self.normalize_multilingual(text, language)
        raw_words_lower = re.findall(r"[a-z0-9']+", normalized_text.lower())
        is_question = "?" in text or any(w in raw_words_lower for w in WH_QUESTIONS)
        has_conditional_if = "if" in raw_words_lower
        is_intensified = any(w in raw_words_lower for w in INTENSIFIERS)

        temporal_bucket: List[str] = []
        subject_bucket: List[str] = []
        object_bucket: List[str] = []
        adjunct_bucket: List[str] = []
        verb_bucket: List[str] = []
        command_bucket: List[str] = []
        negation_bucket: List[str] = []
        wh_bucket: List[str] = []
        svo_tokens: List[SyntaxTokenInfo] = []

        # Check if "have"/"has" is used as main possessive verb (NIOS L10.1: BROTHER/SISTER HAVE?)
        aux_Follows_have = any(
            raw_words_lower[i] in ("have", "has") and i + 1 < len(raw_words_lower) and raw_words_lower[i + 1].endswith(("ed", "en"))
            for i in range(len(raw_words_lower))
        )

        if self.nlp is not None:
            doc: Doc = self.nlp(normalized_text)
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                lower = token.text.lower()
                lemma = token.lemma_.lower() if token.lemma_ != "-PRON-" else lower

                if lower == "if":
                    role = "CONDITIONAL (FACIAL NMM - L8.4/L10.2)"
                elif lower in INTENSIFIERS:
                    role = "INTENSIFIER (FUSED TO NMM/MOVEMENT - L8.4)"
                elif lower in WH_QUESTIONS or lemma in WH_QUESTIONS:
                    wh_bucket.append(lemma.upper())
                    role = "WH-MARKER"
                elif lower in NEGATION_MARKERS or token.dep_ == "neg":
                    negation_bucket.append("NOT")
                    role = "NEGATION"
                elif lower in COMMAND_MODALS:
                    command_bucket.append("MUST")
                    role = "COMMAND-MODAL"
                elif lower in TEMPORAL_MARKERS or token.ent_type_ in ("DATE", "TIME"):
                    temporal_bucket.append(lemma.upper())
                    role = "TIME"
                elif lower in ("have", "has") and not aux_Follows_have:
                    verb_bucket.append("HAVE")
                    role = "POSSESSIVE-VERB (CLAUSE-FINAL - L10.1)"
                elif lower in ISL_OMITTED_STOPWORDS and token.pos_ in ("DET", "ADP", "AUX", "PART"):
                    role = "OMITTED (COPULA/ARTICLE)"
                elif token.dep_ in ("nsubj", "nsubjpass", "csubj", "expl"):
                    mods = [
                        child.lemma_.upper()
                        for child in token.lefts
                        if child.dep_ in ("amod", "compound", "nummod")
                        and child.text.lower() not in ISL_OMITTED_STOPWORDS
                        and child.text.lower() not in INTENSIFIERS
                    ]
                    subject_bucket.extend(mods)
                    subject_bucket.append(lemma.upper())
                    role = "SUBJECT"
                elif token.dep_ in ("dobj", "pobj", "attr", "oprd", "dative"):
                    mods = [
                        child.lemma_.upper()
                        for child in token.lefts
                        if child.dep_ in ("amod", "compound", "nummod")
                        and child.text.lower() not in ISL_OMITTED_STOPWORDS
                        and child.text.lower() not in INTENSIFIERS
                    ]
                    object_bucket.extend(mods)
                    object_bucket.append(lemma.upper())
                    role = "OBJECT"
                elif token.pos_ == "VERB" or token.dep_ == "ROOT":
                    verb_bucket.append(lemma.upper())
                    role = "VERB"
                elif token.dep_ in ("amod", "compound", "nummod") and token.head.dep_ in (
                    "nsubj", "nsubjpass", "dobj", "pobj", "attr"
                ):
                    role = "MODIFIER (ATTACHED)"
                else:
                    if lower not in ISL_OMITTED_STOPWORDS:
                        adjunct_bucket.append(lemma.upper())
                        role = "ADJUNCT"
                    else:
                        role = "OMITTED"

                svo_tokens.append(
                    SyntaxTokenInfo(
                        text=token.text,
                        lemma=lemma,
                        pos=token.pos_,
                        dep=token.dep_,
                        isl_role=role,
                    )
                )
        else:
            words = re.findall(r"[A-Za-z0-9']+", normalized_text)
            found_verb = False
            common_verbs = {
                "explain", "explains", "explained", "teach", "teaches", "calculate",
                "calculates", "solve", "solves", "learn", "learns", "convert",
                "converts", "measure", "measures", "define", "defines", "use", "uses",
                "compute", "computes", "show", "shows", "find", "finds", "differentiate",
                "sit", "sits", "go", "goes", "come", "comes", "eat", "sleep", "like",
                "likes", "pass", "passed", "fail", "finish", "finished", "help", "helps",
            }
            for w in words:
                low = w.lower()
                if low in ("pass", "class", "less", "boss", "chess", "dress"):
                    lemma = low
                elif len(low) > 4 and low not in ("students", "children", "countries", "states"):
                    lemma = re.sub(r"(s|ed|ing)$", "", low)
                else:
                    lemma = low

                if low == "if":
                    role = "CONDITIONAL (FACIAL NMM - L8.4/L10.2)"
                elif low in INTENSIFIERS:
                    role = "INTENSIFIER (FUSED TO NMM/MOVEMENT - L8.4)"
                elif low in WH_QUESTIONS:
                    wh_bucket.append(low.upper())
                    role = "WH-MARKER"
                elif low in NEGATION_MARKERS:
                    negation_bucket.append("NOT")
                    role = "NEGATION"
                elif low in COMMAND_MODALS:
                    command_bucket.append("MUST")
                    role = "COMMAND-MODAL"
                elif low in TEMPORAL_MARKERS:
                    temporal_bucket.append(low.upper())
                    role = "TIME"
                elif low in ("have", "has") and not aux_Follows_have:
                    verb_bucket.append("HAVE")
                    role = "POSSESSIVE-VERB (CLAUSE-FINAL - L10.1)"
                elif low in ISL_OMITTED_STOPWORDS:
                    role = "OMITTED"
                elif low in common_verbs and not found_verb:
                    found_verb = True
                    verb_bucket.append(lemma.upper())
                    role = "VERB"
                elif not found_verb:
                    subject_bucket.append(low.upper())
                    role = "SUBJECT"
                else:
                    object_bucket.append(low.upper())
                    role = "OBJECT"

                svo_tokens.append(
                    SyntaxTokenInfo(
                        text=w,
                        lemma=lemma,
                        pos="TOKEN",
                        dep=role.lower(),
                        isl_role=role,
                    )
                )

        # Construct spatial markers (Locus assignment in signing space - NIOS L10.3)
        spatial_markers: List[Dict] = []
        isl_gloss_sequence: List[str] = []

        # 1. Temporal markers always lead in ISL
        isl_gloss_sequence.extend(temporal_bucket)

        # 2. Subject + Left Spatial Locus (IX-LEFT)
        if subject_bucket:
            isl_gloss_sequence.extend(subject_bucket)
            isl_gloss_sequence.append("IX-LEFT")
            spatial_markers.append(
                {
                    "entity": " ".join(subject_bucket),
                    "locus": "IX-LEFT",
                    "azimuth_deg": -28,
                    "role": "Subject Anchor (NIOS L10.3 Signing Space)",
                }
            )

        # 3. Object + Right Spatial Locus (IX-RIGHT)
        if object_bucket:
            isl_gloss_sequence.extend(object_bucket)
            isl_gloss_sequence.append("IX-RIGHT")
            spatial_markers.append(
                {
                    "entity": " ".join(object_bucket),
                    "locus": "IX-RIGHT",
                    "azimuth_deg": 28,
                    "role": "Object Anchor (NIOS L10.3 Signing Space)",
                }
            )

        # 4. Adjuncts / Modifiers
        isl_gloss_sequence.extend(adjunct_bucket)

        # 5. Verb (S-O-V finality, including clause-final HAVE per NIOS L10.1)
        isl_gloss_sequence.extend(verb_bucket)

        # 6. Clause-final Command Modal (MUST), Negation (NOT), & WH-Question markers (NIOS L10.1/L10.2)
        is_negated = len(negation_bucket) > 0
        is_command = len(command_bucket) > 0 or (not subject_bucket and bool(verb_bucket) and is_negated)
        isl_gloss_sequence.extend(command_bucket)
        isl_gloss_sequence.extend(negation_bucket)
        isl_gloss_sequence.extend(wh_bucket)

        # Determine NIOS Lesson 10.2 Sentence Type
        if is_question:
            if wh_bucket:
                sentence_type = "Interrogative — WH-Question (Question Word Clause-Final - NIOS L10.2)"
            else:
                sentence_type = "Interrogative — Yes/No Question (Clause-Final HAVE/FINISH + Brow Raise - NIOS L10.2)"
        elif is_negated:
            sentence_type = "Negation Sentence (Clause-Final NOT + Head Shake - NIOS L10.2)"
        elif is_command:
            sentence_type = "Order / Command Sentence (Clause-Final MUST - NIOS L10.2)"
        else:
            sentence_type = "Simple Declarative Sentence (S-O-V Topological Order - NIOS L10.2)"

        # Extract NIOS Lesson 9 & 11 Word-Level Structures and Sign Families
        word_level_structures: List[Dict] = []
        sign_families: List[Dict] = []
        for gloss in isl_gloss_sequence:
            if gloss in NIOS_230_MORPHOLOGY:
                info = NIOS_230_MORPHOLOGY[gloss]
                word_level_structures.append(
                    {
                        "gloss": gloss,
                        "isl_compound": info["compound"],
                        "rule": info["category"],
                    }
                )
            if gloss in NIOS_230_SIGN_FAMILIES:
                sign_families.append(
                    {
                        "gloss": gloss,
                        "family": NIOS_230_SIGN_FAMILIES[gloss],
                    }
                )

        # Non-Manual Markers (NMM - NIOS Lesson 8.3, 8.4, 10.2)
        non_manual_markers: List[str] = []
        if has_conditional_if:
            non_manual_markers.append("NMM: Conditional Brow Raise + Clause Pause (No separate 'IF' sign - NIOS L8.4/L10.2)")
        if is_intensified:
            non_manual_markers.append("NMM: Intensified Facial Expression + Amplified Movement ('Very/Extremely' - NIOS L8.4)")
        if is_question:
            if wh_bucket:
                non_manual_markers.append("NMM: Eyebrows Raised + Head Back / Chin Up (ISL WH-Question - NIOS L3.2/L10.2)")
            else:
                non_manual_markers.append("NMM: Eyebrows Raised + Head Tilt Forward (Yes/No Question - NIOS L10.2)")
        if is_negated:
            non_manual_markers.append("NMM: Simultaneous Horizontal Head Shake with Clause-Final NOT (NIOS L8.4/L10.2)")
        if spatial_markers:
            non_manual_markers.append("NMM: Eye Gaze & Pointing Aligned to Signing Space Loci (IX-LEFT -> IX-RIGHT - NIOS L10.3)")

        # Build continuous 3D Avatar Bone Rotation Keyframes
        avatar_keyframes: List[Dict] = []
        for idx, gloss in enumerate(isl_gloss_sequence):
            frames = self._synthesize_procedural_keyframes(
                gloss=gloss,
                index=idx,
                is_question=is_question,
                is_negated=is_negated,
                is_intensified=is_intensified,
            )
            avatar_keyframes.extend(frames)

        # Append neutral rest pose frame at end of utterance
        avatar_keyframes.append(
            {
                "gloss": "REST",
                "duration": 0.40,
                "bones": {
                    "RightArm": {"x": 0.0, "y": 0.0, "z": -1.25},
                    "RightForeArm": {"x": 0.0, "y": -0.15, "z": 0.0},
                    "RightHand": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "LeftArm": {"x": 0.0, "y": 0.0, "z": 1.25},
                    "LeftForeArm": {"x": 0.0, "y": 0.15, "z": 0.0},
                    "LeftHand": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "Head": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "Spine": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "nmm": {"eyebrowRaise": 0.0, "mouthOpen": 0.0, "headShake": 0.0},
            }
        )

        output = ISLParsedOutput(
            original_text=text,
            language=language,
            normalized_english=normalized_text,
            svo_tokens=[asdict(t) for t in svo_tokens],
            isl_sov_gloss=isl_gloss_sequence,
            isl_gloss_string=" ".join(isl_gloss_sequence),
            spatial_markers=spatial_markers,
            non_manual_markers=non_manual_markers,
            avatar_keyframes=avatar_keyframes,
            parser_backend=self.parser_name,
            sentence_type=sentence_type,
            word_level_structures=word_level_structures,
            sign_families=sign_families,
        )
        return asdict(output)
