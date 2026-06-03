from src.skills.reading import ClozeItem, generate_cloze, generate_sentence_cloze
from src.skills.writing import ScrambleItem, generate_scramble, evaluate_sentence
from src.skills.listening import ListenItem, generate_listen_exercise, text_to_audio_features
from src.skills.grammar_ex import (
    GrammarPattern, GrammarExercise,
    extract_patterns, generate_pattern_completion, generate_error_correction,
)
