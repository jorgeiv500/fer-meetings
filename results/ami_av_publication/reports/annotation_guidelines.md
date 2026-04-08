# Annotation Guidelines

## Objective

Assign a three-class **observable valence** label to each AMI close-up clip:

- `negative`
- `neutral`
- `positive`

The task is to label **visible facial affect in the clip**, not to infer internal mood, personality, or long-term psychological state.

## Unit of Annotation

- One row equals one clip.
- Each clip should be judged as a whole, not by a single frame.
- Use the full clip playback and the thumbnail strip together.

## Label Definitions

### `positive`

Use when the clip shows clearly positive observable affect, for example:

- smiling or laughter-like expression
- relaxed positive mouth/eye configuration
- visible pleasant reaction sustained over the clip

### `neutral`

Use when the clip shows:

- no clear positive or negative facial affect
- low-intensity or ambiguous expression
- conversational face without reliable affective polarity

### `negative`

Use when the clip shows clearly negative observable affect, for example:

- frown, tension, displeasure
- visible frustration, sadness-like or aversive expression
- sustained negative observable valence over the clip

## Exclusion Rules

Mark `exclude_from_gold=true` when:

- the face is not visible enough for judgment
- the clip is dominated by pose/occlusion or severe blur
- the target affect is too ambiguous even after review
- the clip should not be used in the final gold set

## Annotation Procedure

### Single-rater minimum

- fill `gold_label`
- add short justification in `notes` when uncertain

### Publication-grade preferred workflow

- `rater_1_label` and `rater_2_label` completed independently
- if both agree, final label can be resolved automatically
- if they disagree, fill `adjudicated_label`
- keep `gold_label` empty unless a final resolved label is intentionally set

## Tie-Breaking Principles

- prefer `neutral` when affect is weak or mixed
- do not over-interpret brief mouth motion as positive
- do not use conversational context to infer hidden state unless the face itself supports it

## What To Avoid

- inferring intent, stress, deception, or personality
- using model suggestions as the final label without human review
- treating low-confidence clips as strong affective evidence
