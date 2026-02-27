# PSOLA Voice Disguise Design

## Problem

The current voice disguising sounds robotic and metallic, making speech difficult to listen to. Root causes:

1. **Naive phase vocoder bin-shifting** creates spectral gaps between harmonics
2. **Cascaded STFT passes** (pitch shift -> formant shift -> spectral jitter) compound windowing artifacts
3. **4-semitone constant shift** pushes voices into unnatural registers
4. **Spectral jitter** adds audible noise-like artifacts

## Solution

Replace the phase vocoder pipeline with PSOLA (Pitch-Synchronous Overlap-Add), a time-domain technique that manipulates individual pitch periods in the waveform directly. This preserves the natural spectral shape of speech while changing pitch.

## Core Algorithm: PSOLA

1. **Pitch Detection** — Analyze input in overlapping ~30ms frames. Compute autocorrelation to find fundamental period (F0). Classify each frame as voiced or unvoiced.

2. **Pitch Mark Placement** — Walk the waveform placing marks at each glottal pulse (one per pitch period). Voiced segments use detected period; unvoiced segments use fixed ~5ms spacing.

3. **Analysis Windows** — Extract Hann-windowed segments centered on each pitch mark, sized to 2x local pitch period.

4. **Synthesis** — Place new pitch marks at desired output pitch period spacing. Overlap-add windowed segments at new positions.

Key properties:
- Formants naturally preserved (reuses original waveform shape)
- No spectral gaps (no FFT bin shifting)
- Coherent harmonic structure maintained

## Three-Tier Architecture

### Low: PSOLA Pitch Shift Only
- Constant 2.5-semitone shift
- Formants naturally preserved
- Defeats casual recognition

### Medium: PSOLA + Formant Warp
- PSOLA pitch shift (2.5 semitones)
- Single-pass LPC spectral envelope warp (ratio 1.15) applied after PSOLA
- Changes both pitch and vowel quality

### High: PSOLA + Formant Warp + Pitch Contour Warping
- Per-frame pitch contour modification (shift varies smoothly over time via random walk)
- Breaks prosody patterns used in forensic voice analysis
- Formant warp (ratio 1.15)
- Replaces spectral jitter — achieves anti-forensic goal without audible artifacts

## Implementation Structure

### Files modified

- `pitch_shift_transformer.rs` — Rewritten with PSOLA (pitch detection, marking, overlap-add synthesis). Same struct, same trait, new internals. Pitch detector and pitch marker are private helpers within this file.
- `formant_shift_transformer.rs` — Simplified to standalone single-pass LPC spectral envelope warp. No longer wraps PitchShiftTransformer internally.
- `voice_morph_transformer.rs` — Rewritten. Composes PSOLA (with pitch contour warping) + formant warp. No spectral jitter.

### Constants

| Constant | Old | New |
|----------|-----|-----|
| `DEFAULT_SEMITONES` | 4.0 | 2.5 |
| `DEFAULT_FORMANT_SHIFT_RATIO` | 1.2 | 1.15 |
| `DEFAULT_JITTER_AMOUNT` | 0.15 | Removed |
| `SPECTRAL_SMOOTHING` | 0.3 | Removed |
| Pitch contour warp range | — | +/-0.5 semitones random walk |

### No changes to
- Domain traits (`AudioTransformer`)
- `AudioSegment`, `ProcessAudioUseCase`
- CLI/Desktop argument handling (Low/Medium/High labels unchanged)
- Word censoring / bleep functionality

### Dependencies
- No new external dependencies required
- Pure Rust implementation
- No runtime downloads

## Testing

Existing test assertions (changes audio, preserves length, preserves amplitude range) remain valid. New tests added for:
- Pitch detection accuracy on synthetic signals
- Voiced/unvoiced classification
- Pitch contour warping variation over time
