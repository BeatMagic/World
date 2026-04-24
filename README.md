# WORLD - a high-quality speech analysis, manipulation and synthesis system

WORLD is free software for high-quality speech analysis, manipulation and synthesis.
It can estimate Fundamental frequency (F0), aperiodicity and spectral envelope and also generate the speech like input speech with only estimated parameters.

This source code is released under the modified-BSD license.
There is no patent in all algorithms in WORLD.

## Performance optimization (AVX-512 + AOCL-FFTW)

This fork adds an optional AVX-512 path to the CompositeF0 analysis pipeline
(DIO + Harvest + ZCR + Gaussian filter + voicing merge), plus build-time
support for [AOCL-FFTW](https://github.com/amd/amd-fftw) — AMD's fork of FFTW
with AVX-512 codelets tuned for Zen4 / Zen5.

The upstream WORLD library is unchanged in structure; the optimization is
layered behind a thin runtime dispatcher. On non-AVX-512 CPUs (or when built
without AVX-512 support), the library falls back to the scalar implementation
automatically — no source changes needed.

### Measured speedup (AMD Ryzen 9 9950X3D / znver5)

CompositeF0 on a 44.1 kHz 12.7 s speech WAV (hop=256, 10-run best-of):

| Configuration                          | FFT backend              | WORLD tier  | total_ms | vs baseline |
| :------------------------------------- | :----------------------- | :---------- | -------: | ----------: |
| baseline                               | upstream FFTW 3.3.10 AVX2 | scalar      |   720.1  |         —   |
| WORLD AVX-512 (K1-K10)                 | upstream FFTW 3.3.10 AVX2 | AVX-512     |   693.8  |      -3.6%  |
| AOCL-FFTW AVX-512                      | AOCL-FFTW 5.2 AVX-512    | scalar      |   618.1  |     -14.2%  |
| **AOCL-FFTW AVX-512 + WORLD AVX-512**  | AOCL-FFTW 5.2 AVX-512    | AVX-512     | **577.7**|   **-19.8%**|

F0 output matches the scalar reference element-wise (max absolute diff
`~2e-12 Hz`, zero voicing-decision flips).

### Runtime dispatcher

The dispatcher (`src/simd/simd_dispatch.h`) detects AVX-512F/DQ/BW/VL at
program start and binds one function pointer per SIMD kernel. No
re-compilation or feature flag needed at call sites.

Environment-variable overrides (useful for debugging and bisection):

- `WORLD_FORCE_SIMD_TIER=scalar` — force scalar tier even on AVX-512 CPUs.
- `WORLD_FORCE_SIMD_TIER=avx512` — force AVX-512 tier (no-op if the CPU or
  build lacks support).
- `WORLD_AVX512_DISABLE=K1,K3` or `=ALL` — selectively fall back specific
  kernels (K1..K10) to scalar, to isolate a regression.

### Building with AVX-512 + AOCL-FFTW

On Windows we recommend **MSYS2 MinGW-w64** (GCC). MSVC works but does not
support the AVX-512 subset flags that FFTW's own configure expects, so the
FFTW build itself needs MinGW/GCC or WSL.

1. Build the FFT backend once. AOCL-FFTW 5.2 source lives at
   <https://github.com/amd/amd-fftw> (tag `5.2`). On MSYS2 MinGW-w64:

   ```bash
   ./bootstrap.sh
   # NOTE: do NOT pass --enable-dynamic-dispatcher on MinGW — it requires
   # GNU ifunc, which PE-COFF does not support. Omitting it compiles the
   # AVX-512 codelets statically (no runtime dispatch), which is fine for
   # a Zen4/Zen5-targeted build.
   ./configure --prefix=/d/dev/fftw-aocl/install \
       --enable-shared --disable-static \
       --enable-sse2 --enable-avx --enable-avx2 --enable-avx512 \
       --enable-amd-opt \
       --disable-dependency-tracking --with-our-malloc \
       CFLAGS="-O3 -mtune=znver5 -march=znver5"
   make -j && make install
   ```

   On Linux, add `--enable-dynamic-dispatcher` for a portable binary with
   runtime codelet selection.

2. Point CMake at the install prefix via `WORLD_AOCL_FFTW_DIR`:

   ```bash
   cmake -B build -G "MinGW Makefiles" \
       -DCMAKE_BUILD_TYPE=Release \
       -DWORLD_AOCL_FFTW_DIR=/d/dev/fftw-aocl/install
   cmake --build build -j
   ```

   CMake status line should print `FFT backend: AOCL-FFTW at ...`. If
   `WORLD_AOCL_FFTW_DIR` is unset, CMake falls back to system FFTW3, then
   to the bundled Ooura FFT.

3. Runtime DLL: ensure the FFTW DLL directory is on `PATH` (Windows) or
   `LD_LIBRARY_PATH` (Linux).

### pyworld integration notes

pyworld's `setup.py` invokes the WORLD C/C++ sources directly; to pick up
AOCL-FFTW from Python the user needs to set `WORLD_AOCL_FFTW_DIR` (or
`AOCL_ROOT`) in the build environment and adjust `setup.py` to compile the
FFTW backend path (`src/fft_fftw3.cpp`) with the correct include / link
flags. This is out of scope for this fork; a companion pyworld patch is
tracked separately.

---

## Introduction of WORLD family (2025/02/21)

I introduce useful software in WORLD. If you want to introduce your project in WORLD, please contact me.

PyWorldVocoder (https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) is a Python wrapper for World Vocoder.

Python-WORLD (https://github.com/tuanad121/Python-WORLD) is line-by-line implementation of WORLD vocoder (Matlab, C++) in python.

world-class (https://github.com/yukara-ikemiya/world-class) is a C++ library of WORLD.

World.JS (https://github.com/GloomyGhost-MosquitoSeal/World.JS) is a JavaScript Wrapper for World Vocoder.

World.NET (https://github.com/aqtq314/World.NET) is a C# Wrapper for World Vocoder.

WorldInApple (https://github.com/fuziki/WorldInApple) is a Swift wrapper for World Vocoder.

DotnetWorld (https://github.com/yamachu/DotnetWorld) is a C# wrapper for WORLD.

JA-WORLD (https://gitlab.com/f-matano44/world-for-java) is an independent Java port of WORLD vocoder.

The Speech Signal Processing Toolkit ([SPTK](https://github.com/sp-nitech/SPTK)) wraps WORLD as UNIX-like commands.
- DIO and Harvest -> `pitch`
  - https://sp-nitech.github.io/sptk/4.3/main/pitch.html
- D4C -> `ap`
  - https://sp-nitech.github.io/sptk/4.3/main/ap.html
- CheapTrick -> `pitch_spec`
  - https://sp-nitech.github.io/sptk/4.3/main/pitch_spec.html
- Synthesis -> `world_synth`
  - https://sp-nitech.github.io/sptk/4.3/main/world_synth.html

[diffsptk](https://github.com/sp-nitech/diffsptk) implements some of the WORLD components within the PyTorch framework.
- D4C -> `Aperiodicity`
  - https://sp-nitech.github.io/diffsptk/2.5.0/modules/ap.html
- CheapTrick -> `PitchAdaptiveSpectralAnalysis`
  - https://sp-nitech.github.io/diffsptk/2.5.0/modules/pitch_spec.html

Note: To avoid making the project complicated, I decided not to merge it to my repository and introduce your project here. The other reason is that I can't support some computer languages.

## References
When you cite the latest version of WORLD in your paper, please use the sentence "WORLD \[1\] (D4C edition [2])" and cite the following papers.  
[1] M. Morise, F. Yokomori, and K. Ozawa: WORLD: a vocoder-based high-quality speech synthesis system for real-time applications, IEICE transactions on information and systems, vol. E99-D, no. 7, pp. 1877-1884, 2016. https://www.jstage.jst.go.jp/article/transinf/E99.D/7/E99.D_2015EDP7457/_article  
[2] M. Morise: D4C, a band-aperiodicity estimator for high-quality speech synthesis, Speech Communication, vol. 84, pp. 57-65, Nov. 2016. http://www.sciencedirect.com/science/article/pii/S0167639316300413  
If you used the real-time synthesis function, you can refer the following reference.  
[3] M. Morise: Implementation of sequential real-time waveform generator for high-quality vocoder, in Proc. APSIPA ASC 2020, pp. 821-825, Online, Dec. 7-10, 2020. http://www.apsipa.org/proceedings/2020/pdfs/0000821.pdf  

In CheapTrick, you can refer the following references.  
[4] M. Morise: CheapTrick, a spectral envelope estimator for high-quality speech synthesis, Speech Communication, vol. 67, pp. 1-7, March 2015. http://www.sciencedirect.com/science/article/pii/S0167639314000697  
[5] M. Morise: Error evaluation of an F0-adaptive spectral envelope estimator in robustness against the additive noise and F0 error, IEICE transactions on information and systems, vol. E98-D, no. 7, pp. 1405-1408, July 2015.  

In DIO, you can refer the following reference.  
[6] M. Morise, H. Kawahara and H. Katayose: Fast and reliable F0 estimation method based on the period extraction of vocal fold vibration of singing voice and speech, AES 35th International Conference, CD-ROM Proceeding, Feb. 2009.

In Harvest, you can refer the following reference.  
[7] M. Morise: Harvest: A high-performance fundamental frequency estimator from speech signals, in Proc. INTERSPEECH 2017, pp. 2321–2325, 2017. https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.html

In the codec of spectral envelope, you can refer the following reference.  
[8] M. Morise, G. Miyashita and K. Ozawa: Low-dimensional representation of spectral envelope without deterioration for full-band speech analysis/synthesis system, in Proc. INTERSPEECH 2017, pp. 409-413, 2017. http://www.isca-speech.org/archive/Interspeech_2017/abstracts/0067.html

A paper was published to demonstrate that the current version of WORLD was superior to the similar vocoders in the sound quality of re-synthesized speech. This paper also includes the detailed information in the D4C LoveTrain used in the latest version.  
[9] M. Morise and Y. Watanabe: Sound quality comparison among high-quality vocoders by using re-synthesized speech, Acoust. Sci. & Tech., vol. 39, no. 3, pp. 263-265, May 2018. https://www.jstage.jst.go.jp/article/ast/39/3/39_E1779/_article/-char/en
