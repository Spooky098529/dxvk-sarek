#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "util_bit.h"

#ifdef DXVK_ARCH_X86
  #include <emmintrin.h>
#endif

namespace dxvk {

  /**
   * \brief Helpers for d3d9.expandR8 and d3d11.expandR8
   *
   * The application sees one byte per texel while the Vulkan
   * image is R8G8B8A8. Uploads replicate each byte into all four
   * channels so that the format swizzles return the same values
   * as with R8, readbacks take the R channel.
   *
   * On x86, rows are processed 16 texels at a time with SSE2,
   * the remaining texels use the scalar path.
   */

  /**
   * \brief Expands one row of R8 texels to R8G8B8A8
   *
   * \param [out] pDst Destination, \c Width * 4 bytes
   * \param [in] pSrc Source, \c Width bytes
   * \param [in] Width Texel count
   */
  inline void expandR8ToRGBA8Row(
          uint8_t*  pDst,
    const uint8_t*  pSrc,
          uint32_t  Width) {
    uint32_t x = 0;

#ifdef DXVK_ARCH_X86
    for (; x + 16 <= Width; x += 16) {
      // abcd... -> aabbccdd... -> aaaabbbbccccdddd...
      __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pSrc + x));
      __m128i lo  = _mm_unpacklo_epi8(src, src);
      __m128i hi  = _mm_unpackhi_epi8(src, src);

      __m128i* dst = reinterpret_cast<__m128i*>(pDst + 4 * x);
      _mm_storeu_si128(dst + 0, _mm_unpacklo_epi16(lo, lo));
      _mm_storeu_si128(dst + 1, _mm_unpackhi_epi16(lo, lo));
      _mm_storeu_si128(dst + 2, _mm_unpacklo_epi16(hi, hi));
      _mm_storeu_si128(dst + 3, _mm_unpackhi_epi16(hi, hi));
    }
#endif

    for (; x < Width; x++) {
      uint32_t texel = uint32_t(pSrc[x]) * 0x01010101u;
      std::memcpy(pDst + 4 * x, &texel, sizeof(texel));
    }
  }


  /**
   * \brief Packs one row of R8G8B8A8 texels to R8
   *
   * Keeps the R channel of each texel.
   * \param [out] pDst Destination, \c Width bytes
   * \param [in] pSrc Source, \c Width * 4 bytes
   * \param [in] Width Texel count
   */
  inline void packRGBA8ToR8Row(
          uint8_t*  pDst,
    const uint8_t*  pSrc,
          uint32_t  Width) {
    uint32_t x = 0;

#ifdef DXVK_ARCH_X86
    const __m128i mask = _mm_set1_epi32(0xFF);

    for (; x + 16 <= Width; x += 16) {
      // Keep the low byte of each dword, then narrow 32 -> 16 -> 8 bits.
      // Values are at most 255, so the saturating packs are exact.
      const __m128i* src = reinterpret_cast<const __m128i*>(pSrc + 4 * x);
      __m128i a = _mm_and_si128(_mm_loadu_si128(src + 0), mask);
      __m128i b = _mm_and_si128(_mm_loadu_si128(src + 1), mask);
      __m128i c = _mm_and_si128(_mm_loadu_si128(src + 2), mask);
      __m128i d = _mm_and_si128(_mm_loadu_si128(src + 3), mask);

      __m128i ab = _mm_packs_epi32(a, b);
      __m128i cd = _mm_packs_epi32(c, d);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + x), _mm_packus_epi16(ab, cd));
    }
#endif

    for (; x < Width; x++)
      pDst[x] = pSrc[4 * x];
  }


  /**
   * \brief Expands pitched R8 data to tightly packed R8G8B8A8
   *
   * \param [out] pDst Destination, \c Width * \c Height * \c Depth * 4 bytes
   * \param [in] pSrc First source texel
   * \param [in] SrcRowPitch Source row pitch in bytes
   * \param [in] SrcSlicePitch Source slice pitch in bytes
   */
  inline void expandR8ToRGBA8(
          void*     pDst,
    const void*     pSrc,
          uint32_t  Width,
          uint32_t  Height,
          uint32_t  Depth,
          size_t    SrcRowPitch,
          size_t    SrcSlicePitch) {
    auto dst = reinterpret_cast<uint8_t*>(pDst);
    auto src = reinterpret_cast<const uint8_t*>(pSrc);

    // Tightly packed source, process everything as one long row
    size_t count = size_t(Width) * Height * Depth;

    if ((Height == 1 || SrcRowPitch == Width)
     && (Depth == 1 || SrcSlicePitch == size_t(Width) * Height)
     && count <= UINT32_MAX) {
      expandR8ToRGBA8Row(dst, src, uint32_t(count));
      return;
    }

    for (uint32_t z = 0; z < Depth; z++) {
      for (uint32_t y = 0; y < Height; y++) {
        expandR8ToRGBA8Row(dst, src + z * SrcSlicePitch + y * SrcRowPitch, Width);
        dst += size_t(Width) * 4;
      }
    }
  }


  /**
   * \brief Packs tightly packed R8G8B8A8 data to pitched R8
   *
   * Keeps the R channel of each texel.
   * \param [out] pDst First destination texel
   * \param [in] pSrc Source, \c Width * \c Height * \c Depth * 4 bytes
   * \param [in] DstRowPitch Destination row pitch in bytes
   * \param [in] DstSlicePitch Destination slice pitch in bytes
   */
  inline void packRGBA8ToR8(
          void*     pDst,
    const void*     pSrc,
          uint32_t  Width,
          uint32_t  Height,
          uint32_t  Depth,
          size_t    DstRowPitch,
          size_t    DstSlicePitch) {
    auto dst = reinterpret_cast<uint8_t*>(pDst);
    auto src = reinterpret_cast<const uint8_t*>(pSrc);

    for (uint32_t z = 0; z < Depth; z++) {
      for (uint32_t y = 0; y < Height; y++) {
        packRGBA8ToR8Row(dst + z * DstSlicePitch + y * DstRowPitch, src, Width);
        src += size_t(Width) * 4;
      }
    }
  }

}
