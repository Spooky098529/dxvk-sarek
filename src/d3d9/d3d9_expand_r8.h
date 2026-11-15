#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dxvk {

  /**
   * \brief Helpers for d3d9.expandR8
   *
   * The D3D9 mapping keeps one byte per texel while the Vulkan
   * image is R8G8B8A8. Uploads replicate each byte into all four
   * channels so that the format swizzles return the same values
   * as with R8, readbacks take the R channel.
   */

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

    for (uint32_t z = 0; z < Depth; z++) {
      for (uint32_t y = 0; y < Height; y++) {
        const uint8_t* srcRow = src + z * SrcSlicePitch + y * SrcRowPitch;

        for (uint32_t x = 0; x < Width; x++) {
          uint32_t texel = uint32_t(srcRow[x]) * 0x01010101u;
          std::memcpy(dst, &texel, sizeof(texel));
          dst += sizeof(texel);
        }
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
        uint8_t* dstRow = dst + z * DstSlicePitch + y * DstRowPitch;

        for (uint32_t x = 0; x < Width; x++) {
          dstRow[x] = src[0];
          src += 4;
        }
      }
    }
  }

}
