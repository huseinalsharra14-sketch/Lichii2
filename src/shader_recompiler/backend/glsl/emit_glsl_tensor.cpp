// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Software HMMA (TensorMulAdd) for the GLSL backend.
// This is a line-by-line port of the SPIR-V fallback in
// backend/spirv/emit_spirv_tensor.cpp, so both backends produce the same results.

#include <array>
#include <string_view>

#include "shader_recompiler/backend/glsl/emit_glsl_instructions.h"
#include "shader_recompiler/backend/glsl/glsl_emit_context.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::GLSL {

void EmitContext::DefineSoftwareMatrixFunctions(const IR::Program& program) {
    std::array<bool, 2> needed{};
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::TensorMulAdd || inst.Flags<u32>() == 0) {
                continue;
            }
            const u32 depth = inst.Flags<u32>();
            if (depth != 8 && depth != 16) {
                throw LogicError("Invalid software HMMA depth");
            }
            needed[depth == 16] = true;
        }
    }
    if (!needed[0] && !needed[1]) {
        return;
    }

    // Lane shuffle. NV_shader_thread_shuffle works on 32-wide warps, so the
    // lane index is relative to the warp. Otherwise use ARB_shader_ballot with the
    // absolute subgroup index, as the SPIR-V path does (64-wide AMD waves included).
    if (profile.support_gl_warp_intrinsics) {
        header += "uint HmmaShfl(uint v,uint lane){bool valid;"
                  "return shuffleNV(v,lane&31u,32u,valid);}";
    } else {
        header += "uint HmmaShfl(uint v,uint lane){return readInvocationARB(v,lane);}";
    }

    // Round an f32 to the nearest f16 value (ties to even), kept in f32.
    // Done explicitly rather than with unpackHalf2x16(packHalf2x16(x)), which drivers
    // may optimize away. Half subnormals lie on the exactly representable 2^-24 grid.
    header += "float HmmaRoundHalf(float value){"
              "uint bits=floatBitsToUint(value);"
              "uint magnitude=bits&0x7fffffffu;"
              "uint tie=(magnitude>>13u)&1u;"
              "uint rounded=(magnitude+0xfffu+tie)&(~0x1fffu);"
              "uint normal=rounded>=0x47800000u?0x7f800000u:rounded;"
              "precise float scaled=uintBitsToFloat(magnitude)*16777216.0;"
              "precise float sub_f=roundEven(scaled)*5.9604644775390625e-8;"
              "uint subnormal=floatBitsToUint(sub_f);"
              "uint finite=magnitude>=0x38800000u?normal:subnormal;"
              "uint result=magnitude>=0x7f800000u?magnitude:finite;"
              "return uintBitsToFloat(result|(bits&0x80000000u));}";

    for (u32 variant = 0; variant < 2; ++variant) {
        if (!needed[variant]) {
            continue;
        }
        const u32 pairs = variant ? 8 : 4;
        header += fmt::format(
            "uvec2 SoftHmma{}(uvec4 ma,uvec2 mb,uvec2 mc){{"
            "vec2 c0=unpackHalf2x16(mc.x);"
            "vec2 c1=unpackHalf2x16(mc.y);"
            "precise float acc0=c0.x;precise float acc1=c0.y;"
            "precise float acc2=c1.x;precise float acc3=c1.y;"
            "uint lane=gl_SubGroupInvocationARB;"
            "uint warp_base=lane&(~31u);"
            "uint row_base=lane&28u;"
            "uint col_base=(lane&3u)<<3u;"
            "for(uint pair=0u;pair<{}u;++pair){{"
            "bool upper=pair>=4u;"
            "uint component=pair&3u;"
            "uint a_lane=warp_base+row_base+component;"
            "uint b_lane=warp_base+col_base+component;"
            "uint b=upper?mb.y:mb.x;"
            "vec2 bs0=unpackHalf2x16(HmmaShfl(b,b_lane));"
            "vec2 bs1=unpackHalf2x16(HmmaShfl(b,b_lane+4u));"
            "vec2 as0=unpackHalf2x16(HmmaShfl(upper?ma.z:ma.x,a_lane));"
            "vec2 as1=unpackHalf2x16(HmmaShfl(upper?ma.w:ma.y,a_lane));"
            "acc0=fma(as0.y,bs0.y,fma(as0.x,bs0.x,acc0));"
            "acc1=fma(as0.y,bs1.y,fma(as0.x,bs1.x,acc1));"
            "acc2=fma(as1.y,bs0.y,fma(as1.x,bs0.x,acc2));"
            "acc3=fma(as1.y,bs1.y,fma(as1.x,bs1.x,acc3));"
            // HMMA.F16 rounds after each K=8 group, including the midpoint of K=16.
            // All shuffles above run before this branch.
            "if(component==3u){{"
            "acc0=HmmaRoundHalf(acc0);acc1=HmmaRoundHalf(acc1);"
            "acc2=HmmaRoundHalf(acc2);acc3=HmmaRoundHalf(acc3);"
            "}}"
            "}}"
            "return uvec2(packHalf2x16(vec2(acc0,acc1)),packHalf2x16(vec2(acc2,acc3)));}}",
            variant ? 16 : 8, pairs);
    }
}

void EmitTensorMulAdd(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b,
                      std::string_view c) {
    const u32 depth = inst.Flags<u32>();
    if (depth != 8 && depth != 16) {
        throw NotImplementedException("Unsupported matrix multiply depth");
    }
    ctx.AddU32x2("{}=SoftHmma{}({},{},{});", inst, depth, a, b, c);
}

} // namespace Shader::Backend::GLSL
