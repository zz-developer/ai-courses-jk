import { load, tex, dvi2svg } from "node-tikzjax";
import type { MarkdownTransformContext } from '@slidev/types'
import { defineTransformersSetup } from '@slidev/types'

export async function renderTexToSvg(texString: string): Promise<string> {
  await load();
  const dvi = await tex(texString);
  return dvi2svg(dvi);
}

function renderer(ctx: MarkdownTransformContext) {
    // Check "```tikz" block but should not be "```tikz source" in one RegExp
    const reg = /```tikz(?!\s+source\b)/gm;
    ctx.s.replace(reg, (full, options = '', code = '') => {
        const svg = renderTexToSvg(code);
        return `<div class="tikz">${svg}</div>`;
    })
}