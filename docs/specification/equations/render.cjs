// Regenerate SVGs: install mathjax-full@3.2.2 in a separate tools directory,
// set NODE_PATH to its node_modules, then node render.cjs.
const fs = require('fs');
const path = require('path');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', {
  InputJax: new TeX({packages: AllPackages}),
  OutputJax: new SVG({fontCache: 'none'})
});
const manifest = JSON.parse(fs.readFileSync(path.join(__dirname,'manifest.json'),'utf8'));
for (const item of manifest) {
  const raw = fs.readFileSync(path.join(__dirname,item.file+'.tex'),'utf8');
  let tex = raw.replace(/\\tag\{[^}]+\}/g,'').trim();
  // Presentation-only line breaks keep wide equations legible on phones.
  if (/[,;]\\q(?:quad|uad)/.test(tex)) {
    tex = '\\begin{gathered}' + tex.replace(/([,;])\\q(?:quad|uad)\s*/g, '$1\\\\') + '\\end{gathered}';
  }
  if (item.label === '9-2' || item.label === '8-5') {
    tex = '\\begin{gathered}' + tex.replace(/\n(?=[+\-=])/g, '\\\\') + '\\end{gathered}';
  }
  const node = doc.convert(tex,{display:true});
  const markup = adaptor.outerHTML(node);
  if (/data-mjx-error|data-mml-node="merror"/.test(markup)) throw new Error(item.file+': '+markup);
  const svg = markup.match(/<svg\b[^>]*>([\s\S]*)<\/svg>/);
  const view = markup.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
  const [x,y,w,h] = view;
  const pad=600, labelWidth=item.label ? 3800 : 0;
  const width=w+pad*2+labelWidth, height=Math.max(h+pad*2,2400);
  const label=item.label ? `<text x="${width-pad}" y="${height/2+330}" text-anchor="end" font-family="Arial, sans-serif" font-size="900" fill="#111111">(${item.label})</text>` : '';
  const result = `<svg xmlns="http://www.w3.org/2000/svg" width="${Math.ceil(width*0.018)}" height="${Math.ceil(height*0.018)}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${item.label ? 'Equation '+item.label : 'Auxiliary equation'}"><rect width="100%" height="100%" rx="250" fill="#ffffff"/><g fill="#111111" stroke="#111111" transform="translate(${pad-x} ${(height-h)/2-y})">${svg[1].replace(/currentColor/g,'#111111')}</g>${label}</svg>\n`;
  if (/href=|<script|<foreignObject/.test(result)) throw new Error('Unexpected dependency: '+item.file);
  fs.writeFileSync(path.join(__dirname,item.file+'.svg'),result);
}
console.log('Rendered '+manifest.length+' standalone SVG equations');
