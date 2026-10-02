"use strict";
class EmbeddingView {
  constructor(canvas, onSelect) {
    this.canvas=canvas;this.ctx=canvas.getContext('2d');this.onSelect=onSelect;
    this.yaw=.4;this.pitch=.25;this.zoom=1;this.points=[];this.query=null;this.selected=null;
    this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(canvas);
    canvas.onpointerdown=e=>{this.drag={x:e.clientX,y:e.clientY,moved:false};canvas.setPointerCapture(e.pointerId);};
    canvas.onpointermove=e=>{
      if(this.drag){const dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y;this.yaw+=dx*.008;this.pitch+=dy*.008;this.drag={x:e.clientX,y:e.clientY,moved:true};this.draw();}
      else {const rect=canvas.getBoundingClientRect();const hit=this.nearest(e.clientX-rect.left,e.clientY-rect.top);canvas.title=hit?.text?.slice(0,190)||'Drag to rotate. Scroll to zoom. Select a memory.';}
    };
    canvas.onpointerup=e=>{if(this.drag&&!this.drag.moved){const r=canvas.getBoundingClientRect();const p=this.nearest(e.clientX-r.left,e.clientY-r.top);if(p){this.selected=p.id;this.onSelect(p);this.draw();}}this.drag=null;};
    canvas.onwheel=e=>{e.preventDefault();this.zoom=Math.min(4,Math.max(.4,this.zoom*Math.exp(-e.deltaY*.001)));this.draw();};
  }
  update(data){this.points=data.points;this.query=data.query;this.selected=this.points[0]?.id;this.draw();if(this.points[0])this.onSelect(this.points[0]);}
  nearest(x,y){return this.screen?.filter(p=>Math.hypot(p.x-x,p.y-y)<15).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];}
  draw(){
    const canvas=this.canvas,box=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
    if(!box.width)return;canvas.width=box.width*dpr;canvas.height=box.height*dpr;
    const ctx=this.ctx;ctx.scale(dpr,dpr);const w=box.width,h=box.height;
    ctx.clearRect(0,0,w,h);const all=[...this.points,...(this.query?[this.query]:[])];
    const extent=Math.max(.3,...all.flatMap(p=>p.position.map(Math.abs)));
    const transform=position=>{const [a,b,c]=position;const x=a*Math.cos(this.yaw)+c*Math.sin(this.yaw),z=-a*Math.sin(this.yaw)+c*Math.cos(this.yaw);const y=b*Math.cos(this.pitch)-z*Math.sin(this.pitch);const depth=b*Math.sin(this.pitch)+z*Math.cos(this.pitch);const scale=Math.min(w,h)*.32/extent*this.zoom/(1+depth/extent*.13);return{x:w/2+x*scale,y:h/2-y*scale,z:depth};};
    ctx.lineWidth=1;ctx.strokeStyle='#2d473d';ctx.fillStyle='#92a69b';ctx.font='10px system-ui';
    for(let axis=0;axis<3;axis++){const p=[0,0,0],q=[0,0,0];p[axis]=-extent;q[axis]=extent;const a=transform(p),b=transform(q);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.fillText('PC'+(axis+1),b.x+5,b.y);}
    this.screen=this.points.map(p=>({...p,...transform(p.position)})).sort((a,b)=>a.z-b.z);
    const query=this.query?transform(this.query.position):null;
    for(const p of this.screen){
      if(query&&p.id===this.selected){ctx.strokeStyle='#b7ff5a88';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(query.x,query.y);ctx.lineTo(p.x,p.y);ctx.stroke();ctx.setLineDash([]);}
      ctx.beginPath();ctx.arc(p.x,p.y,p.id===this.selected?7:4,0,Math.PI*2);ctx.fillStyle=p.id===this.selected?'#b7ff5a':'#52e2bd99';ctx.fill();
      if(p.id===this.selected){ctx.fillStyle='#b7ff5a';ctx.fillText('Selected memory',p.x+11,p.y-8);}
    }
    if(query){ctx.fillStyle='#ffcc66';ctx.beginPath();ctx.moveTo(query.x,query.y-9);ctx.lineTo(query.x+8,query.y+6);ctx.lineTo(query.x-8,query.y+6);ctx.closePath();ctx.fill();ctx.fillText('QUERY',query.x+12,query.y+3);}
    if(!this.points.length){ctx.fillStyle='#92a69b';ctx.textAlign='center';ctx.fillText('Add a memory or search the web to populate this space.',w/2,h/2);ctx.textAlign='left';}
  }
  destroy(){this.resize.disconnect();}
}
function vectorHeatmap(canvas,vector){
  const w=canvas.clientWidth,dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=108*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const max=Math.max(.01,...vector.map(Math.abs)),cols=32,rows=Math.ceil(vector.length/cols);
  for(let i=0;i<vector.length;i++){const v=vector[i]/max;ctx.fillStyle=v>=0?`rgba(82,226,189,${.15+.85*v})`:`rgba(109,152,255,${.15-.85*v})`;ctx.fillRect(i%cols*w/cols,Math.floor(i/cols)*108/rows,w/cols-1,108/rows-1);}
}
function chartSVG(rows,agents,field,title,cumulative=false,setup={}){
  const colors={custom:'#b7ff5a',memorizz:'#52e2bd',naive:'#ffcc66',decisions:'#ad9bff'};
  const series=agents.map(agent=>{let sum=cumulative?Number(setup[agent]?.[field]||0):0,unknown=cumulative&&setup[agent]?.[field]===null;return{agent,points:rows.filter(r=>r.agent===agent).map(r=>{const value=r[field];if(value===null||value===undefined){if(cumulative)unknown=true;return{x:r.turn,y:null};}if(unknown)return{x:r.turn,y:null};sum+=Number(value);return{x:r.turn,y:cumulative?sum:Number(value)};})};});
  const maxX=Math.max(1,...rows.map(r=>r.turn)),maxY=Math.max(field.includes('usd')?.000001:1,...series.flatMap(s=>s.points.filter(p=>p.y!==null).map(p=>p.y)))*1.15;
  const w=580,h=230,left=65,right=18,top=20,bottom=38;const x=t=>left+(t-1)/Math.max(1,maxX-1)*(w-left-right),y=v=>h-bottom-v/maxY*(h-top-bottom);
  let svg=`<svg viewBox="0 0 ${w} ${h}" class="experiment-chart" role="img" aria-label="${escapeHTML(title)}">`;
  for(let tick=0;tick<=4;tick++){const value=maxY*tick/4,yy=y(value);svg+=`<line x1="${left}" x2="${w-right}" y1="${yy}" y2="${yy}" stroke="var(--line)"/><text x="${left-8}" y="${yy+4}" text-anchor="end" fill="var(--muted)" font-size="10">${field.includes('usd')?'$'+value.toFixed(4):value.toFixed(field==='seconds'?1:0)}</text>`;}
  for(const s of series){svg+=`<g data-chart-agent="${s.agent}">`;let path='',gap=true;for(const p of s.points){if(p.y===null){gap=true;continue;}path+=(gap?'M':'L')+x(p.x)+' '+y(p.y)+' ';gap=false;}svg+=`<path d="${path}" fill="none" stroke="${colors[s.agent]}" stroke-width="2.5"/>`;for(const p of s.points)if(p.y!==null)svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="4" fill="${colors[s.agent]}"><title>${escapeHTML(s.agent)} · turn ${p.x}: ${p.y}</title></circle>`;svg+='</g>';}
  for(let t=1;t<=maxX;t++)svg+=`<text x="${x(t)}" y="${h-16}" text-anchor="middle" fill="var(--muted)" font-size="10">${t}</text>`;
  return svg+'</svg>';
}
