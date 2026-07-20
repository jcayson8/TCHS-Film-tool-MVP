import { spawn } from 'child_process';
import sharp from 'sharp';

const COLORS={white:[235,235,235],black:[25,25,25],blue:[35,90,180],navy:[20,45,90],orange:[235,105,25],red:[190,45,45],gray:[130,130,130],silver:[185,185,185],gold:[210,165,45],yellow:[230,205,40],green:[45,130,70],maroon:[105,35,55],purple:[100,55,145]};
const rgb=n=>COLORS[String(n||'').toLowerCase()]||null;
const dist=(r,g,b,c)=>Math.sqrt((r-c[0])**2+(g-c[1])**2+(b-c[2])**2);

function extractFrame(videoPath){return new Promise((resolve,reject)=>{const chunks=[];const p=spawn('ffmpeg',['-loglevel','error','-ss','1','-i',videoPath,'-frames:v','1','-vf','scale=480:-1','-f','image2pipe','-vcodec','png','pipe:1']);p.stdout.on('data',d=>chunks.push(d));p.on('error',reject);p.stderr.on('data',()=>{});p.on('close',code=>code===0?resolve(Buffer.concat(chunks)):reject(new Error('frame extraction failed')));});}

export async function classifyPossession(videoPath,{jerseyColor,helmetColor,homeAway}){
  const targets=[rgb(jerseyColor),rgb(helmetColor)].filter(Boolean);
  if(!targets.length)return {filmSide:'needs_review',confidence:0,reason:'Choose jersey and helmet colors'};
  try{
    const frame=await extractFrame(videoPath);
    const {data,info}=await sharp(frame).removeAlpha().raw().toBuffer({resolveWithObject:true});
    let match=0,left=0,right=0,center=0,top=0,bottom=0; const threshold=92;
    for(let y=0;y<info.height;y+=2){for(let x=0;x<info.width;x+=2){const i=(y*info.width+x)*3;const ok=targets.some(c=>dist(data[i],data[i+1],data[i+2],c)<threshold);if(!ok)continue;match++;if(x<info.width/2)left++;else right++;if(x>info.width*.25&&x<info.width*.75)center++;if(y<info.height/2)top++;else bottom++;}}
    const sampled=Math.ceil(info.width/2)*Math.ceil(info.height/2); const ratio=match/Math.max(1,sampled);
    if(ratio<.008)return {filmSide:'needs_review',confidence:Math.round(ratio*3000),reason:'Uniform colors were not clear enough'};
    const spread=1-Math.abs(left-right)/Math.max(1,match); const centerShare=center/Math.max(1,match); const depthBalance=1-Math.abs(top-bottom)/Math.max(1,match);
    // Experimental visual heuristic. Corrections should be treated as the source of truth.
    const offenseScore=.45*spread+.35*centerShare+.20*depthBalance+(homeAway==='home'?.02:0);
    const confidence=Math.round(Math.min(.69,Math.abs(offenseScore-.52)*2+.50)*100);
    if(confidence<58)return {filmSide:'needs_review',confidence,reason:'The uniform was found, but possession was uncertain'};
    return {filmSide:offenseScore>=.52?'offense':'defense',confidence,reason:'Experimental uniform-position estimate'};
  }catch(e){return {filmSide:'needs_review',confidence:0,reason:e.message};}
}
